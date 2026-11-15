/**
 * @file providers.ts
 * @description Parses provider briefs, probes endpoints, builds model catalogs, and manages capabilities.
 * Implements robust JSON document extraction, wire format normalization (OpenAI, Anthropic, Gemini),
 * zero-pricing inference, and requirement matching for the make.fun free-model-aggregator-01 entry.
 *
 * Brief documents supported:
 * - Providers Directory (schema: free-model-aggregator-01/providers/v1)
 * - Ratings File (schema: free-model-aggregator-01/ratings/v1)
 * - Workload File (schema: free-model-aggregator-01/workload/v1)
 *
 * Probe rules per wire:
 * - OpenAI / Anthropic: GET /v1/models with standard or custom headers.
 * - Gemini: GET /v1beta/models with x-goog-api-key.
 * Zero-pricing inference automatically detects free models when pricing fields are zero or absent.
 */

import { readFileSync } from 'node:fs';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type Json = Record<string, unknown>;

export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

export interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface ProviderDirectory {
  schema: string;
  providers: ProviderDirectoryEntry[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface CatalogReport {
  providers: CatalogProvider[];
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsFile {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadFile {
  schema: string;
  manifest: WorkloadRequest[];
}

export interface Brief {
  directory: ProviderDirectory;
  ratings: RatingsFile | null;
  workload: WorkloadFile | null;
  error: string | null;
}

export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: ProviderDirectoryEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export function joinBase(envUrl: string, base: string): string {
  const trimmedBase = (base || '').trim();
  const trimmedEnv = (envUrl || '').trim().replace(/\/+$/, '');
  if (!trimmedBase) {
    return trimmedEnv;
  }
  if (trimmedBase.startsWith('http://') || trimmedBase.startsWith('https://')) {
    return trimmedBase.replace(/\/+$/, '');
  }
  const cleanBase = trimmedBase.replace(/^\/+/, '').replace(/\/+$/, '');
  return cleanBase ? `${trimmedEnv}/${cleanBase}` : trimmedEnv;
}

export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  const k = key || '';
  if (!k) return {};
  switch (auth) {
    case 'bearer':
      return { authorization: `Bearer ${k}` };
    case 'x-api-key':
      return { 'x-api-key': k };
    case 'x-goog-api-key':
      return { 'x-goog-api-key': k };
    case 'none':
    default:
      return {};
  }
}

export function ratingOfModel(ratings: RatingsFile | null, providerId: string, modelId: string): RatingRow | null {
  if (!ratings || !ratings.ratings) return null;
  for (const r of ratings.ratings) {
    if (r.provider === providerId && r.model === modelId) {
      return r;
    }
  }
  return null;
}

export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) {
    return !!row.free;
  }
  if (model.free !== null) {
    return model.free;
  }
  return false;
}

export function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined) {
    if (caps.context_length === null || caps.context_length < requires.min_context) {
      return false;
    }
  }
  return true;
}

export function readBrief(path: string): Brief {
  if (!path) {
    return {
      directory: { schema: PROVIDERS_SCHEMA, providers: [] },
      ratings: null,
      workload: null,
      error: 'MAKEFUN_PROMPT_FILE is not set',
    };
  }

  let text = '';
  try {
    text = readFileSync(path, 'utf8');
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      directory: { schema: PROVIDERS_SCHEMA, providers: [] },
      ratings: null,
      workload: null,
      error: `Failed to read brief file at ${path}: ${msg}`,
    };
  }

  const objects: Json[] = [];
  let i = 0;
  const len = text.length;

  while (i < len) {
    const start = text.indexOf('{', i);
    if (start === -1) break;

    let depth = 0;
    let inString = false;
    let escaped = false;
    let end = -1;

    for (let j = start; j < len; j++) {
      const ch = text[j];
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (ch === '\\') {
          escaped = true;
        } else if (ch === '"') {
          inString = false;
        }
      } else {
        if (ch === '"') {
          inString = true;
        } else if (ch === '{') {
          depth++;
        } else if (ch === '}') {
          depth--;
          if (depth === 0) {
            end = j;
            break;
          }
        }
      }
    }

    if (end !== -1) {
      const chunk = text.slice(start, end + 1);
      try {
        const parsed = JSON.parse(chunk);
        if (isRecord(parsed)) {
          objects.push(parsed);
        }
      } catch {
        // ignore invalid JSON blocks
      }
      i = start + 1;
    } else {
      break;
    }
  }

  let directory: ProviderDirectory | null = null;
  let ratings: RatingsFile | null = null;
  let workload: WorkloadFile | null = null;

  for (const obj of objects) {
    const schema = typeof obj.schema === 'string' ? obj.schema : '';
    if (schema === PROVIDERS_SCHEMA && Array.isArray(obj.providers)) {
      const rawProviders = obj.providers;
      const validProviders: ProviderDirectoryEntry[] = [];
      for (const p of rawProviders) {
        if (!isRecord(p)) continue;
        const id = typeof p.id === 'string' ? p.id.trim() : '';
        if (!id) continue;
        let wire: Wire = 'openai';
        if (p.wire === 'anthropic' || p.wire === 'gemini' || p.wire === 'openai') {
          wire = p.wire;
        } else {
          continue; // skip invalid wire
        }
        let auth: AuthScheme = 'bearer';
        if (p.auth === 'x-api-key' || p.auth === 'x-goog-api-key' || p.auth === 'none' || p.auth === 'bearer') {
          auth = p.auth;
        }
        const base = (typeof p.path === 'string' && p.path) ? p.path : (typeof p.base_path === 'string' && p.base_path) ? p.base_path : (typeof p.base_url === 'string' && p.base_url) ? p.base_url : (typeof p.base === 'string' && p.base) ? p.base : '';
        const key_env = typeof p.key_env === 'string' ? p.key_env : '';
        const key = typeof p.key === 'string' ? p.key : '';
        const name = typeof p.name === 'string' && p.name.trim() ? p.name.trim() : id;

        validProviders.push({ id, name, wire, auth, base, key_env, key });
      }
      directory = { schema: PROVIDERS_SCHEMA, providers: validProviders };
    } else if (schema === RATINGS_SCHEMA && Array.isArray(obj.ratings)) {
      const rawRatings = obj.ratings;
      const validRatings: RatingRow[] = [];
      for (const r of rawRatings) {
        if (!isRecord(r)) continue;
        const model = typeof r.model === 'string' ? r.model.trim() : '';
        const provider = typeof r.provider === 'string' ? r.provider.trim() : '';
        if (!model || !provider) continue;
        const rating_milli = typeof r.rating_milli === 'number' && Number.isFinite(r.rating_milli) ? r.rating_milli : 0;
        const context_length = typeof r.context_length === 'number' && Number.isFinite(r.context_length) ? r.context_length : null;
        const tools = r.tools === true;
        const vision = r.vision === true;
        const json = r.json === true;
        const free = r.free === true;
        validRatings.push({ model, provider, rating_milli, context_length, tools, vision, json, free });
      }
      const rawBest = Array.isArray(obj.best) ? obj.best : [];
      const best: { id: string; rating_milli: number }[] = [];
      for (const b of rawBest) {
        if (!isRecord(b)) continue;
        const id = typeof b.id === 'string' ? b.id.trim() : '';
        if (!id) continue;
        const rm = typeof b.rating_milli === 'number' && Number.isFinite(b.rating_milli) ? b.rating_milli : 0;
        best.push({ id, rating_milli: rm });
      }
      ratings = { schema: RATINGS_SCHEMA, ratings: validRatings, best };
    } else if (schema === WORKLOAD_SCHEMA && Array.isArray(obj.manifest)) {
      const rawManifest = obj.manifest;
      const validManifest: WorkloadRequest[] = [];
      for (const w of rawManifest) {
        if (!isRecord(w)) continue;
        const id = typeof w.id === 'string' ? w.id.trim() : '';
        if (!id) continue;
        const api = w.api === 'anthropic' ? 'anthropic' : 'openai';
        const stream = w.stream === true;
        const weight = w.weight === 2 ? 2 : 1;
        const reqObj = isRecord(w.requires) ? w.requires : {};
        const requires: Requirement = {};
        if (reqObj.tools === true) requires.tools = true;
        if (reqObj.vision === true) requires.vision = true;
        if (reqObj.json === true) requires.json = true;
        if (typeof reqObj.min_context === 'number' && Number.isFinite(reqObj.min_context)) {
          requires.min_context = reqObj.min_context;
        }
        validManifest.push({ id, api, stream, weight, requires });
      }
      workload = { schema: WORKLOAD_SCHEMA, manifest: validManifest };
    }
  }

  if (!directory || directory.providers.length === 0) {
    return {
      directory: { schema: PROVIDERS_SCHEMA, providers: [] },
      ratings,
      workload,
      error: 'No valid providers directory found in brief file',
    };
  }

  return {
    directory,
    ratings,
    workload,
    error: null,
  };
}

export function zeroPriced(rec: Json): boolean | null {
  const pricing = rec.pricing;
  if (!isRecord(pricing)) return null;
  const values: number[] = [];
  for (const k of Object.keys(pricing)) {
    const val = pricing[k];
    if (typeof val === 'number' && Number.isFinite(val)) {
      values.push(val);
    } else if (typeof val === 'string' && val.trim() !== '') {
      const parsed = Number(val);
      if (Number.isFinite(parsed)) {
        values.push(parsed);
      }
    }
  }
  if (values.length === 0) return null;
  return values.every((v) => v === 0);
}

export function boolOrNull(v: unknown): boolean | null {
  if (typeof v === 'boolean') return v;
  return null;
}

export async function probeProvider(envUrl: string, provider: ProviderDirectoryEntry, timeoutMs = 5000): Promise<Probe> {
  const fullBase = joinBase(envUrl, provider.base);
  const endpoint = provider.wire === 'gemini' ? `${fullBase}/v1beta/models` : `${fullBase}/v1/models`;

  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
  }

  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      provider,
      ok: false,
      status: 0,
      models: [],
      error: `GET ${endpoint} threw error: ${msg}`,
    };
  }

  if (!response.ok) {
    return {
      provider,
      ok: false,
      status: response.status,
      models: [],
      error: `GET ${endpoint} answered HTTP ${response.status}`,
    };
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      provider,
      ok: false,
      status: response.status,
      models: [],
      error: `GET ${endpoint} JSON parse failed: ${msg}`,
    };
  }

  if (!isRecord(body)) {
    return {
      provider,
      ok: false,
      status: response.status,
      models: [],
      error: `GET ${endpoint} returned non-object JSON body`,
    };
  }

  let rawList: unknown[] = [];
  if (provider.wire === 'gemini') {
    if (Array.isArray(body.models)) {
      rawList = body.models;
    }
  } else {
    if (Array.isArray(body.data)) {
      rawList = body.data;
    } else if (Array.isArray(body.models)) {
      rawList = body.models;
    }
  }

  const probedModels: ProbedModel[] = [];
  for (const item of rawList) {
    if (!isRecord(item)) continue;
    let rawId = typeof item.id === 'string' ? item.id.trim() : typeof item.name === 'string' ? item.name.trim() : '';
    if (!rawId) continue;
    if (provider.wire === 'gemini' && rawId.startsWith('models/')) {
      rawId = rawId.slice(7).trim();
      if (!rawId) continue;
    }

    const displayName = typeof item.display_name === 'string' ? item.display_name.trim() : typeof item.displayName === 'string' ? item.displayName.trim() : rawId;
    const ctx = typeof item.context_length === 'number' && Number.isFinite(item.context_length) ? item.context_length : typeof item.context_window === 'number' && Number.isFinite(item.context_window) ? item.context_window : typeof item.inputTokenLimit === 'number' && Number.isFinite(item.inputTokenLimit) ? item.inputTokenLimit : null;

    const freeVal = boolOrNull(item.free) ?? zeroPriced(item);
    const toolsVal = boolOrNull(item.tools) ?? boolOrNull(item.supports_tools);
    const visionVal = boolOrNull(item.vision) ?? boolOrNull(item.supports_vision);
    const jsonVal = boolOrNull(item.json) ?? boolOrNull(item.supports_json);

    probedModels.push({
      id: rawId,
      display_name: displayName,
      free: freeVal,
      context_length: ctx,
      tools: toolsVal,
      vision: visionVal,
      json: jsonVal,
    });
  }

  return {
    provider,
    ok: true,
    status: response.status,
    models: probedModels,
    error: null,
  };
}

export function buildCatalog(_envUrl: string, probes: Probe[], ratings: RatingsFile | null): CatalogReport {
  const catalogProviders: CatalogProvider[] = [];

  for (const probe of probes) {
    if (!probe.ok) continue;

    const seenIds = new Set<string>();
    const modelsList: CatalogModel[] = [];

    for (const model of probe.models) {
      if (seenIds.has(model.id)) continue;
      seenIds.add(model.id);

      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      const isFree = modelFree(model, row);
      const ctx = row?.context_length ?? model.context_length ?? null;

      modelsList.push({
        id: model.id,
        free: isFree,
        context_length: ctx,
      });
    }

    modelsList.sort((a, b) => a.id.localeCompare(b.id));

    catalogProviders.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models: modelsList,
    });
  }

  return {
    providers: catalogProviders,
  };
}

