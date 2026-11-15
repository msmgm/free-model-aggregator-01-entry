/**
 * @file main.ts
 * @description HTTP server for the make.fun free-model-aggregator-01 competition entry.
 *
 * Background (Trial Flow):
 * The sandbox sets MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, and MAKEFUN_TRIAL_SEED.
 * The server reads the brief (providers, quality table, workload manifest), probes every
 * provider's catalog route, builds ONE catalog, listens on 0.0.0.0 at a free port (never 8080),
 * POSTs $MAKEFUN_ENV_URL/register {protocol_version: 1, port, catalog}, then serves the client's
 * 20 requests one at a time (GET /v1/models, POST /v1/chat/completions, POST /v1/messages, JSON or SSE),
 * polls GET $MAKEFUN_ENV_URL/work every 100 ms, and exits 0 when it says {done: true}.
 * A process still running at the 180 s wall scores 0. 'node main.ts --list' prints the catalog
 * JSON and exits 0 without starting a server (probe first; with no env/brief it prints {"providers":[]} and exits 0).
 *
 * Routing Policy:
 * Receding-horizon min-cost-flow plan with cap 2 per provider, re-planned before each request
 * from the remaining manifest with remaining caps, failed providers excluded; failover retries
 * in the planner's order, unused providers first.
 */

import { createServer, IncomingMessage, Server, ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  isRecord,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import { candidatesFor, orderCandidates, planAssignment, PLANNER_CAPS } from './lib/plan.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  openAIError,
  openAIEventsFromAnthropic,
  openAIToAnthropic,
  anthropicToOpenAIResponse,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import { geminiEventsToAnthropic, geminiEventsToOpenAI, geminiToAnthropic, geminiToOpenAI } from './lib/gemini.ts';
import { callUpstream, type ClientApi } from './lib/upstream.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
const used = new Map<string, number>();
const failed = new Set<string>();

function log(message: string): void {
  process.stderr.write('aggregator: ' + message + '\n');
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS))
  );
  for (const probe of probes) {
    if (!probe.ok && probe.error !== null) log(probe.error);
  }
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}
// __SETUP__
const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function probeById(id: string): Probe | undefined {
  return probes.find((p) => p.provider.id === id);
}

function servedModels(): ServedModel[] {
  const result: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const sorted = [...probe.models].sort(byId);
    for (const model of sorted) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      result.push({
        id: model.id,
        provider: probe.provider.id,
        display_name: model.display_name,
      });
    }
  }
  return result;
}

function requirementsFromBody(body: Json): Requirement {
  const req: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    req.tools = true;
  }
  if (isRecord(body.response_format)) {
    const rfType = body.response_format.type;
    if (rfType === 'json_object' || rfType === 'json_schema') {
      req.json = true;
    }
  }
  if (Array.isArray(body.messages)) {
    for (const m of body.messages) {
      if (!isRecord(m)) continue;
      const content = m.content;
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'image' || block.type === 'image_url') {
            req.vision = true;
          }
        }
      }
    }
  }
  return req;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) {
    return entry.requires;
  }
  return requirementsFromBody(body);
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let totalBytes = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    totalBytes += buf.length;
    if (totalBytes > MAX_BODY_BYTES) {
      return null;
    }
    chunks.push(buf);
  }
  if (totalBytes === 0) {
    return null;
  }
  const raw = Buffer.concat(chunks).toString('utf8');
  try {
    const parsed = JSON.parse(raw);
    if (isRecord(parsed)) {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const jsonStr = JSON.stringify(body);
  const payload = Buffer.from(jsonStr, 'utf8');
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': payload.length,
  });
  res.end(payload);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string
): void {
  if (wire === 'anthropic') {
    sendJson(res, status, anthropicError(type, message));
  } else {
    sendJson(res, status, openAIError(message, type, null));
  }
}
// __HELPERS__
async function serveRequest(
  clientApi: ClientApi,
  body: Json
): Promise<{
  status: number;
  wire: 'openai' | 'anthropic';
  json?: unknown;
  stream?: AsyncGenerator<string> | ReadableStream<Uint8Array> | null;
}> {
  const requires = requirementsFor(body);
  const manifest = brief.workload?.manifest ?? [];
  const currentEntry = manifest[manifestIndex - 1];
  const remaining =
    currentEntry !== undefined
      ? manifest.slice(manifestIndex - 1)
      : [{ id: 'current', api: clientApi, stream: body.stream === true, weight: 1 as const, requires }];

  const candidatesByRequest = new Map<string, ReturnType<typeof candidatesFor>>();
  for (const req of remaining) {
    candidatesByRequest.set(req.id, candidatesFor(brief.ratings, probes, req.requires, req.api));
  }

  const caps: Record<string, number> = {};
  for (const probe of probes) {
    const id = probe.provider.id;
    if (failed.has(id)) {
      caps[id] = 0;
    } else {
      const defaultCap = PLANNER_CAPS[id] ?? 2;
      const usedCount = used.get(id) ?? 0;
      caps[id] = Math.max(0, defaultCap - usedCount);
    }
  }

  const plan = planAssignment(remaining, candidatesByRequest, caps);
  const currentId = currentEntry?.id ?? 'current';
  const planned = plan.get(currentId);
  const currentCands =
    candidatesByRequest.get(currentId) ?? candidatesFor(brief.ratings, probes, requires, clientApi);
  const ordered = orderCandidates(currentCands, planned?.provider ?? null, used, failed);

  for (const cand of ordered) {
    const probe = probeById(cand.provider);
    if (!probe) continue;

    const prevUsed = used.get(cand.provider) ?? 0;
    used.set(cand.provider, prevUsed + 1);

    const call = await callUpstream(ENV_URL, probe.provider, cand.model, clientApi, body);
    if (call.status === 0) {
      used.set(cand.provider, prevUsed);
      continue;
    }

    if (!call.ok || call.response === null) {
      failed.add(cand.provider);
      continue;
    }

    const translated = await translateResponse(
      clientApi,
      probe.provider.wire,
      body.stream === true,
      call.response,
      cand.model
    );

    if (translated === null) {
      failed.add(cand.provider);
      continue;
    }

    return translated;
  }

  return { status: 503, wire: clientApi };
}
// __SERVE__
async function translateResponse(
  clientApi: ClientApi,
  providerWire: 'openai' | 'anthropic' | 'gemini',
  stream: boolean,
  response: Response,
  model: string
): Promise<{
  status: number;
  wire: 'openai' | 'anthropic';
  json?: unknown;
  stream?: AsyncGenerator<string> | ReadableStream<Uint8Array> | null;
} | null> {
  if (stream) {
    const body = response.body;
    if (!body) return null;
    if (clientApi === 'openai') {
      if (providerWire === 'openai') {
        return { status: 200, wire: 'openai', stream: body };
      }
      if (providerWire === 'anthropic') {
        return { status: 200, wire: 'openai', stream: openAIEventsFromAnthropic(body, model) };
      }
      if (providerWire === 'gemini') {
        return { status: 200, wire: 'openai', stream: geminiEventsToOpenAI(body, model) };
      }
    } else {
      if (providerWire === 'anthropic') {
        return { status: 200, wire: 'anthropic', stream: body };
      }
      if (providerWire === 'openai') {
        return { status: 200, wire: 'anthropic', stream: anthropicEventsFromOpenAI(body, model) };
      }
      if (providerWire === 'gemini') {
        return { status: 200, wire: 'anthropic', stream: geminiEventsToAnthropic(body, model) };
      }
    }
    return null;
  } else {
    let parsed: unknown;
    try {
      parsed = await response.json();
    } catch {
      return null;
    }
    if (!isRecord(parsed)) {
      return null;
    }

    if (clientApi === 'openai') {
      if (providerWire === 'openai') {
        return { status: 200, wire: 'openai', json: parsed };
      }
      if (providerWire === 'anthropic') {
        return { status: 200, wire: 'openai', json: anthropicToOpenAIResponse(parsed, model) };
      }
      if (providerWire === 'gemini') {
        return { status: 200, wire: 'openai', json: geminiToOpenAI(parsed, model) };
      }
    } else {
      if (providerWire === 'openai') {
        return { status: 200, wire: 'anthropic', json: openAIToAnthropic(parsed, model) };
      }
      if (providerWire === 'anthropic') {
        return { status: 200, wire: 'anthropic', json: parsed };
      }
      if (providerWire === 'gemini') {
        return { status: 200, wire: 'anthropic', json: geminiToAnthropic(parsed, model) };
      }
    }
    return null;
  }
}
// __TRANSLATE__
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, anthropicModelList(list));
  } else {
    sendJson(res, 200, openAIModelList(list));
  }
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const result = await serveRequest('openai', body);
  if (result.status !== 200) {
    sendError(res, 'openai', result.status, 'no_free_model', 'no free model on the directory meets this request');
    return;
  }
  if (result.json !== undefined) {
    sendJson(res, 200, result.json);
    return;
  } else if (result.stream) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    const s = result.stream;
    if (typeof (s as ReadableStream).getReader === 'function') {
      const reader = (s as ReadableStream<Uint8Array>).getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) res.write(value);
        }
      } finally {
        try {
          reader.releaseLock();
        } catch {
          // ignore
        }
        res.end();
      }
    } else {
      try {
        for await (const chunk of s as AsyncGenerator<string>) {
          res.write(chunk);
        }
      } catch {
        // ignore stream errors
      } finally {
        res.end();
      }
    }
  } else {
    sendJson(res, 200, null);
  }
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const result = await serveRequest('anthropic', body);
  if (result.status !== 200) {
    sendError(res, 'anthropic', result.status, 'overloaded_error', 'no free model on the directory meets this request');
    return;
  }
  if (result.json !== undefined) {
    sendJson(res, 200, result.json);
    return;
  } else if (result.stream) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    const s = result.stream;
    if (typeof (s as ReadableStream).getReader === 'function') {
      const reader = (s as ReadableStream<Uint8Array>).getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) res.write(value);
        }
      } finally {
        try {
          reader.releaseLock();
        } catch {
          // ignore
        }
        res.end();
      }
    } else {
      try {
        for await (const chunk of s as AsyncGenerator<string>) {
          res.write(chunk);
        }
      } catch {
        // ignore stream errors
      } finally {
        res.end();
      }
    }
  } else {
    sendJson(res, 200, null);
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const method = req.method ?? 'GET';
  const urlStr = req.url ?? '/';
  const url = new URL(urlStr, 'http://localhost');
  const pathname = url.pathname;

  if (method === 'GET' && pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (method === 'POST' && pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (method === 'POST' && pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }

  const wireType = pathname === '/v1/messages' ? 'anthropic' : 'openai';
  sendError(res, wireType, 404, 'not_found', 'not found');
}
// __ROUTES__
function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      server.off('error', reject);
      resolve();
    });
  });
}

async function register(port: number): Promise<void> {
  const url = `${ENV_URL}/register`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      accept: 'application/json',
    },
    body: JSON.stringify({
      protocol_version: PROTOCOL_VERSION,
      port,
      catalog,
    }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) {
    throw new Error(`Register failed with HTTP ${res.status}`);
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (isRecord(body) && typeof body.requests === 'number' && Number.isFinite(body.requests)) {
    log(`registered successfully, trial has ${body.requests} requests`);
  } else {
    log('registered successfully');
  }
}

let pollInFlight = false;
function poll(server: Server): void {
  setInterval(async () => {
    if (pollInFlight) return;
    pollInFlight = true;
    try {
      const res = await fetch(`${ENV_URL}/work`, {
        method: 'GET',
        headers: { accept: 'application/json' },
        signal: AbortSignal.timeout(2000),
      });
      if (res.ok) {
        const body = await res.json();
        if (isRecord(body) && body.done === true) {
          log('work done, shutting down');
          server.close(() => {
            process.exit(0);
          });
          setTimeout(() => process.exit(0), 1000).unref();
        }
      }
    } catch {
      // ignore poll errors
    } finally {
      pollInFlight = false;
    }
  }, POLL_INTERVAL_MS).unref();
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    await new Promise<void>((resolve, reject) => {
      process.stdout.write(JSON.stringify(catalog) + '\n', (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
    process.exit(0);
  }

  if (ENV_URL === '') {
    throw new Error('MAKEFUN_ENV_URL is not set');
  }

  const server = createServer(async (req, res) => {
    try {
      await handle(req, res);
    } catch (err) {
      const errObj = err as Error;
      log(errObj.stack ?? String(errObj));
      if (!res.headersSent) {
        const urlStr = req.url ?? '/';
        const wireType = urlStr.includes('/v1/messages') ? 'anthropic' : 'openai';
        sendError(res, wireType, 500, 'internal_error', errObj.message || 'internal server error');
      } else {
        try {
          res.end();
        } catch {
          // ignore
        }
      }
    }
  });

  await listen(server);

  const address = server.address();
  if (!address || typeof address === 'string') {
    throw new Error('Server address is not a TCP object');
  }
  const port = address.port;
  if (port === 8080) {
    throw new Error('Assigned port cannot be 8080');
  }

  log(`listening on port ${port}, ${catalog.providers.length} providers, ${servedModels().length} served models`);

  await register(port);

  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));

  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
// __START__
// __END__
