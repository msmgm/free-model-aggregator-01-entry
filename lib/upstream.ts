/**
 * @file upstream.ts
 * @description Build and send one upstream provider call on the provider's own wire,
 * returning the raw Response for the caller to translate. Retries are the caller's job.
 *
 * Designed for the make.fun free-model-aggregator-01 entry (Node 22, TypeScript strict).
 *
 * Key characteristics:
 * - Pure functions for building URL, headers, and request body.
 * - No clock reading, no Date, no Math.random.
 * - Standard library only (fetch, AbortSignal).
 */

import { authHeaders, isRecord, joinBase, type Json, type ProviderDirectoryEntry } from './providers.ts';
import { anthropicToOpenAI, openAIToAnthropicRequest } from './wire.ts';
import { anthropicToGemini, openAIToGemini } from './gemini.ts';

// __TYPES__
export type ClientApi = 'openai' | 'anthropic';

export interface UpstreamCall {
  ok: boolean;
  status: number;
  response: Response | null;
  error: string | null;
}
// __BUILD__
export function upstreamUrl(envUrl: string, provider: ProviderDirectoryEntry, model: string, stream: boolean): string {
  const base = joinBase(envUrl, provider.base);
  if (provider.wire === 'openai') {
    return `${base}/v1/chat/completions`;
  }
  if (provider.wire === 'anthropic') {
    return `${base}/v1/messages`;
  }
  if (provider.wire === 'gemini') {
    return `${base}/v1beta/models/${encodeURIComponent(model)}${stream ? ':streamGenerateContent' : ':generateContent'}`;
  }
  return `${base}/v1/chat/completions`;
}

export function upstreamBody(provider: ProviderDirectoryEntry, model: string, clientApi: ClientApi, body: Json): Json {
  if (provider.wire === 'openai') {
    if (clientApi === 'anthropic') {
      return anthropicToOpenAI(body, model);
    }
    return {
      ...body,
      model,
      ...(body.stream === true ? { stream_options: { include_usage: true } } : {}),
    };
  }
  if (provider.wire === 'anthropic') {
    if (clientApi === 'anthropic') {
      return {
        ...body,
        model,
      };
    }
    return openAIToAnthropicRequest(body, model);
  }
  if (provider.wire === 'gemini') {
    if (clientApi === 'anthropic') {
      return anthropicToGemini(body, model);
    }
    return openAIToGemini(body, model);
  }
  return body;
}

export function upstreamHeaders(provider: ProviderDirectoryEntry, stream: boolean): Record<string, string> {
  return {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
    ...(stream ? { accept: 'text/event-stream' } : { accept: 'application/json' }),
    ...(provider.wire === 'anthropic' ? { 'anthropic-version': '2023-06-01' } : {}),
  };
}
// __CALL__
export async function callUpstream(envUrl: string, provider: ProviderDirectoryEntry, model: string, clientApi: ClientApi, body: Json): Promise<UpstreamCall> {
  const stream = body.stream === true;
  const url = upstreamUrl(envUrl, provider, model, stream);
  const init: RequestInit = {
    method: 'POST',
    headers: upstreamHeaders(provider, stream),
    body: JSON.stringify(upstreamBody(provider, model, clientApi, body)),
    signal: AbortSignal.timeout(10000),
  };
  try {
    const response = await fetch(url, init);
    return {
      ok: response.ok,
      status: response.status,
      response,
      error: response.ok ? null : 'HTTP ' + response.status,
    };
  } catch (err) {
    return {
      ok: false,
      status: 0,
      response: null,
      error: (err as Error).message,
    };
  }
}
// __END__
