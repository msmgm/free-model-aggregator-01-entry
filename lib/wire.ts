/**
 * @file wire.ts
 * @description OpenAI <-> Anthropic client-wire translations for requests, responses,
 * and server-sent-event streams.
 *
 * Designed for the make.fun free-model-aggregator-01 entry (Node 22, TypeScript strict).
 *
 * Key characteristics:
 * - Uses FIXED timestamps (no clock reading).
 * - Every function is a pure function of its arguments.
 * - No Date, no Math.random, no network calls.
 * - Complete implementation of model listings, error normalization, request/response
 *   translation, and SSE stream generators.
 */

import { isRecord, type Json } from './providers.ts';

// __TYPES__
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}
// __LISTS__
const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_OPENAI_CREATED,
      owned_by: m.provider,
    })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}
// __ERRORS__
export function anthropicError(type: string, message: string): Json {
  return {
    type: 'error',
    error: {
      type,
      message,
    },
  };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return {
    error: {
      message,
      type,
      code,
    },
  };
}
// __REQ_TRANSLATE__
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const item of content) {
      if (isRecord(item) && item.type === 'text' && typeof item.text === 'string') {
        parts.push(item.text);
      }
    }
    return parts.join('\n');
  }
  return '';
}

export function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') {
    return { type: 'text', text: block.text };
  }
  if (block.type === 'image' && isRecord(block.source)) {
    const src = block.source;
    if (src.type === 'base64' && typeof src.data === 'string') {
      const mediaType = typeof src.media_type === 'string' && src.media_type ? src.media_type : 'image/png';
      return {
        type: 'image_url',
        image_url: {
          url: `data:${mediaType};base64,${src.data}`,
        },
      };
    }
    if (src.type === 'url' && typeof src.url === 'string') {
      return {
        type: 'image_url',
        image_url: {
          url: src.url,
        },
      };
    }
  }
  if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
    return {
      type: 'image_url',
      image_url: {
        url: block.image_url.url,
      },
    };
  }
  return null;
}

export function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const t = textOf(content);
    if (t) return t;
    try {
      return JSON.stringify(content);
    } catch {
      return '';
    }
  }
  if (content !== undefined && content !== null) {
    if (typeof content === 'object') {
      try {
        return JSON.stringify(content);
      } catch {
        return '';
      }
    }
    return String(content);
  }
  return '';
}

export function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // ignore
    }
  }
  return {};
}

export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

export function openAIFinishOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];

  const sysText = textOf(body.system);
  if (sysText) {
    messages.push({ role: 'system', content: sysText });
  }

  if (Array.isArray(body.messages)) {
    for (const m of body.messages) {
      if (!isRecord(m)) continue;
      const role = m.role === 'assistant' ? 'assistant' : 'user';
      const content = m.content;

      if (typeof content === 'string') {
        messages.push({ role, content });
      } else if (Array.isArray(content)) {
        if (role === 'assistant') {
          const toolCalls: Json[] = [];
          const textParts: string[] = [];

          for (const block of content) {
            if (!isRecord(block)) continue;
            if (block.type === 'text' && typeof block.text === 'string') {
              textParts.push(block.text);
            } else if (block.type === 'tool_use' && typeof block.id === 'string' && typeof block.name === 'string') {
              const inputArg = argumentsOf(block.input);
              toolCalls.push({
                id: block.id,
                type: 'function',
                function: {
                  name: block.name,
                  arguments: JSON.stringify(inputArg),
                },
              });
            }
          }

          const msgObj: Json = { role: 'assistant', content: textParts.join('') || null };
          if (toolCalls.length > 0) {
            msgObj.tool_calls = toolCalls;
          }
          messages.push(msgObj);
        } else {
          // user role
          const parts: Json[] = [];
          for (const block of content) {
            if (!isRecord(block)) continue;
            if (block.type === 'tool_result' && typeof block.tool_use_id === 'string') {
              messages.push({
                role: 'tool',
                tool_call_id: block.tool_use_id,
                content: toolResultText(block.content),
              });
            } else {
              const p = partOf(block);
              if (p) parts.push(p);
            }
          }
          if (parts.length > 0) {
            messages.push({ role: 'user', content: parts });
          }
        }
      }
    }
  }

  const result: Json = { model, messages };

  if (typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens)) {
    result.max_tokens = body.max_tokens;
  }
  if (typeof body.temperature === 'number' && Number.isFinite(body.temperature)) {
    result.temperature = body.temperature;
  }
  if (typeof body.top_p === 'number' && Number.isFinite(body.top_p)) {
    result.top_p = body.top_p;
  }
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    result.stop = body.stop_sequences.filter((s) => typeof s === 'string');
  }

  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const openaiTools: Json[] = [];
    for (const t of body.tools) {
      if (!isRecord(t)) continue;
      const name = typeof t.name === 'string' ? t.name : '';
      if (!name) continue;
      const desc = typeof t.description === 'string' ? t.description : '';
      const params = isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} };
      openaiTools.push({
        type: 'function',
        function: {
          name,
          description: desc,
          parameters: params,
        },
      });
    }
    if (openaiTools.length > 0) {
      result.tools = openaiTools;
    }
  }

  if (isRecord(body.tool_choice)) {
    const tc = body.tool_choice;
    const type = tc.type;
    if (type === 'auto') {
      result.tool_choice = 'auto';
    } else if (type === 'any') {
      result.tool_choice = 'required';
    } else if (type === 'tool' && typeof tc.name === 'string') {
      result.tool_choice = {
        type: 'function',
        function: { name: tc.name },
      };
    }
  }

  if (body.stream === true) {
    result.stream = true;
    result.stream_options = { include_usage: true };
  }

  return result;
}
// __RESP_TRANSLATE__
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};

  const content: Json[] = [];
  const rawContent = message.content;
  const textStr = textOf(rawContent);
  if (textStr) {
    content.push({ type: 'text', text: textStr });
  }

  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const tc of toolCalls) {
    if (!isRecord(tc)) continue;
    const id = typeof tc.id === 'string' ? tc.id : 'toolu_unknown';
    const fn = isRecord(tc.function) ? tc.function : {};
    const name = typeof fn.name === 'string' ? fn.name : 'unknown';
    const input = argumentsOf(fn.arguments);
    content.push({
      type: 'tool_use',
      id,
      name,
      input,
    });
  }

  if (content.length === 0) {
    content.push({ type: 'text', text: '' });
  }

  const usageRecord = isRecord(body.usage) ? body.usage : {};
  const promptTokens = typeof usageRecord.prompt_tokens === 'number' && Number.isFinite(usageRecord.prompt_tokens) ? usageRecord.prompt_tokens : 0;
  const completionTokens = typeof usageRecord.completion_tokens === 'number' && Number.isFinite(usageRecord.completion_tokens) ? usageRecord.completion_tokens : 0;

  const usage = {
    input_tokens: promptTokens,
    output_tokens: completionTokens,
  };

  const bodyId = typeof body.id === 'string' ? body.id : 'aggregate';
  const id = bodyId.startsWith('msg_') ? bodyId : 'msg_' + bodyId;

  return {
    id,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage,
  };
}

export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const rawContent = body.content;
  const textParts: string[] = [];
  const toolCalls: Json[] = [];

  if (Array.isArray(rawContent)) {
    for (const block of rawContent) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') {
        textParts.push(block.text);
      } else if (block.type === 'tool_use' && typeof block.id === 'string' && typeof block.name === 'string') {
        const inputArg = argumentsOf(block.input);
        toolCalls.push({
          id: block.id,
          type: 'function',
          function: {
            name: block.name,
            arguments: JSON.stringify(inputArg),
          },
        });
      }
    }
  } else if (typeof rawContent === 'string') {
    textParts.push(rawContent);
  }

  const text = textParts.join('');
  const message: Json = {
    role: 'assistant',
    content: text,
  };
  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }

  const usageRecord = isRecord(body.usage) ? body.usage : {};
  const inputTokens = typeof usageRecord.input_tokens === 'number' && Number.isFinite(usageRecord.input_tokens) ? usageRecord.input_tokens : 0;
  const outputTokens = typeof usageRecord.output_tokens === 'number' && Number.isFinite(usageRecord.output_tokens) ? usageRecord.output_tokens : 0;
  const totalTokens = inputTokens + outputTokens;

  const usage = {
    prompt_tokens: inputTokens,
    completion_tokens: outputTokens,
    total_tokens: totalTokens,
  };

  const bodyId = typeof body.id === 'string' ? body.id : 'aggregate';
  const cleanId = bodyId.replace(/^msg_/, '');
  const id = 'chatcmpl_' + cleanId;

  const finishReason = openAIFinishOf(body.stop_reason);

  return {
    id,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: finishReason,
        logprobs: null,
      },
    ],
    usage,
  };
}
// __SSE__
export function sseEvent(event: string | null, data: unknown): string {
  return (event === null ? '' : 'event: ' + event + '\n') + 'data: ' + JSON.stringify(data) + '\n\n';
}

export async function* anthropicEventsFromOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  let started = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStopReason = 'end_turn';
  let hasFinished = false;

  let textBlockStarted = false;
  let textBlockIndex = 0;
  const toolBlocks = new Map<number, { blockIndex: number; id: string; name: string; partialJson: string }>();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const normalized = buffer.replace(/\r\n/g, '\n');
      const parts = normalized.split('\n\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        const lines = part.split('\n');
        let dataStr = '';
        for (const line of lines) {
          if (line.startsWith('data:')) {
            dataStr = line.slice(5).trim();
          }
        }
        if (!dataStr) continue;
        if (dataStr === '[DONE]') {
          hasFinished = true;
          break;
        }

        let chunk: Json;
        try {
          const parsed = JSON.parse(dataStr);
          if (isRecord(parsed)) {
            chunk = parsed;
          } else {
            continue;
          }
        } catch {
          continue;
        }

        const usage = isRecord(chunk.usage) ? chunk.usage : null;
        if (usage) {
          if (typeof usage.prompt_tokens === 'number' && Number.isFinite(usage.prompt_tokens)) {
            inputTokens = usage.prompt_tokens;
          }
          if (typeof usage.completion_tokens === 'number' && Number.isFinite(usage.completion_tokens)) {
            outputTokens = usage.completion_tokens;
          }
        }

        const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
        if (choices.length === 0) continue;
        const choice = isRecord(choices[0]) ? choices[0] : {};

        const finishReason = choice.finish_reason;
        if (finishReason !== null && finishReason !== undefined) {
          pendingStopReason = stopReasonOf(finishReason);
        }

        const delta = isRecord(choice.delta) ? choice.delta : {};

        if (!started) {
          started = true;
          yield sseEvent('message_start', {
            type: 'message_start',
            message: {
              id: 'msg_stream',
              type: 'message',
              role: 'assistant',
              model,
              content: [],
              stop_reason: null,
              stop_sequence: null,
              usage: { input_tokens: inputTokens, output_tokens: 0 },
            },
          });
        }

        const contentDelta = delta.content;
        if (typeof contentDelta === 'string' && contentDelta.length > 0) {
          if (!textBlockStarted) {
            textBlockStarted = true;
            textBlockIndex = 0;
            yield sseEvent('content_block_start', {
              type: 'content_block_start',
              index: textBlockIndex,
              content_block: { type: 'text', text: '' },
            });
          }
          yield sseEvent('content_block_delta', {
            type: 'content_block_delta',
            index: textBlockIndex,
            delta: { type: 'text_delta', text: contentDelta },
          });
        }

        const toolCallsDelta = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
        for (const tc of toolCallsDelta) {
          if (!isRecord(tc)) continue;
          const upstreamIdx = typeof tc.index === 'number' ? tc.index : 0;
          let entry = toolBlocks.get(upstreamIdx);
          if (!entry) {
            if (textBlockStarted) {
              yield sseEvent('content_block_stop', {
                type: 'content_block_stop',
                index: textBlockIndex,
              });
              textBlockStarted = false;
            }
            const blockIndex = (textBlockStarted ? 1 : 0) + toolBlocks.size;
            const tcId = typeof tc.id === 'string' ? tc.id : 'toolu_' + upstreamIdx;
            const fn = isRecord(tc.function) ? tc.function : {};
            const fnName = typeof fn.name === 'string' ? fn.name : 'unknown';
            entry = { blockIndex, id: tcId, name: fnName, partialJson: '' };
            toolBlocks.set(upstreamIdx, entry);

            yield sseEvent('content_block_start', {
              type: 'content_block_start',
              index: blockIndex,
              content_block: {
                type: 'tool_use',
                id: entry.id,
                name: entry.name,
                input: {},
              },
            });
          }

          const fn = isRecord(tc.function) ? tc.function : {};
          const argsDelta = typeof fn.arguments === 'string' ? fn.arguments : '';
          if (argsDelta) {
            entry.partialJson += argsDelta;
            yield sseEvent('content_block_delta', {
              type: 'content_block_delta',
              index: entry.blockIndex,
              delta: {
                type: 'input_json_delta',
                partial_json: argsDelta,
              },
            });
          }
        }
      }
      if (hasFinished) break;
    }
  } finally {
    reader.releaseLock();
  }

  if (!started) {
    started = true;
    yield sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });
  }

  if (textBlockStarted) {
    yield sseEvent('content_block_stop', {
      type: 'content_block_stop',
      index: textBlockIndex,
    });
  }
  for (const entry of toolBlocks.values()) {
    yield sseEvent('content_block_stop', {
      type: 'content_block_stop',
      index: entry.blockIndex,
    });
  }

  yield sseEvent('message_delta', {
    type: 'message_delta',
    delta: {
      stop_reason: pendingStopReason,
      stop_sequence: null,
    },
    usage: {
      output_tokens: outputTokens,
    },
  });

  yield sseEvent('message_stop', {
    type: 'message_stop',
  });
}

export async function* openAIEventsFromAnthropic(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  let firstChunkEmitted = false;
  let hasEnded = false;

  const blocks = new Map<number, { type: 'text' | 'tool_use'; text: string; id: string; name: string; partialJson: string }>();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const normalized = buffer.replace(/\r\n/g, '\n');
      const parts = normalized.split('\n\n');
      buffer = parts.pop() || '';

      let currentEvent: string | null = null;

      for (const part of parts) {
        const lines = part.split('\n');
        currentEvent = null;
        let dataStr = '';

        for (const line of lines) {
          if (line.startsWith('event:')) {
            currentEvent = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            dataStr = line.slice(5).trim();
          }
        }

        if (!dataStr) continue;

        let data: Json;
        try {
          const parsed = JSON.parse(dataStr);
          if (isRecord(parsed)) {
            data = parsed;
          } else {
            continue;
          }
        } catch {
          continue;
        }

        const eventType = currentEvent || data.type;

        if (eventType === 'message_start') {
          const msg = isRecord(data.message) ? data.message : {};
          const usage = isRecord(msg.usage) ? msg.usage : {};
          if (typeof usage.input_tokens === 'number' && Number.isFinite(usage.input_tokens)) {
            inputTokens = usage.input_tokens;
          }
        } else if (eventType === 'content_block_start') {
          const idx = typeof data.index === 'number' ? data.index : 0;
          const cb = isRecord(data.content_block) ? data.content_block : {};
          const cbType = cb.type === 'tool_use' ? 'tool_use' : 'text';
          const id = typeof cb.id === 'string' ? cb.id : 'toolu_' + idx;
          const name = typeof cb.name === 'string' ? cb.name : 'unknown';
          blocks.set(idx, { type: cbType, text: '', id, name, partialJson: '' });
        } else if (eventType === 'content_block_delta') {
          const idx = typeof data.index === 'number' ? data.index : 0;
          const delta = isRecord(data.delta) ? data.delta : {};
          const block = blocks.get(idx);
          if (block) {
            if (delta.type === 'text_delta' && typeof delta.text === 'string') {
              block.text += delta.text;
              const piece = delta.text;
              if (piece) {
                if (!firstChunkEmitted) {
                  firstChunkEmitted = true;
                  yield 'data: ' + JSON.stringify({
                    id: 'chatcmpl-sim',
                    object: 'chat.completion.chunk',
                    created: FIXED_OPENAI_CREATED,
                    model,
                    choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
                  }) + '\n\n';
                }
                yield 'data: ' + JSON.stringify({
                  choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
                }) + '\n\n';
              }
            } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
              block.partialJson += delta.partial_json;
            }
          }
        } else if (eventType === 'message_delta') {
          const delta = isRecord(data.delta) ? data.delta : {};
          if (typeof delta.stop_reason === 'string' && delta.stop_reason) {
            stopReason = delta.stop_reason;
          }
          const usage = isRecord(data.usage) ? data.usage : {};
          if (typeof usage.output_tokens === 'number' && Number.isFinite(usage.output_tokens)) {
            outputTokens = usage.output_tokens;
          }
        } else if (eventType === 'message_stop') {
          hasEnded = true;
          break;
        }
      }
      if (hasEnded) break;
    }
  } finally {
    reader.releaseLock();
  }

  if (!firstChunkEmitted) {
    firstChunkEmitted = true;
    yield 'data: ' + JSON.stringify({
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
    }) + '\n\n';
  }

  const toolCallsPayload: Json[] = [];
  let toolCallIndexCounter = 0;
  for (const block of blocks.values()) {
    if (block.type === 'tool_use') {
      let args = block.partialJson;
      if (!args) args = '{}';
      toolCallsPayload.push({
        index: toolCallIndexCounter++,
        id: block.id,
        type: 'function',
        function: {
          name: block.name,
          arguments: args,
        },
      });
    }
  }

  const finishChunkDelta: Json = {};
  if (toolCallsPayload.length > 0) {
    finishChunkDelta.tool_calls = toolCallsPayload;
  }

  const openAIFinish = openAIFinishOf(stopReason);

  yield 'data: ' + JSON.stringify({
    choices: [
      {
        index: 0,
        delta: finishChunkDelta,
        finish_reason: openAIFinish,
      },
    ],
  }) + '\n\n';

  const totalTokens = inputTokens + outputTokens;
  yield 'data: ' + JSON.stringify({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: inputTokens,
      completion_tokens: outputTokens,
      total_tokens: totalTokens,
    },
  }) + '\n\n';

  yield 'data: [DONE]\n\n';
}

export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const sysParts: string[] = [];

  if (body.system !== undefined && body.system !== null) {
    const st = textOf(body.system);
    if (st) sysParts.push(st);
  }

  if (Array.isArray(body.messages)) {
    for (const m of body.messages) {
      if (!isRecord(m)) continue;
      const role = typeof m.role === 'string' ? m.role : '';
      if (role === 'system') {
        const st = textOf(m.content);
        if (st) sysParts.push(st);
        continue;
      }

      if (role === 'user') {
        const content = m.content;
        if (typeof content === 'string') {
          if (content !== '') {
            messages.push({ role: 'user', content });
          }
        } else if (Array.isArray(content)) {
          const parts: Json[] = [];
          for (const block of content) {
            if (typeof block === 'string') {
              if (block) parts.push({ type: 'text', text: block });
            } else if (isRecord(block)) {
              if (block.type === 'text' && typeof block.text === 'string') {
                parts.push({ type: 'text', text: block.text });
              } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
                const url = block.image_url.url;
                if (url.startsWith('data:')) {
                  const match = url.match(/^data:([^;,]*)(;base64)?,(.*)$/s);
                  if (match) {
                    const mime = match[1] || 'image/png';
                    const data = match[3] || '';
                    parts.push({
                      type: 'image',
                      source: {
                        type: 'base64',
                        media_type: mime || 'image/png',
                        data,
                      },
                    });
                  } else {
                    parts.push({ type: 'image', source: { type: 'url', url } });
                  }
                } else {
                  parts.push({ type: 'image', source: { type: 'url', url } });
                }
              }
            }
          }
          if (parts.length > 0) {
            messages.push({ role: 'user', content: parts });
          }
        }
      } else if (role === 'assistant') {
        const content = m.content;
        const textStr = textOf(content);
        const toolCalls = Array.isArray(m.tool_calls) ? m.tool_calls : [];

        const assistantContent: Json[] = [];
        if (textStr) {
          assistantContent.push({ type: 'text', text: textStr });
        }
        for (const tc of toolCalls) {
          if (!isRecord(tc)) continue;
          const id = typeof tc.id === 'string' && tc.id ? tc.id : 'toolu_unknown';
          const fn = isRecord(tc.function) ? tc.function : {};
          const name = typeof fn.name === 'string' ? fn.name : 'unknown';
          const input = argumentsOf(fn.arguments);
          assistantContent.push({
            type: 'tool_use',
            id,
            name,
            input,
          });
        }

        if (assistantContent.length > 0) {
          messages.push({
            role: 'assistant',
            content: assistantContent.length === 1 && assistantContent[0].type === 'text' ? (assistantContent[0].text as string) : assistantContent,
          });
        }
      } else if (role === 'tool') {
        const toolCallId = typeof m.tool_call_id === 'string' ? m.tool_call_id : '';
        const toolResText = toolResultText(m.content);
        messages.push({
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: toolCallId,
              content: toolResText,
            },
          ],
        });
      }
    }
  }

  const result: Json = { model, messages };

  if (sysParts.length > 0) {
    result.system = sysParts.join('\n');
  }

  const maxTokens = typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens)
    ? body.max_tokens
    : typeof body.max_completion_tokens === 'number' && Number.isFinite(body.max_completion_tokens)
    ? body.max_completion_tokens
    : null;
  if (maxTokens !== null) {
    result.max_tokens = maxTokens;
  }

  const temp = typeof body.temperature === 'number' && Number.isFinite(body.temperature) ? body.temperature : null;
  if (temp !== null) {
    result.temperature = temp;
  }

  const topP = typeof body.top_p === 'number' && Number.isFinite(body.top_p) ? body.top_p : null;
  if (topP !== null) {
    result.top_p = topP;
  }

  const stop = body.stop;
  if (typeof stop === 'string' && stop !== '') {
    result.stop_sequences = [stop];
  } else if (Array.isArray(stop) && stop.length > 0) {
    const stops = stop.filter((s) => typeof s === 'string' && s !== '');
    if (stops.length > 0) {
      result.stop_sequences = stops;
    }
  }

  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const anthropicTools: Json[] = [];
    for (const t of body.tools) {
      if (!isRecord(t)) continue;
      const fn = isRecord(t.function) ? t.function : t;
      const name = typeof fn.name === 'string' ? fn.name : (typeof t.name === 'string' ? t.name : '');
      if (!name) continue;
      const desc = typeof fn.description === 'string' ? fn.description : (typeof t.description === 'string' ? t.description : '');
      const params = isRecord(fn.parameters) ? fn.parameters : (isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} });
      anthropicTools.push({
        name,
        description: desc,
        input_schema: params,
      });
    }
    if (anthropicTools.length > 0) {
      result.tools = anthropicTools;
    }
  }

  if (typeof body.tool_choice === 'string') {
    const tc = body.tool_choice;
    if (tc === 'auto') {
      result.tool_choice = { type: 'auto' };
    } else if (tc === 'required') {
      result.tool_choice = { type: 'any' };
    } else if (tc === 'none') {
      result.tool_choice = { type: 'auto' };
    }
  } else if (isRecord(body.tool_choice)) {
    const tc = body.tool_choice;
    const type = tc.type;
    if (type === 'auto') {
      result.tool_choice = { type: 'auto' };
    } else if (type === 'required') {
      result.tool_choice = { type: 'any' };
    } else if (type === 'function' && isRecord(tc.function) && typeof tc.function.name === 'string') {
      result.tool_choice = { type: 'tool', name: tc.function.name };
    } else if (typeof tc.name === 'string') {
      result.tool_choice = { type: 'tool', name: tc.name };
    }
  }

  if (body.stream === true) {
    result.stream = true;
  }

  return result;
}
// __END__
