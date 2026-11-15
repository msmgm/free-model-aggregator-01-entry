/**
 * @file gemini.ts
 * @description Gemini-wire translations (requests, responses, SSE streams) in both directions.
 * Designed for the make.fun free-model-aggregator-01 entry (Node 22, TypeScript strict).
 *
 * Echo Rule: Providers echo what was parsed from upstream requests (system text, user text,
 * image bytes, tool names, max tokens, stop sequences, JSON mode). The answer text is always
 * 'ANSWER ECHO{...}' and is passed through whole.
 *
 * Determinism & Constraints:
 * - Pure functions only.
 * - No clock reading (fixed timestamps or none).
 * - No Math.random.
 * - No network calls.
 * - Standard library only.
 */

import { isRecord, type Json } from './providers.ts';
import { openAIFinishOf, stopReasonOf, sseEvent, textOf, toolResultText } from './wire.ts';

export function geminiFinishToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

export function geminiFinishToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

export function partsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  const textPieces: string[] = [];
  for (const p of parts) {
    if (isRecord(p) && typeof p.text === 'string') {
      textPieces.push(p.text);
    }
  }
  return textPieces.join('');
}

export function parseDataUrl(url: string): { mime: string; data: string } | null {
  if (typeof url !== 'string') return null;
  const match = url.match(/^data:([^;,]*)(;base64)?,(.*)$/s);
  if (!match) return null;
  if (!match[2]) return null;
  const mime = match[1] || 'image/png';
  const data = match[3];
  return { mime, data };
}

// __HELPERS__
export function openAIToGemini(body: Json, _model: string): Json {
  const systemTexts: string[] = [];
  const contents: Json[] = [];

  const messages = Array.isArray(body.messages) ? body.messages : [];
  let lastUserMsgParts: Json[] | null = null;

  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = typeof m.role === 'string' ? m.role : '';
    const content = m.content;

    if (role === 'system') {
      const st = textOf(content);
      if (st) systemTexts.push(st);
    } else if (role === 'user') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (typeof block === 'string') {
            parts.push({ text: block });
          } else if (isRecord(block)) {
            if (block.type === 'text' && typeof block.text === 'string') {
              parts.push({ text: block.text });
            } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
              const url = block.image_url.url;
              const parsedData = parseDataUrl(url);
              if (parsedData) {
                parts.push({
                  inlineData: {
                    mimeType: parsedData.mime,
                    data: parsedData.data,
                  },
                });
              } else {
                parts.push({
                  fileData: {
                    fileUri: url,
                    mimeType: 'image/png',
                  },
                });
              }
            }
          }
        }
      }
      if (parts.length === 0) {
        parts.push({ text: '' });
      }
      lastUserMsgParts = parts;
      contents.push({ role: 'user', parts });
    } else if (role === 'assistant') {
      const ast = textOf(content);
      contents.push({
        role: 'model',
        parts: [{ text: ast }],
      });
    } else if (role === 'tool') {
      const tt = toolResultText(content);
      contents.push({
        role: 'user',
        parts: [{ text: tt }],
      });
    }
  }

  if (lastUserMsgParts !== null) {
    const lastContent = contents[contents.length - 1];
    let needsPush = true;
    if (isRecord(lastContent) && lastContent.role === 'user' && Array.isArray(lastContent.parts)) {
      if (lastContent.parts === lastUserMsgParts) {
        needsPush = false;
      }
    }
    if (contents.length === 0 || needsPush) {
      contents.push({ role: 'user', parts: lastUserMsgParts });
    }
  } else if (contents.length === 0) {
    contents.push({ role: 'user', parts: [{ text: '' }] });
  }

  const systemInstruction = systemTexts.length > 0 ? { parts: [{ text: systemTexts.join('\n') }] } : null;

  let tools: Json[] | undefined = undefined;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const decls: Json[] = [];
    for (const t of body.tools) {
      if (!isRecord(t)) continue;
      const fn = isRecord(t.function) ? t.function : t;
      const name = typeof fn.name === 'string' ? fn.name : '';
      if (!name) continue;
      const description = typeof fn.description === 'string' ? fn.description : '';
      const parameters = isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} };
      decls.push({
        name,
        description,
        parameters,
      });
    }
    if (decls.length > 0) {
      tools = [{ functionDeclarations: decls }];
    }
  }

  const generationConfig: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens)
    ? body.max_tokens
    : typeof body.max_completion_tokens === 'number' && Number.isFinite(body.max_completion_tokens)
    ? body.max_completion_tokens
    : null;
  if (maxTokens !== null) {
    generationConfig.maxOutputTokens = maxTokens;
  }

  const temp = typeof body.temperature === 'number' && Number.isFinite(body.temperature) ? body.temperature : null;
  if (temp !== null) {
    generationConfig.temperature = temp;
  }

  const topP = typeof body.top_p === 'number' && Number.isFinite(body.top_p) ? body.top_p : null;
  if (topP !== null) {
    generationConfig.topP = topP;
  }

  const stop = body.stop;
  if (typeof stop === 'string' && stop !== '') {
    generationConfig.stopSequences = [stop];
  } else if (Array.isArray(stop) && stop.length > 0) {
    const stops = stop.filter((s) => typeof s === 'string' && s !== '');
    if (stops.length > 0) {
      generationConfig.stopSequences = stops;
    }
  }

  const rf = body.response_format;
  if (isRecord(rf)) {
    const type = rf.type;
    if (type === 'json_object' || type === 'json_schema') {
      generationConfig.responseMimeType = 'application/json';
    }
  }

  const result: Json = { contents };
  if (systemInstruction) result.systemInstruction = systemInstruction;
  if (tools) result.tools = tools;
  if (Object.keys(generationConfig).length > 0) {
    result.generationConfig = generationConfig;
  }

  return result;
}

// __REQ_FROM_OPENAI__
export function anthropicToGemini(body: Json, _model: string): Json {
  const systemText = textOf(body.system);
  const systemInstruction = systemText ? { parts: [{ text: systemText }] } : null;

  const contents: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let lastUserMsgParts: Json[] | null = null;

  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = typeof m.role === 'string' ? m.role : '';
    const content = m.content;

    if (role === 'user') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') {
            parts.push({ text: block.text });
          } else if (block.type === 'image' && isRecord(block.source)) {
            const src = block.source;
            if (src.type === 'base64' && typeof src.data === 'string') {
              const mimeType = typeof src.media_type === 'string' && src.media_type ? src.media_type : 'image/png';
              parts.push({
                inlineData: {
                  mimeType,
                  data: src.data,
                },
              });
            } else if (src.type === 'url' && typeof src.url === 'string') {
              parts.push({
                fileData: {
                  fileUri: src.url,
                  mimeType: 'image/png',
                },
              });
            }
          }
        }
      }
      if (parts.length === 0) {
        parts.push({ text: '' });
      }
      lastUserMsgParts = parts;
      contents.push({ role: 'user', parts });
    } else if (role === 'assistant') {
      const textParts: string[] = [];
      if (typeof content === 'string') {
        textParts.push(content);
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') {
            textParts.push(block.text);
          }
        }
      }
      const ast = textParts.join('');
      if (ast) {
        contents.push({
          role: 'model',
          parts: [{ text: ast }],
        });
      }
    }
  }

  if (lastUserMsgParts !== null) {
    const lastContent = contents[contents.length - 1];
    let needsPush = true;
    if (isRecord(lastContent) && lastContent.role === 'user' && Array.isArray(lastContent.parts)) {
      if (lastContent.parts === lastUserMsgParts) {
        needsPush = false;
      }
    }
    if (contents.length === 0 || needsPush) {
      contents.push({ role: 'user', parts: lastUserMsgParts });
    }
  } else if (contents.length === 0) {
    contents.push({ role: 'user', parts: [{ text: '' }] });
  }

  let tools: Json[] | undefined = undefined;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const decls: Json[] = [];
    for (const t of body.tools) {
      if (!isRecord(t)) continue;
      const name = typeof t.name === 'string' ? t.name : '';
      if (!name) continue;
      const description = typeof t.description === 'string' ? t.description : '';
      const parameters = isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} };
      decls.push({
        name,
        description,
        parameters,
      });
    }
    if (decls.length > 0) {
      tools = [{ functionDeclarations: decls }];
    }
  }

  const generationConfig: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens) ? body.max_tokens : null;
  if (maxTokens !== null) {
    generationConfig.maxOutputTokens = maxTokens;
  }

  const temp = typeof body.temperature === 'number' && Number.isFinite(body.temperature) ? body.temperature : null;
  if (temp !== null) {
    generationConfig.temperature = temp;
  }

  const topP = typeof body.top_p === 'number' && Number.isFinite(body.top_p) ? body.top_p : null;
  if (topP !== null) {
    generationConfig.topP = topP;
  }

  const stopSequences = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s) => typeof s === 'string' && s !== '') : [];
  if (stopSequences.length > 0) {
    generationConfig.stopSequences = stopSequences;
  }

  const result: Json = { contents };
  if (systemInstruction) result.systemInstruction = systemInstruction;
  if (tools) result.tools = tools;
  if (Object.keys(generationConfig).length > 0) {
    result.generationConfig = generationConfig;
  }

  return result;
}

// __REQ_FROM_ANTHROPIC__
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const firstCandidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(firstCandidate.content) ? firstCandidate.content : {};
  const parts = content.parts;
  const text = partsText(parts);

  const finishReason = typeof firstCandidate.finishReason === 'string' ? firstCandidate.finishReason : 'STOP';

  const usageMetadata = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const promptTokens = typeof usageMetadata.promptTokenCount === 'number' && Number.isFinite(usageMetadata.promptTokenCount)
    ? usageMetadata.promptTokenCount
    : 0;
  const completionTokens = typeof usageMetadata.candidatesTokenCount === 'number' && Number.isFinite(usageMetadata.candidatesTokenCount)
    ? usageMetadata.candidatesTokenCount
    : 0;
  const totalTokens = typeof usageMetadata.totalTokenCount === 'number' && Number.isFinite(usageMetadata.totalTokenCount)
    ? usageMetadata.totalTokenCount
    : promptTokens + completionTokens;

  const usage = {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: totalTokens,
  };

  const modelVersion = typeof body.modelVersion === 'string' && body.modelVersion !== '' ? body.modelVersion : 'sim';

  return {
    id: 'chatcmpl-' + modelVersion,
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: text,
        },
        finish_reason: geminiFinishToOpenAI(finishReason),
        logprobs: null,
      },
    ],
    usage,
  };
}

export function geminiToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const firstCandidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(firstCandidate.content) ? firstCandidate.content : {};
  const parts = content.parts;
  const text = partsText(parts);

  const finishReason = typeof firstCandidate.finishReason === 'string' ? firstCandidate.finishReason : 'STOP';

  const usageMetadata = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const promptTokens = typeof usageMetadata.promptTokenCount === 'number' && Number.isFinite(usageMetadata.promptTokenCount)
    ? usageMetadata.promptTokenCount
    : 0;
  const completionTokens = typeof usageMetadata.candidatesTokenCount === 'number' && Number.isFinite(usageMetadata.candidatesTokenCount)
    ? usageMetadata.candidatesTokenCount
    : 0;

  const usage = {
    input_tokens: promptTokens,
    output_tokens: completionTokens,
  };

  const modelVersion = typeof body.modelVersion === 'string' && body.modelVersion !== '' ? body.modelVersion : 'sim';

  return {
    id: 'msg_' + modelVersion,
    type: 'message',
    role: 'assistant',
    model,
    content: [
      {
        type: 'text',
        text,
      },
    ],
    stop_reason: geminiFinishToAnthropic(finishReason),
    stop_sequence: null,
    usage,
  };
}

// __RESP_TO_CLIENT__
export async function* geminiEventsToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  let initialChunkEmitted = false;
  let lastFinishReason = 'STOP';
  let promptTokens = 0;
  let completionTokens = 0;
  let totalTokens = 0;
  let hasUsage = false;

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

        let frame: Json;
        try {
          const parsed = JSON.parse(dataStr);
          if (isRecord(parsed)) {
            frame = parsed;
          } else {
            continue;
          }
        } catch {
          continue;
        }

        const usageMetadata = isRecord(frame.usageMetadata) ? frame.usageMetadata : null;
        if (usageMetadata) {
          hasUsage = true;
          if (typeof usageMetadata.promptTokenCount === 'number' && Number.isFinite(usageMetadata.promptTokenCount)) {
            promptTokens = usageMetadata.promptTokenCount;
          }
          if (typeof usageMetadata.candidatesTokenCount === 'number' && Number.isFinite(usageMetadata.candidatesTokenCount)) {
            completionTokens = usageMetadata.candidatesTokenCount;
          }
          if (typeof usageMetadata.totalTokenCount === 'number' && Number.isFinite(usageMetadata.totalTokenCount)) {
            totalTokens = usageMetadata.totalTokenCount;
          }
        }

        const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
        for (const cand of candidates) {
          if (!isRecord(cand)) continue;
          const fr = cand.finishReason;
          if (typeof fr === 'string' && fr !== '') {
            lastFinishReason = fr;
          }
          const content = isRecord(cand.content) ? cand.content : {};
          const cParts = Array.isArray(content.parts) ? content.parts : [];
          const piece = partsText(cParts);
          if (piece) {
            if (!initialChunkEmitted) {
              initialChunkEmitted = true;
              yield 'data: ' + JSON.stringify({
                id: 'chatcmpl-sim',
                object: 'chat.completion.chunk',
                created: 1767225600,
                model,
                choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
              }) + '\n\n';
            }
            yield 'data: ' + JSON.stringify({
              choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
            }) + '\n\n';
          }
        }
      }
    }
  } catch {
    // Never throw; swallow stream errors
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // ignore
    }
  }

  if (!initialChunkEmitted) {
    initialChunkEmitted = true;
    yield 'data: ' + JSON.stringify({
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: 1767225600,
      model,
      choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
    }) + '\n\n';
  }

  yield 'data: ' + JSON.stringify({
    choices: [{ index: 0, delta: {}, finish_reason: geminiFinishToOpenAI(lastFinishReason) }],
  }) + '\n\n';

  const finalTotal = hasUsage && totalTokens > 0 ? totalTokens : promptTokens + completionTokens;
  yield 'data: ' + JSON.stringify({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: finalTotal,
    },
  }) + '\n\n';

  yield 'data: [DONE]\n\n';
}

export async function* geminiEventsToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  let messageStarted = false;
  let textBlockStarted = false;
  let lastFinishReason = 'STOP';
  let promptTokens = 0;
  let completionTokens = 0;

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

        let frame: Json;
        try {
          const parsed = JSON.parse(dataStr);
          if (isRecord(parsed)) {
            frame = parsed;
          } else {
            continue;
          }
        } catch {
          continue;
        }

        const usageMetadata = isRecord(frame.usageMetadata) ? frame.usageMetadata : null;
        if (usageMetadata) {
          if (typeof usageMetadata.promptTokenCount === 'number' && Number.isFinite(usageMetadata.promptTokenCount)) {
            promptTokens = usageMetadata.promptTokenCount;
          }
          if (typeof usageMetadata.candidatesTokenCount === 'number' && Number.isFinite(usageMetadata.candidatesTokenCount)) {
            completionTokens = usageMetadata.candidatesTokenCount;
          }
        }

        const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
        for (const cand of candidates) {
          if (!isRecord(cand)) continue;
          const fr = cand.finishReason;
          if (typeof fr === 'string' && fr !== '') {
            lastFinishReason = fr;
          }
          const content = isRecord(cand.content) ? cand.content : {};
          const cParts = Array.isArray(content.parts) ? content.parts : [];
          const piece = partsText(cParts);

          if (!messageStarted) {
            messageStarted = true;
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
                usage: { input_tokens: promptTokens, output_tokens: 0 },
              },
            });
          }

          if (piece) {
            if (!textBlockStarted) {
              textBlockStarted = true;
              yield sseEvent('content_block_start', {
                type: 'content_block_start',
                index: 0,
                content_block: { type: 'text', text: '' },
              });
            }
            yield sseEvent('content_block_delta', {
              type: 'content_block_delta',
              index: 0,
              delta: { type: 'text_delta', text: piece },
            });
          }
        }
      }
    }
  } catch {
    // Never throw
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // ignore
    }
  }

  if (!messageStarted) {
    messageStarted = true;
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
        usage: { input_tokens: promptTokens, output_tokens: 0 },
      },
    });
  }

  if (textBlockStarted) {
    yield sseEvent('content_block_stop', {
      index: 0,
    });
  }

  yield sseEvent('message_delta', {
    type: 'message_delta',
    delta: {
      stop_reason: geminiFinishToAnthropic(lastFinishReason),
      stop_sequence: null,
    },
    usage: {
      output_tokens: completionTokens,
    },
  });

  yield sseEvent('message_stop', {
    type: 'message_stop',
  });
}

// __SSE__
// __END__
