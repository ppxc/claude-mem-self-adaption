// SPDX-License-Identifier: Apache-2.0

import { getCredential } from '../../shared/EnvManager.js';
import { SettingsDefaultsManager } from '../../shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../shared/paths.js';
import { logger } from '../../utils/logger.js';
import type { ActiveSession, ConversationMessage } from '../worker-types.js';
import { DatabaseManager } from './DatabaseManager.js';
import { SessionManager } from './SessionManager.js';
import { ClassifiedProviderError } from './provider-errors.js';
import { withRetry, parseRetryAfterMs } from './retry.js';
import { OpenAICompatibleProvider, type ProviderQueryResult } from './OpenAICompatibleProvider.js';

/**
 * Anthropic Messages API client configuration.
 *
 * Uses the Anthropic Messages API format (POST /v1/messages) with a
 * configurable base URL. This allows connecting to any Anthropic-compatible
 * endpoint (proxies, LiteLLM, third-party services) via
 * CLAUDE_MEM_ANTHROPIC_BASE_URL. Distinct from the `claude` provider which
 * uses the Agent SDK subprocess.
 */

const ANTHROPIC_VERSION = '2023-06-01';
const DEFAULT_MAX_CONTEXT_MESSAGES = 20;
const DEFAULT_MAX_ESTIMATED_TOKENS = 100000;
const CHARS_PER_TOKEN_ESTIMATE = 4;

/**
 * Classify an Anthropic API fetch failure into ClassifiedProviderError.
 * Mirrors classifyClaudeServerError from the server-beta ClaudeObservationProvider
 * but adapted for the worker ClassifiedProviderError type.
 */
export function classifyAnthropicApiError(input: {
  status?: number;
  bodyText?: string;
  headers?: Headers | { get(name: string): string | null };
  cause: unknown;
}): ClassifiedProviderError {
  const status = input.status;
  const body = input.bodyText ?? '';
  const lower = body.toLowerCase();
  const retryAfterMs = input.headers ? parseRetryAfterMs(input.headers.get('retry-after')) : undefined;

  if (lower.includes('overloaded')) {
    return new ClassifiedProviderError(
      `Anthropic overloaded${status !== undefined ? ` (status ${status})` : ''}`,
      { kind: 'transient', cause: input.cause },
    );
  }

  if (status === 401 || status === 403 || lower.includes('invalid api key')) {
    return new ClassifiedProviderError(
      `Anthropic auth invalid${status !== undefined ? ` (status ${status})` : ''}`,
      { kind: 'auth_invalid', cause: input.cause },
    );
  }

  if (status === 429) {
    return new ClassifiedProviderError('Anthropic rate limit (429)', {
      kind: 'rate_limit',
      cause: input.cause,
      ...(retryAfterMs !== undefined ? { retryAfterMs } : {}),
    });
  }

  if (lower.includes('quota exceeded')) {
    return new ClassifiedProviderError('Anthropic quota exhausted', {
      kind: 'quota_exhausted',
      cause: input.cause,
    });
  }

  if (
    lower.includes('prompt is too long') ||
    lower.includes('context window') ||
    lower.includes('max_tokens')
  ) {
    return new ClassifiedProviderError('Anthropic context overflow', {
      kind: 'unrecoverable',
      cause: input.cause,
    });
  }

  if (status === 529) {
    return new ClassifiedProviderError('Anthropic overloaded (529)', {
      kind: 'transient',
      cause: input.cause,
    });
  }

  if (status !== undefined && status >= 500 && status < 600) {
    return new ClassifiedProviderError(`Anthropic upstream error (status ${status})`, {
      kind: 'transient',
      cause: input.cause,
    });
  }

  if (status === 400) {
    return new ClassifiedProviderError('Anthropic bad request (400)', {
      kind: 'unrecoverable',
      cause: input.cause,
    });
  }

  if (status === undefined) {
    const message = input.cause instanceof Error ? input.cause.message : String(input.cause);
    return new ClassifiedProviderError(`Anthropic network error: ${message}`, {
      kind: 'transient',
      cause: input.cause,
    });
  }

  return new ClassifiedProviderError(
    `Anthropic API error: ${status}${body ? ` - ${body.substring(0, 200)}` : ''}`,
    { kind: 'unrecoverable', cause: input.cause },
  );
}

interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface AnthropicResponse {
  content?: Array<{ type?: string; text?: string }>;
  usage?: { input_tokens?: number; output_tokens?: number };
  error?: { type?: string; message?: string };
}

interface AnthropicConfig {
  apiKey: string;
  model: string;
  apiUrl: string;
}

export class AnthropicApiProvider extends OpenAICompatibleProvider<AnthropicConfig> {
  protected readonly providerName = 'Anthropic';
  protected readonly syntheticIdPrefix = 'anthropic';
  protected readonly requireNonEmptyToTruncate = false;
  protected readonly forwardEmptyMessageResponse = true;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    super(dbManager, sessionManager);
  }

  protected getConfig(): AnthropicConfig {
    return this.getAnthropicConfig();
  }

  protected missingApiKeyError(): Error {
    return new Error('Anthropic API key not configured. Set CLAUDE_MEM_ANTHROPIC_API_KEY in settings or ANTHROPIC_API_KEY environment variable.');
  }

  protected prepareSessionExtras(session: ActiveSession, _config: AnthropicConfig): void {
    session.endpointClass = 'anthropic';
  }

  protected estimateTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN_ESTIMATE);
  }

  protected buildLastUsage(result: ProviderQueryResult): ActiveSession['lastUsage'] {
    if (typeof result.inputTokens !== 'number' || typeof result.outputTokens !== 'number') {
      return null;
    }
    return {
      input: result.inputTokens,
      output: result.outputTokens,
    };
  }

  protected truncateHistoryForAnthropic(history: ConversationMessage[]): ConversationMessage[] {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
    const MAX_CONTEXT_MESSAGES = parseInt(settings.CLAUDE_MEM_ANTHROPIC_MAX_CONTEXT_MESSAGES) || DEFAULT_MAX_CONTEXT_MESSAGES;
    const MAX_ESTIMATED_TOKENS = parseInt(settings.CLAUDE_MEM_ANTHROPIC_MAX_TOKENS) || DEFAULT_MAX_ESTIMATED_TOKENS;
    return this.truncateHistory(history, MAX_CONTEXT_MESSAGES, MAX_ESTIMATED_TOKENS);
  }

  private conversationToAnthropicMessages(history: ConversationMessage[]): AnthropicMessage[] {
    return history.map(msg => ({
      role: msg.role === 'assistant' ? 'assistant' as const : 'user' as const,
      content: msg.content,
    }));
  }

  protected async query(history: ConversationMessage[], config: AnthropicConfig): Promise<ProviderQueryResult> {
    return this.queryAnthropicMultiTurn(history, config.apiKey, config.model, config.apiUrl);
  }

  private async queryAnthropicMultiTurn(
    history: ConversationMessage[],
    apiKey: string,
    model: string,
    apiUrl: string,
  ): Promise<ProviderQueryResult> {
    const truncatedHistory = this.truncateHistoryForAnthropic(history);
    const messages = this.conversationToAnthropicMessages(truncatedHistory);
    const totalChars = truncatedHistory.reduce((sum, m) => sum + m.content.length, 0);
    const estimatedTokens = this.estimateTokens(truncatedHistory.map(m => m.content).join(''));

    logger.debug('SDK', `Querying Anthropic API multi-turn (${model})`, {
      turns: truncatedHistory.length,
      totalChars,
      estimatedTokens,
      baseUrl: apiUrl,
    });

    const data = await withRetry<AnthropicResponse>(async (attemptSignal) => {
      let response: Response;
      try {
        response = await fetch(`${apiUrl}/v1/messages`, {
          method: 'POST',
          headers: {
            'x-api-key': apiKey,
            'anthropic-version': ANTHROPIC_VERSION,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model,
            max_tokens: 4096,
            temperature: 0.3,
            messages,
          }),
          signal: attemptSignal,
        });
      } catch (networkError: unknown) {
        throw classifyAnthropicApiError({ cause: networkError });
      }

      if (!response.ok) {
        const errorText = await response.text();
        throw classifyAnthropicApiError({
          status: response.status,
          bodyText: errorText,
          headers: response.headers,
          cause: new Error(`Anthropic API error: ${response.status} - ${errorText}`),
        });
      }

      const responseData = await response.json() as AnthropicResponse;

      if (responseData.error) {
        throw classifyAnthropicApiError({
          status: response.status,
          bodyText: `${responseData.error.type ?? ''} ${responseData.error.message ?? ''}`,
          headers: response.headers,
          cause: new Error(`Anthropic API error: ${responseData.error.type} - ${responseData.error.message}`),
        });
      }

      return responseData;
    }, { label: `Anthropic ${model}` });

    const blocks = Array.isArray(data.content) ? data.content : [];
    const content = blocks
      .filter(block => block?.type === 'text' && typeof block.text === 'string')
      .map(block => block.text!)
      .join('\n')
      .trim();

    if (!content) {
      logger.warn('SDK', 'Anthropic returned empty content', {
        provider: 'anthropic',
        model,
      });
    }

    const usage = data.usage;
    const inputTokens = typeof usage?.input_tokens === 'number' ? usage.input_tokens : undefined;
    const outputTokens = typeof usage?.output_tokens === 'number' ? usage.output_tokens : undefined;
    const tokensUsed =
      inputTokens !== undefined || outputTokens !== undefined
        ? (inputTokens ?? 0) + (outputTokens ?? 0)
        : undefined;

    if (tokensUsed) {
      logger.info('SDK', 'Anthropic API usage', {
        model,
        inputTokens: inputTokens || 0,
        outputTokens: outputTokens || 0,
        totalTokens: tokensUsed,
        messagesInContext: truncatedHistory.length,
      });
    }

    return { content, tokensUsed, inputTokens, outputTokens };
  }

  private getAnthropicConfig(): AnthropicConfig {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const apiKey = settings.CLAUDE_MEM_ANTHROPIC_API_KEY || getCredential('ANTHROPIC_API_KEY') || getCredential('CLAUDE_MEM_ANTHROPIC_API_KEY') || '';
    const model = settings.CLAUDE_MEM_ANTHROPIC_MODEL || 'claude-sonnet-4-6';

    // Base URL: settings value wins, then ANTHROPIC_BASE_URL env var, else default.
    const baseUrl = settings.CLAUDE_MEM_ANTHROPIC_BASE_URL || process.env.ANTHROPIC_BASE_URL || 'https://api.anthropic.com';
    // Strip trailing slash for consistent URL construction
    const apiUrl = baseUrl.replace(/\/+$/, '');

    return { apiKey, model, apiUrl };
  }
}

export function isAnthropicAvailable(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return !!(
    settings.CLAUDE_MEM_ANTHROPIC_API_KEY ||
    getCredential('ANTHROPIC_API_KEY') ||
    getCredential('CLAUDE_MEM_ANTHROPIC_API_KEY')
  );
}

export function isAnthropicSelected(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return settings.CLAUDE_MEM_PROVIDER === 'anthropic';
}
