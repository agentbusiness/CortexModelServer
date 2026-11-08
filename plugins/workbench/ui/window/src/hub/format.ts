/** Units the hub writes: memory in GB, contexts as `32k`, token counts in millions. */

import { RpcError } from '@cortex/sdk';
import { GB } from './api';

export const gb = (bytes: number | undefined, digits = 1) => ((bytes ?? 0) / GB).toFixed(digits);

/** `32 GB`, `24 GB`, `192 GB`: a GPU's size, whole. */
export const gbWhole = (bytes: number | undefined) => `${Math.round((bytes ?? 0) / GB)} GB`;

/** `8k`, `32k`, `256k`, `1M`. */
export function ctxLabel(tokens: number | undefined): string {
  if (!tokens) return '—';
  if (tokens >= 1_048_576 && tokens % 1_048_576 === 0) return `${tokens / 1_048_576}M`;
  if (tokens >= 1024) return `${Math.round(tokens / 1024)}k`;
  return String(tokens);
}

/** `1.2`, `0.3`, `14`: millions, for `M tokens/h`. */
export function millions(count: number | undefined): string {
  const value = (count ?? 0) / 1_000_000;
  if (value === 0) return '0';
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(value < 0.1 ? 2 : 1);
}

export const tokS = (value: number | undefined) => (value ? (value >= 100 ? value.toFixed(0) : value.toFixed(1)) : '—');

export const pct = (share: number | undefined) => `${Math.round((share ?? 0) * 100)}%`;

/** The serve plugin does not answer on that computer: no cortexmodelserver there. */
export function noServer(cause: unknown): boolean {
  return cause instanceof RpcError && (cause.code === 'unknown_method' || cause.code === 'unavailable' || cause.code === 'not_found');
}

/** Context sizes the editor offers. */
export const CONTEXTS = [8192, 16384, 32768, 65536, 131072, 262144];

/** The state of a lane as its pip shows it. */
export type LaneTone = 'serving' | 'loading' | 'unloaded' | 'error';

export function laneTone(state: string, loaded: boolean | undefined, error?: string): LaneTone {
  const s = (state || '').toLowerCase();
  if (error || s.startsWith('failed') || s === 'crashed' || s === 'error') return 'error';
  if (s === 'starting' || s === 'probing' || s === 'restarting' || s === 'loading' || s === 'draining' || s === 'reloading') return 'loading';
  if (s === 'active' && loaded !== false) return 'serving';
  return 'unloaded';
}

export const TONE_LABEL: Record<LaneTone, string> = { serving: 'SERVING', loading: 'LOADING', unloaded: 'IDLE', error: 'ERROR' };
