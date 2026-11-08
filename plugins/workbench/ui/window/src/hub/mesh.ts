/**
 * The mesh added up from what each computer reports: its live server view when it runs
 * cortexmodelserver (`serve.sample`), else what its mesh card says (`gpus`, `serving`, `models`).
 */

import { serveOf } from './serve-data';
import { EMPTY_TOTALS, type Computer, type Gpu, type Totals } from './types';

export const LOCAL_COMPUTER: Computer = { node: '', name: 'This computer', reachable: true, path: 'local' };

/** The computer runs cortexmodelserver (it answers `serve.*`, or its card says so). */
export function isServer(computer: Computer): boolean {
  const view = serveOf(computer.node);
  if (view.status || view.sampleAt) return true;
  if (view.absent) return false;
  return computer.card?.software === 'cortexmodelserver';
}

export function gpusOf(computer: Computer): Gpu[] {
  const view = serveOf(computer.node);
  if (view.gpus.length) return view.gpus;
  return (computer.card?.gpus ?? []).map((gpu) => ({ ...gpu }));
}

/** A computer's throughput and load now. */
export function totalsOf(computer: Computer): Totals {
  const view = serveOf(computer.node);
  if (view.totals && computer.reachable) return view.totals;
  const gpus = gpusOf(computer);
  const serving = computer.card?.serving;
  const models = computer.card?.models ?? [];
  const memory = gpus.reduce((sum, gpu) => sum + gpu.memory_bytes, 0);
  return {
    ...EMPTY_TOTALS,
    gpu_bytes: memory,
    gpu_used_bytes: gpus.reduce((sum, gpu) => sum + (gpu.used_bytes ?? 0), 0),
    gpu_util: memory ? gpus.reduce((sum, gpu) => sum + (gpu.util ?? 0) * gpu.memory_bytes, 0) / memory : 0,
    power_w: gpus.reduce((sum, gpu) => sum + (gpu.power_w ?? 0), 0),
    tok_s: serving?.tok_s ?? models.reduce((sum, model) => sum + (model.slots_busy ? model.tok_s ?? 0 : 0), 0),
    prefill_tok_s: serving?.prefill_tok_s ?? 0,
    tokens_hour: serving?.tokens_hour ?? models.reduce((sum, model) => sum + (model.tokens_hour ?? 0), 0),
    tokens_day: serving?.tokens_day ?? 0,
    requests_min: serving?.requests_min ?? 0,
    ttft_ms: serving?.ttft_ms ?? 0,
    slots: models.reduce((sum, model) => sum + model.slots, 0),
    slots_busy: models.reduce((sum, model) => sum + model.slots_busy, 0),
    queued: models.reduce((sum, model) => sum + model.queued, 0),
    lanes_serving: serving?.lanes_serving ?? models.filter((model) => model.loaded).length,
    lanes_loading: serving?.lanes_loading ?? 0,
    lanes_failed: serving?.lanes_failed ?? 0,
  };
}

/** Every computer's totals added up; GPU load weighted by each GPU's memory. */
export function sumTotals(list: Totals[]): Totals {
  const sum = { ...EMPTY_TOTALS };
  let weighted = 0;
  let ttftWeight = 0;
  for (const totals of list) {
    for (const key of Object.keys(sum) as (keyof Totals)[]) if (key !== 'gpu_util' && key !== 'ttft_ms') sum[key] += totals[key] ?? 0;
    weighted += (totals.gpu_util ?? 0) * (totals.gpu_bytes ?? 0);
    if (totals.ttft_ms && totals.requests_min) { sum.ttft_ms += totals.ttft_ms * totals.requests_min; ttftWeight += totals.requests_min; }
  }
  sum.gpu_util = sum.gpu_bytes ? weighted / sum.gpu_bytes : 0;
  sum.ttft_ms = ttftWeight ? sum.ttft_ms / ttftWeight : 0;
  return sum;
}

export type ServerState = 'offline' | 'serving' | 'idle';
export function stateOf(computer: Computer): ServerState {
  if (!computer.reachable) return 'offline';
  const totals = totalsOf(computer);
  return totals.slots_busy > 0 || totals.tok_s > 0.5 ? 'serving' : 'idle';
}

/** One model running somewhere: a lane of a server, or a model of a card. */
export interface Running { node: string; name: string; engine: string; slots: number; busy: number; gpu: number | null; tok_s: number; queued: number; tokens_hour: number; loaded: boolean }

export function runningOf(computer: Computer): Running[] {
  const view = serveOf(computer.node);
  if (!view.absent && view.lanes.length) {
    return view.lanes.map((lane) => ({ node: computer.node, name: lane.name || lane.settings.recipe, engine: lane.engine, slots: lane.slots ?? 0, busy: lane.slots_busy ?? 0, gpu: lane.settings.gpu ?? 0,
      tok_s: lane.tok_s ?? 0, queued: lane.queued ?? 0, tokens_hour: lane.tokens_hour ?? 0, loaded: lane.loaded !== false }));
  }
  return (computer.card?.models ?? []).map((model) => ({
    node: computer.node, name: modelName(model), engine: model.engine || model.plugin, slots: model.slots, busy: model.slots_busy, gpu: model.gpu ?? null,
    tok_s: model.slots_busy ? model.tok_s ?? 0 : 0, queued: model.queued ?? 0, tokens_hour: model.tokens_hour ?? 0, loaded: model.loaded !== false,
  }));
}

export function modelName(model: { variant: string; model_id: string; family: string }): string {
  const raw = model.model_id || model.variant || model.family;
  return raw.replace(/[-_]/g, ' ');
}

/** The number a summary tile sorts the grid by. */
export type SortKey = 'gpu_load' | 'gpu_memory' | 'tok_s' | 'tokens_hour' | 'requests' | 'slots' | 'servers' | 'power';

export function sortValue(computer: Computer, key: SortKey): number {
  const totals = totalsOf(computer);
  switch (key) {
    case 'gpu_load': return totals.gpu_util;
    case 'gpu_memory': return totals.gpu_used_bytes;
    case 'tok_s': return totals.tok_s;
    case 'tokens_hour': return totals.tokens_hour;
    case 'requests': return totals.requests_min;
    case 'slots': return totals.slots_busy * 1000 + totals.slots;
    case 'servers': return (computer.reachable ? 2 : 0) + (isServer(computer) ? 1 : 0);
    case 'power': return totals.power_w;
  }
}
