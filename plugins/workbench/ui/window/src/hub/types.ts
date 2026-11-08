/**
 * The wire shapes the hub reads, mirrored from the protocol crate: `serve@1`
 * (protocol/src/contracts/serve.rs), the mesh card (contracts/mesh.rs) and the frame's lanes
 * (contracts/frame.rs). Fields the Rust side marks `#[serde(default)]` are optional here.
 */

export interface Gpu {
  index: number;
  name: string;
  backend: string;
  memory_bytes: number;
  used_bytes?: number;
  util?: number;
  power_w?: number;
  temp_c?: number;
  unified?: boolean;
  uuid?: string;
  cores?: number;
}

export interface RecipeKeys {
  slots?: string; ctx?: string; cache_ram?: string; checkpoints?: string;
  kv_type?: string; speculative?: string; idle_unload_s?: string; memory_limit?: string;
}

export interface Recipe {
  id: string;
  name: string;
  engine: string;
  origin: string;
  kind: string;
  family?: string;
  tags?: string[];
  platforms?: string[];
  backends?: string[];
  repo?: string;
  revision?: string;
  weights_bytes?: number;
  compute_bytes?: number;
  kv_bytes_per_token?: number;
  min_gpu_bytes?: number;
  ctx_default?: number;
  ctx_max?: number;
  ctx_per_slot?: boolean;
  ctx_min_per_slot?: number;
  slots_default?: number;
  slots_max?: number;
  vision?: boolean;
  speculative?: string;
  keys?: RecipeKeys;
  auto?: boolean;
  available?: boolean;
  reason?: string;
  downloaded?: boolean;
}

export interface LaneSettings {
  id?: string;
  recipe: string;
  gpu?: number;
  slots?: number;
  ctx?: number;
  cache_ram_mib?: number;
  checkpoints?: number;
  kv_type?: string;
  speculative?: boolean;
  idle_unload_s?: number;
  pinned?: boolean;
  lend?: boolean;
  tags?: string[];
  manual?: boolean;
}

export interface LaneView {
  settings: LaneSettings;
  name: string;
  engine: string;
  origin?: string;
  kind: string;
  family?: string;
  quant?: string;
  state: string;
  loaded?: boolean;
  slots?: number;
  slots_busy?: number;
  queued?: number;
  ctx?: number;
  tok_s?: number;
  tokens_hour?: number;
  requests_hour?: number;
  memory_bytes?: number;
  error?: string;
}

export interface LanesView { lanes: LaneView[] }

export interface Fit {
  fits: boolean;
  weights_bytes: number;
  kv_bytes: number;
  compute_bytes: number;
  total_bytes: number;
  gpu_bytes: number;
  others_bytes: number;
  max_slots: number;
  max_ctx: number;
  note: string;
}

export interface PlanChange { op: 'add' | 'remove' | 'change' | string; lane: LaneSettings; why: string }
export interface Plan { lanes: LaneSettings[]; changes: PlanChange[]; notes: string[] }

export interface Calibration {
  lane: string;
  recipe: string;
  decode_tok_s: number;
  prefill_tok_s: number;
  aggregate_tok_s: number;
  ttft_ms: number;
  at_ms: number;
  error?: string;
}

export interface CalibrationResult { results: Calibration[] }

export interface Engine { id: string; version: string; enabled: boolean }
export interface Want { app: string; model_id: string; family: string; tags: string[]; recipe?: string; loaded?: boolean }

export interface Totals {
  gpu_util: number;
  gpu_used_bytes: number;
  gpu_bytes: number;
  tok_s: number;
  prefill_tok_s: number;
  tokens_hour: number;
  tokens_day: number;
  requests_min: number;
  ttft_ms: number;
  slots_busy: number;
  slots: number;
  queued: number;
  lanes_serving: number;
  lanes_loading: number;
  lanes_failed: number;
  power_w: number;
}

export interface ServeStatus {
  node: string;
  name: string;
  platform: string;
  gpus: Gpu[];
  lanes: LaneView[];
  engines: Engine[];
  wants: Want[];
  totals?: Totals;
}

export interface ServeSample { at_ms: number; gpus: Gpu[]; lanes: LaneView[]; totals: Totals }

export interface Cluster {
  totals: Totals;
  servers_online: number;
  servers: number;
  reporting: number;
  gpus: number;
  updated_ms: number;
}

export interface HistoryPoint { at_ms: number; totals: Totals }
export interface History { points: HistoryPoint[] }

/** `frame.lanes.changed` / `frame.lanes.list`. */
export interface LaneState {
  id: string;
  plugin: string;
  version?: string;
  instance?: string;
  state: string;
  pid?: number;
  started_ms?: number;
  restarts?: number;
}

// --- the mesh -------------------------------------------------------------------------------

export interface ModelCard {
  plugin: string;
  model_id: string;
  variant: string;
  family: string;
  tags?: string[];
  ctx_total: number;
  slots: number;
  slots_busy: number;
  busy_foreground?: number;
  queued: number;
  tok_s?: number;
  loaded?: boolean;
  kind?: string;
  engine?: string;
  recipe?: string;
  gpu?: number | null;
  tokens_hour?: number;
}

export interface GpuCard { index: number; name: string; backend: string; memory_bytes: number; used_bytes?: number; util?: number; power_w?: number; unified?: boolean }

export interface ServingCard {
  tok_s: number;
  prefill_tok_s?: number;
  tokens_hour: number;
  tokens_day?: number;
  requests_min?: number;
  ttft_ms?: number;
  lanes_serving?: number;
  lanes_loading?: number;
  lanes_failed?: number;
}

export interface Card {
  node: string;
  name: string;
  software: string;
  version: string;
  chip: string;
  memory_bytes: number;
  cpu_cores?: number;
  gpu_cores?: number;
  os_version?: string;
  models?: ModelCard[];
  gpus?: GpuCard[];
  serving?: ServingCard | null;
  updated_ms?: number;
}

export interface Peer {
  member: { node: string; name: string; local?: boolean };
  card?: Card | null;
  reachable: boolean;
  path: string;
  rtt_ms?: number;
}

/** One computer of the grid: this one (node `''`) or a mesh member. */
export interface Computer { node: string; name: string; reachable: boolean; path: string; card?: Card | null }

export const EMPTY_TOTALS: Totals = {
  gpu_util: 0, gpu_used_bytes: 0, gpu_bytes: 0, tok_s: 0, prefill_tok_s: 0, tokens_hour: 0, tokens_day: 0,
  requests_min: 0, ttft_ms: 0, slots_busy: 0, slots: 0, queued: 0, lanes_serving: 0, lanes_loading: 0, lanes_failed: 0, power_w: 0,
};
