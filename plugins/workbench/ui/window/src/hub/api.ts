/** The machine readings and alert shapes the cards show (the `system` plugin), and their units. */

export interface Metrics {
  at: number;
  machine: string;
  memory_total: number;
  cpu: number;
  cores: number[];
  p_cores: number;
  e_cores: number;
  memory_used: number;
  memory_compressed: number;
  swap_used: number;
  pressure: string;
  gpu?: number | null;
  gpu_active?: number | null;
  cpu_power?: number | null;
  gpu_power?: number | null;
  ane_power?: number | null;
  cpu_temp?: number | null;
  gpu_temp?: number | null;
  fan_rpm?: number | null;
  thermal: string;
  disk_total: number;
  disk_free: number;
  disk_read: number;
  disk_write: number;
  net_rx: number;
  net_tx: number;
  battery?: { percent: number; charging: boolean; watts?: number | null } | null;
  load: [number, number, number];
}

export interface Alert {
  id: string;
  kind: string;
  severity: string;
  text: string;
  at: number;
  pid?: number | null;
  port?: number | null;
  path?: string | null;
  actions: string[];
  dismissed: boolean;
}

export interface Finding {
  severity: string;
  kind: string;
  text: string;
  fix?: string | null;
  source: string;
  /** Stable across sweeps. */
  key: string;
  /** A decision covers it. */
  ignored?: boolean;
  /** What it is about: findings are grouped and decided by it. */
  group?: string;
  /** The covering decision's verdict; empty when open. */
  decision?: string;
  /** A fix command that ran and did not clear it. */
  tried?: string | null;
  /** The model's second look found it worse than its rating; raised one level. */
  raised?: string;
}

/** The person's decision about a group of findings (`system.finding.decide`). */
export interface Decision {
  /** expected | false_alarm | accepted | vendor */
  verdict: string;
  group: string;
  /** Only these findings; empty: the whole group up to `severity`. */
  keys: string[];
  severity: string;
  at: number;
  /** 0: no end. */
  until: number;
}

export interface SweepReport {
  at: number;
  duration_ms: number;
  findings: Finding[];
  checked: string[];
  skipped: string[];
  decisions?: Decision[];
  /** The latest checks after a fix, newest first. */
  fixes?: { at: number; group: string; text: string; command: string; gone: boolean }[];
}


export const GB = 1_073_741_824;
export const MB = 1_048_576;

export function size(bytes: number): string {
  if (bytes >= GB) return `${(bytes / GB).toFixed(1)} GB`;
  if (bytes >= MB) return `${Math.round(bytes / MB)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

export function rate(bytes: number): string {
  if (bytes >= MB) return `${(bytes / MB).toFixed(1)} MB/s`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB/s`;
  return `${bytes} B/s`;
}

export interface SecurityObservation {
  id: string; at: number; source: string; kind: string; severity: string; summary: string; rule: string; subject: string;
  pid?: number | null; process_started?: number | null; parent_pid?: number | null; executable?: string | null;
  process_started_us?: number | null; process_version?: number | null;
  path?: string | null; remote?: string | null; port?: number | null; count: number;
}
export interface SecurityClaim { text: string; evidence: string[] }
export interface SecurityInvestigation {
  state: string; summary: SecurityClaim[]; alternatives: SecurityClaim[]; unknowns: string[]; next_steps: string[];
  checked_at: number; error?: string | null; observations: SecurityObservation[];
}
export interface SecurityIncident {
  id: string; key: string; first_at: number; last_at: number; title: string; severity: string; state: string;
  observations: SecurityObservation[]; investigation?: SecurityInvestigation | null; revision: number; omitted_observations?: number;
}
