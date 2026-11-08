/**
 * One live view of each computer's model server, shared by its card, the sidebar, the summary
 * strip and the editor. Through that computer's runtime (`runtimeFor(node)`, a mesh member through
 * `mesh.link`) it reads `serve.status.get` (GPUs, lanes, engines, wants), follows the
 * `serve.sample` stream (about every 2 s), falls back to polling `serve.lanes.list` every 10 s, and
 * listens to `serve.changed` and `frame.lanes.changed` for the activity log. A computer whose frame
 * has no serve plugin (`unknown_method` / `unavailable`) is marked `absent`: its card shows the
 * models of its mesh card, read-only.
 */

import { useSyncExternalStore } from 'react';
import { errorMessage, runtimeFor } from '@cortex/sdk';
import { laneTone, noServer } from './format';
import type { Gpu, LaneState, LaneView, LanesView, ServeSample, ServeStatus, Totals } from './types';

export interface ServeSnapshot {
  status: ServeStatus | null;
  lanes: LaneView[];
  gpus: Gpu[];
  /** This computer's totals now (the latest sample, else the status). */
  totals: Totals | null;
  /** When the latest sample arrived (0: none yet). */
  sampleAt: number;
  absent: boolean;
  loading: boolean;
  problem: string | null;
}

const EMPTY: ServeSnapshot = { status: null, lanes: [], gpus: [], totals: null, sampleAt: 0, absent: false, loading: true, problem: null };

// --- the activity log: lane events of the whole mesh, newest first, the last 50 -----------------

export interface ActivityEvent { id: number; at: number; node: string; computer: string; lane: string; text: string; tone: 'ok' | 'warn' | 'faint' }

let activity: ActivityEvent[] = [];
let activityId = 0;
const activityListeners = new Set<() => void>();
function logActivity(event: Omit<ActivityEvent, 'id' | 'at'>) {
  activity = [{ ...event, id: ++activityId, at: Date.now() }, ...activity].slice(0, 50);
  for (const listener of activityListeners) listener();
}
export function useActivity(): ActivityEvent[] {
  return useSyncExternalStore((listener) => { activityListeners.add(listener); return () => activityListeners.delete(listener); }, () => activity);
}

// --- every store changing bumps one version, for the views that add the mesh up -------------------

let version = 0;
const versionListeners = new Set<() => void>();
function bump() {
  version++;
  for (const listener of versionListeners) listener();
}
const subscribeVersion = (listener: () => void) => { versionListeners.add(listener); return () => { versionListeners.delete(listener); }; };
/** Re-render whenever any computer's server view changes. */
export function useServeVersion(): number {
  return useSyncExternalStore(subscribeVersion, () => version);
}
/** Re-render only when what `select` reads from the server views changes (a string key). */
export function useServeSelect(select: () => string): string {
  return useSyncExternalStore(subscribeVersion, select);
}

const names = new Map<string, string>();
/** The name the activity log gives a computer. */
export function nameComputer(node: string, name: string) { names.set(node, name); }

class ServeStore {
  private snapshot: ServeSnapshot = EMPTY;
  private listeners = new Set<() => void>();
  private stop: (() => void) | null = null;
  private generation = 0;
  private frameLanes = new Map<string, string>();

  constructor(private readonly node: string) {}

  get = () => this.snapshot;

  private set(change: Partial<ServeSnapshot>) {
    this.snapshot = { ...this.snapshot, ...change };
    for (const listener of this.listeners) listener();
    bump();
  }

  private get client() { return runtimeFor(this.node || null).client; }
  private get computer() { return names.get(this.node) ?? (this.node ? 'a mesh computer' : 'This computer'); }

  /** `serve.status.get`: everything at once; tells whether the computer has a model server at all. */
  readStatus = async () => {
    const generation = this.generation;
    try {
      const status = await this.client.call<ServeStatus>('serve.status.get', {}, { signal: AbortSignal.timeout(8000) });
      if (generation !== this.generation) return;
      const fresh = Date.now() - this.snapshot.sampleAt < 6000;
      this.set({
        status, absent: false, loading: false, problem: null,
        ...(fresh ? {} : { lanes: status.lanes ?? [], gpus: status.gpus ?? [], totals: status.totals ?? this.snapshot.totals }),
      });
    } catch (cause) {
      if (generation !== this.generation) return;
      if (noServer(cause)) this.set({ absent: true, loading: false, problem: null });
      else this.set({ loading: false, problem: errorMessage(cause) });
    }
  };

  /** `serve.lanes.list`: the fallback when samples stop. */
  readLanes = async () => {
    if (this.snapshot.absent) return;
    if (Date.now() - this.snapshot.sampleAt < 6000) return;
    const generation = this.generation;
    try {
      const view = await this.client.call<LanesView>('serve.lanes.list', {}, { signal: AbortSignal.timeout(8000) });
      if (generation !== this.generation) return;
      this.set({ lanes: view.lanes ?? [], problem: null });
    } catch (cause) {
      if (generation !== this.generation) return;
      if (noServer(cause)) this.set({ absent: true });
    }
  };

  refresh = async () => { await this.readStatus(); };

  private onSample = (sample: ServeSample) => {
    if (!sample?.at_ms) return;
    this.set({ lanes: sample.lanes ?? [], gpus: sample.gpus ?? [], totals: sample.totals ?? null, sampleAt: Date.now(), absent: false, loading: false, problem: null });
  };

  private onChanged = (view: LanesView) => {
    const before = new Map(this.snapshot.lanes.map((lane) => [lane.settings.id ?? '', lane]));
    const after = new Map((view?.lanes ?? []).map((lane) => [lane.settings.id ?? '', lane]));
    for (const [id, lane] of after) if (!before.has(id)) logActivity({ node: this.node, computer: this.computer, lane: lane.name || id, text: `added on GPU ${lane.settings.gpu ?? 0}`, tone: 'ok' });
    for (const [id, lane] of before) if (!after.has(id)) logActivity({ node: this.node, computer: this.computer, lane: lane.name || id, text: 'removed', tone: 'faint' });
    this.set({ lanes: view?.lanes ?? this.snapshot.lanes });
    void this.readStatus();
  };

  private onFrameLanes = (lanes: LaneState[]) => {
    const seen = new Set<string>();
    for (const lane of lanes ?? []) {
      seen.add(lane.id);
      const previous = this.frameLanes.get(lane.id);
      if (previous !== lane.state) {
        if (previous !== undefined || this.frameLanes.size) {
          const tone = laneTone(lane.state, true);
          const name = this.snapshot.lanes.find((entry) => entry.settings.id === lane.id)?.name || lane.id;
          logActivity({ node: this.node, computer: this.computer, lane: name, text: previous ? `${previous} → ${lane.state}` : lane.state, tone: tone === 'error' ? 'warn' : tone === 'serving' ? 'ok' : 'faint' });
        }
        this.frameLanes.set(lane.id, lane.state);
      }
    }
    for (const id of [...this.frameLanes.keys()]) if (!seen.has(id)) {
      this.frameLanes.delete(id);
      logActivity({ node: this.node, computer: this.computer, lane: id, text: 'stopped', tone: 'faint' });
    }
  };

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    if (this.listeners.size === 1) {
      const client = this.client;
      const offs = [
        client.on('serve.sample', (_topic, data) => this.onSample(data as ServeSample)),
        client.on('serve.changed', (_topic, data) => this.onChanged(data as LanesView)),
        client.on('frame.lanes.changed', (_topic, data) => this.onFrameLanes((data as { lanes?: LaneState[] })?.lanes ?? [])),
      ];
      // The frame's lanes as they are now, so the first change is told against them.
      void client.call<{ lanes: LaneState[] }>('frame.lanes.list', {}, { signal: AbortSignal.timeout(8000) })
        .then((result) => { for (const lane of result?.lanes ?? []) this.frameLanes.set(lane.id, lane.state); })
        .catch(() => undefined);
      void this.readStatus();
      const lanesTimer = setInterval(() => void this.readLanes(), 10_000);
      // A computer without a server may get one installed; one with a server refreshes its engines and wants.
      const statusTimer = setInterval(() => void this.readStatus(), 30_000);
      this.stop = () => { for (const off of offs) off(); clearInterval(lanesTimer); clearInterval(statusTimer); this.generation++; };
    }
    return () => {
      this.listeners.delete(listener);
      if (!this.listeners.size) { this.stop?.(); this.stop = null; }
    };
  };
}

const stores = new Map<string, ServeStore>();
function storeFor(node: string): ServeStore {
  let store = stores.get(node);
  if (!store) { store = new ServeStore(node); stores.set(node, store); }
  return store;
}

const idle = () => () => undefined;

/** A computer's server view; `enabled` false (an offline computer) reads the last one without connecting. */
export function useServe(node: string, enabled = true) {
  const store = storeFor(node);
  const snapshot = useSyncExternalStore(enabled ? store.subscribe : idle, store.get);
  return { ...snapshot, refresh: store.refresh };
}

/** The latest view of a computer, outside React (the mesh totals, the sidebar). */
export function serveOf(node: string): ServeSnapshot {
  return stores.get(node)?.get() ?? EMPTY;
}

/** A computer's lanes with a live state for one lane id. */
export function laneById(node: string, id: string): LaneView | undefined {
  return serveOf(node).lanes.find((lane) => lane.settings.id === id);
}
