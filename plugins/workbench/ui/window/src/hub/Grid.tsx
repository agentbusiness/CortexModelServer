/**
 * The server grid (BUILD-PLAN §4.3): ct's cluster grid, one card per computer (this one, then every
 * mesh member), with ct's card frame, six machine tiles, drag and keyboard reorder (FLIP) and
 * colours, persisted in the workbench state (`cms.cluster`). Under the tiles each card lists its
 * lanes grouped by GPU, live from `serve.sample` through the card's `ComputerScope`; a computer
 * without cortexmodelserver lists its mesh card's models, read-only.
 */

import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { CSSProperties, KeyboardEvent, PointerEvent, ReactNode } from 'react';
import { Caret, ComputerScope, useShared } from '@cortex/sdk';
import { GB } from './api';
import { MachineReadings, useMetrics } from './readings';
import { useServe } from './serve-data';
import { modelName } from './mesh';
import { ctxLabel, gb, gbWhole, laneTone, millions, pct, tokS, TONE_LABEL } from './format';
import type { Computer, Gpu, LaneView, ModelCard } from './types';

export interface ClusterLayout { order: string[]; colors: Record<string, string> }
const EMPTY_LAYOUT: ClusterLayout = { order: [], colors: {} };
const SWATCHES = [
  { id: 'default', name: 'Default', value: 'var(--rule-2)' },
  { id: 'orange', name: 'Orange', value: 'var(--rubric-lift)' },
  { id: 'blue', name: 'Blue', value: 'var(--counter)' },
  { id: 'green', name: 'Green', value: '#79a88a' },
  { id: 'violet', name: 'Violet', value: '#a995c9' },
  { id: 'gold', name: 'Gold', value: '#c8ab6e' },
] as const;
export const colorStyle = (color?: string) => ({ '--computer-color': (SWATCHES.find((swatch) => swatch.id === color) ?? SWATCHES[0]).value }) as CSSProperties;
/** Routing uses an empty node for this computer; persisted object members need a name. */
export const colorKey = (node: string) => node || 'local';

/** Discovery appends; it never sorts or replaces the person's arrangement. */
export function useClusterLayout(computers: Computer[]) {
  const [stored, setLayout] = useShared<ClusterLayout>('cms.cluster', EMPTY_LAYOUT);
  const layout = { order: stored.order ?? [], colors: stored.colors ?? {} };
  const discovered = computers.map((computer) => computer.node);
  const order = [...layout.order, ...discovered.filter((node) => !layout.order.includes(node))];
  const discoveryKey = JSON.stringify(discovered);
  useEffect(() => {
    setLayout((current) => {
      const known = current.order ?? [];
      const added = discovered.filter((node) => !known.includes(node));
      return added.length ? { colors: current.colors ?? {}, order: [...known, ...added] } : current;
    });
    // The key represents the discovered identities, independently of metric and name updates.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [discoveryKey, setLayout]);
  return { layout, setLayout, order };
}

interface Drag {
  node: string; rect: DOMRect; startX: number; startY: number;
  x: number; y: number; lifted: boolean; pointer: number | null;
}
function moveBefore(order: string[], from: string, target: string) {
  const next = order.filter((node) => node !== from);
  next.splice(order.indexOf(target), 0, from);
  return next;
}

export function ServerGrid({ computers, selected, highlighted, layout, order, onLayout, onReordered, onSelect }: {
  computers: Computer[]; selected: string | null; highlighted: Set<string>; layout: ClusterLayout; order: string[];
  onLayout: (change: (current: ClusterLayout) => ClusterLayout) => void;
  /** A move was saved: the person's order replaces a sort. */
  onReordered?: () => void;
  onSelect: (node: string) => void;
}) {
  const [draft, setDraft] = useState<string[] | null>(null);
  const [lift, setLift] = useState<{ node: string; style?: CSSProperties } | null>(null);
  const [announcement, setAnnouncement] = useState('');
  const screen = useRef<HTMLDivElement>(null);
  const grid = useRef<HTMLDivElement>(null);
  const slots = useRef(new Map<string, HTMLDivElement>());
  const positions = useRef(new Map<string, DOMRect>());
  const drag = useRef<Drag | null>(null);
  const draftRef = useRef<string[]>(order);
  const visibleOrder = [...(draft ?? order), ...order.filter((node) => !(draft ?? order).includes(node))];
  const ordered = visibleOrder.flatMap((node) => computers.filter((computer) => computer.node === node));
  const orderKey = JSON.stringify(ordered.map((computer) => computer.node));

  const finish = (save: boolean) => {
    const active = drag.current;
    if (!active) return;
    const card = slots.current.get(active.node)?.querySelector<HTMLElement>('.ct-cluster-card');
    const from = active.pointer !== null && active.lifted ? card?.getBoundingClientRect() : undefined;
    if (save && active.lifted) {
      const next = draftRef.current;
      onLayout((current) => ({ ...current, order: [...next, ...(current.order ?? []).filter((node) => !next.includes(node))] }));
      onReordered?.();
      setAnnouncement('Server order saved.');
    } else if (active.lifted) setAnnouncement('Move cancelled.');
    drag.current = null;
    setDraft(null);
    setLift(null);
    requestAnimationFrame(() => {
      slots.current.get(active.node)?.querySelector<HTMLButtonElement>('.ct-cluster-grip')?.focus({ preventScroll: true });
      if (from && card?.isConnected && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        const to = card.getBoundingClientRect();
        card.animate([{ transform: `translate(${from.left - to.left}px,${from.top - to.top}px)` }, { transform: 'none' }],
          { duration: 210, easing: 'cubic-bezier(.16,1,.3,1)' });
      }
    });
  };

  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape' && drag.current) { event.preventDefault(); event.stopPropagation(); finish(false); }
    };
    const onBlur = () => { if (drag.current) finish(false); };
    window.addEventListener('keydown', onKey);
    window.addEventListener('blur', onBlur);
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('blur', onBlur); };
    // The handlers read the current drag from its ref.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (drag.current && !computers.some((computer) => computer.node === drag.current?.node)) finish(false);
    // Membership can change during a move; never keep dragging a removed computer.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [orderKey]);

  useLayoutEffect(() => {
    const next = new Map<string, DOMRect>();
    for (const [node, element] of slots.current) {
      // A transformed slot would become the fixed dragged card's containing block.
      for (const animation of element.getAnimations()) animation.cancel();
      const rect = element.getBoundingClientRect();
      const before = positions.current.get(node);
      next.set(node, rect);
      if (before && !(drag.current?.node === node && drag.current.pointer !== null) && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        const x = before.left - rect.left, y = before.top - rect.top;
        if (x || y) element.animate([{ transform: `translate(${x}px,${y}px)` }, { transform: 'none' }], { duration: 210, easing: 'cubic-bezier(.16,1,.3,1)' });
      }
    }
    positions.current = next;
    // Keyed DOM moves can drop focus; keep keyboard reordering on the lifted handle.
    const active = drag.current;
    if (active?.pointer === null) {
      const slot = slots.current.get(active.node);
      slot?.querySelector<HTMLButtonElement>('.ct-cluster-grip')?.focus({ preventScroll: true });
      slot?.scrollIntoView({ block: 'nearest' });
    }
  }, [orderKey]);

  const relocate = (target: string) => {
    const active = drag.current;
    if (!active || active.node === target || !draftRef.current.includes(target)) return;
    const next = moveBefore(draftRef.current, active.node, target);
    draftRef.current = next;
    setDraft(next);
    const position = next.filter((node) => computers.some((computer) => computer.node === node)).indexOf(active.node) + 1;
    setAnnouncement(`Position ${position} of ${computers.length}.`);
  };

  const scrollParent = () => screen.current?.closest<HTMLElement>('.ct-cluster-scroll') ?? null;

  const trackPointer = (x: number, y: number) => {
    const active = drag.current;
    if (!active || active.pointer === null) return;
    active.x = x; active.y = y;
    if (!active.lifted && Math.hypot(x - active.startX, y - active.startY) < 5) return;
    active.lifted = true;
    setLift({ node: active.node, style: { position: 'fixed', left: active.rect.left, top: active.rect.top,
      width: active.rect.width, height: active.rect.height, transform: `translate3d(${x - active.startX}px,${y - active.startY}px,0)`, zIndex: 60 } });
    const bounds = grid.current?.getBoundingClientRect();
    if (!bounds || x < bounds.left || x > bounds.right) return;
    let closest: string | undefined, distance = Infinity;
    for (const [node, rect] of positions.current) {
      const d = Math.hypot(x - (rect.left + rect.width / 2), y - (rect.top + rect.height / 2));
      if (d < distance) { closest = node; distance = d; }
    }
    if (closest !== undefined) relocate(closest);
  };

  // Keep scrolling while a held card is near an edge, including when the pointer stops moving.
  useEffect(() => {
    if (!lift || drag.current?.pointer == null) return;
    let frame = 0;
    const tick = () => {
      const active = drag.current, field = scrollParent();
      if (!active || !field) return;
      const bounds = field.getBoundingClientRect();
      const delta = active.y < bounds.top + 64 ? -10 : active.y > bounds.bottom - 64 ? 10 : 0;
      if (delta) { field.scrollTop += delta; positions.current = new Map([...slots.current].map(([node, element]) => [node, element.getBoundingClientRect()])); trackPointer(active.x, active.y); }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
    // A single scrolling loop reads the current drag from its ref.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lift?.node]);

  const startPointer = (event: PointerEvent<HTMLButtonElement>, node: string) => {
    if (event.button !== 0 || drag.current) return;
    const slot = slots.current.get(node);
    if (!slot) return;
    event.preventDefault();
    event.currentTarget.focus();
    // Capture on the stationary grid; moving a keyed card must not end pointer capture.
    screen.current?.setPointerCapture(event.pointerId);
    for (const animation of slot.getAnimations()) animation.cancel();
    positions.current = new Map([...slots.current].map(([id, element]) => [id, element.getBoundingClientRect()]));
    draftRef.current = visibleOrder;
    drag.current = { node, rect: slot.getBoundingClientRect(), startX: event.clientX, startY: event.clientY,
      x: event.clientX, y: event.clientY, lifted: false, pointer: event.pointerId };
  };

  const onGripKey = (event: KeyboardEvent<HTMLButtonElement>, node: string) => {
    if (event.key === ' ' || event.key === 'Enter') {
      event.preventDefault();
      if (drag.current?.pointer === null) { finish(true); return; }
      if (drag.current) return;
      const rect = slots.current.get(node)?.getBoundingClientRect();
      if (!rect) return;
      draftRef.current = visibleOrder;
      drag.current = { node, rect, startX: 0, startY: 0, x: 0, y: 0, lifted: true, pointer: null };
      setLift({ node });
      setAnnouncement(`Picked up ${computers.find((computer) => computer.node === node)?.name}. Use arrow keys to move; Enter to save; Escape to cancel.`);
      return;
    }
    const active = drag.current;
    if (!active || active.pointer !== null || active.node !== node) return;
    const keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const ids = draftRef.current.filter((id) => computers.some((computer) => computer.node === id));
    const first = positions.current.get(ids[0] ?? '');
    const columns = [...positions.current.values()].filter((rect) => first && Math.abs(rect.top - first.top) < 2).length || 1;
    const index = ids.indexOf(node);
    const offset = event.key === 'ArrowLeft' ? -1 : event.key === 'ArrowRight' ? 1 : event.key === 'ArrowUp' ? -columns : columns;
    const target = event.key === 'Home' ? ids[0] : event.key === 'End' ? ids.at(-1) : ids[Math.max(0, Math.min(ids.length - 1, index + offset))];
    if (target !== undefined) relocate(target);
  };

  return <div ref={screen} className="cms-grid-wrap"
    onPointerMove={(event) => { if (drag.current?.pointer === event.pointerId) trackPointer(event.clientX, event.clientY); }}
    onPointerUp={(event) => { if (drag.current?.pointer === event.pointerId) finish(true); }}
    onPointerCancel={() => finish(false)} onLostPointerCapture={() => { if (drag.current?.pointer != null) finish(false); }}>
    <p id="ct-cluster-move-help" className="ct-visually-hidden">Press Space on a move handle, use arrow keys to rearrange, and Enter to save. Escape cancels.</p>
    {ordered.length === 0 && <p className="ct-hint">No server matches this filter.</p>}
    <div ref={grid} className="ct-cluster-grid">
      {ordered.map((computer) => <div key={computer.node} data-computer={computer.node || 'local'}
        ref={(element) => { if (element) slots.current.set(computer.node, element); else slots.current.delete(computer.node); }}
        style={{ ...colorStyle(layout.colors[colorKey(computer.node)]), ...(lift?.node === computer.node && lift.style ? { height: drag.current?.rect.height } : {}) }}
        className={`ct-cluster-slot${lift?.node === computer.node ? ' is-placeholder' : ''}`}>
        <ComputerScope node={computer.node || null}>
          <ServerCard computer={computer} color={layout.colors[colorKey(computer.node)]} selected={selected === computer.node}
            highlighted={highlighted.has(computer.node)}
            lifted={lift?.node === computer.node} dragStyle={lift?.node === computer.node ? lift.style : undefined}
            onColor={(color) => onLayout((current) => ({ ...current, colors: { ...(current.colors ?? {}), [colorKey(computer.node)]: color } }))}
            onSelect={() => { finish(false); onSelect(computer.node); }}
            grip={<button type="button" className="ct-cluster-grip" aria-label={`Move ${computer.name}`} aria-describedby="ct-cluster-move-help"
              aria-pressed={lift?.node === computer.node} onPointerDown={(event) => startPointer(event, computer.node)}
              onKeyDown={(event) => onGripKey(event, computer.node)}>
              <svg width="12" height="16" viewBox="0 0 12 16" fill="currentColor" aria-hidden>
                {[4, 8, 12].flatMap((y) => [3, 8].map((x) => <circle key={`${x}-${y}`} cx={x} cy={y} r="1" />))}
              </svg>
            </button>} />
        </ComputerScope>
      </div>)}
    </div>
    <span className="ct-visually-hidden" role="status" aria-live="polite">{announcement}</span>
  </div>;
}

function ServerCard({ computer, selected, highlighted, color, lifted, dragStyle, grip, onColor, onSelect }: {
  computer: Computer; selected: boolean; highlighted: boolean; color?: string; lifted: boolean; dragStyle?: CSSProperties;
  grip: ReactNode; onColor: (color: string) => void; onSelect: () => void;
}) {
  const { metrics, history, problem } = useMetrics(computer.reachable);
  const serve = useServe(computer.node, computer.reachable);
  const popover = `ct-color-${computer.node || 'local'}`;
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 5000); return () => clearInterval(timer); }, []);
  const stale = !computer.reachable || !!problem || !!metrics && now - metrics.at > 15_000;
  const lanes = serve.lanes;
  const engines = new Set(lanes.map((lane) => lane.engine).filter(Boolean)).size;
  const slots = lanes.reduce((sum, lane) => sum + (lane.slots ?? lane.settings.slots ?? 0), 0);
  const tokensHour = serve.totals?.tokens_hour ?? lanes.reduce((sum, lane) => sum + (lane.tokens_hour ?? 0), 0);
  const chip = metrics ? `${metrics.machine} · ${Math.round(metrics.memory_total / GB)} GB` : computer.card?.chip ? `${computer.card.chip} · ${Math.round((computer.card.memory_bytes ?? 0) / GB)} GB` : '';
  return <article className={`ct-cluster-card${selected ? ' is-selected' : ''}${highlighted ? ' is-highlit' : ''}${lifted ? ' is-lifted' : ''}${stale ? ' is-stale' : ''}`}
    style={{ ...colorStyle(color), ...dragStyle }} aria-label={computer.name}
    onClick={(event) => { if (!(event.target as HTMLElement).closest('button,[popover],a,input')) onSelect(); }}>
    <header className="ct-cluster-card-head">
      {grip}
      <span className="ct-computer-color" aria-hidden />
      <button type="button" className="ct-cluster-name" onClick={onSelect} title={`Edit the server on ${computer.name}`}>{computer.name}</button>
      <span className="grow" />
      <button type="button" className="ct-cluster-color-button" popoverTarget={popover} aria-label={`Colour for ${computer.name}`} title="Choose colour">
        <span className="ct-computer-color" aria-hidden /><Caret dir="down" />
      </button>
      <div id={popover} popover="auto" className="ct-cluster-swatches" role="group" aria-label={`Colour for ${computer.name}`}>
        {SWATCHES.map((swatch) => <button key={swatch.id} type="button" style={colorStyle(swatch.id)} title={swatch.name}
          aria-label={swatch.name} aria-pressed={(color ?? 'default') === swatch.id} onClick={(event) => {
            onColor(swatch.id); event.currentTarget.closest<HTMLElement>('[popover]')?.hidePopover();
          }}><span aria-hidden />{(color ?? 'default') === swatch.id && <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="var(--ground)" strokeWidth="2" aria-hidden><path d="m3 8 3 3 7-7" /></svg>}</button>)}
      </div>
    </header>
    <div className="ct-cluster-meta ct-label">
      <span className={computer.reachable && !stale ? 'ct-ok' : 'ct-faint'}>{computer.reachable ? stale ? 'RECONNECTING' : computer.node ? 'MESH ONLINE' : 'LOCAL' : 'OFFLINE'}</span>
      <span className="ct-faint">{chip || (computer.reachable ? 'READING METRICS…' : 'WAITING FOR COMPUTER')}</span>
    </div>
    {problem && <p className="ct-cluster-note ct-warn" role="status">Metrics unavailable. Retrying automatically.</p>}
    {stale && metrics && <p className="ct-cluster-note ct-faint">Last sample {new Date(metrics.at).toLocaleTimeString()} · readings below are paused.</p>}
    <div className="ct-cluster-readings"><MachineReadings metrics={metrics} history={history} /></div>
    <div className="cms-lanes">
      <div className="cms-lanes-head"><span className="ct-label">LANES</span><span className="grow" />
        {serve.problem && !serve.absent && <span className="ct-label ct-warn" title={serve.problem}>SERVER UNAVAILABLE</span>}</div>
      {serve.absent || (!computer.reachable && !serve.lanes.length)
        ? <CardModels computer={computer} absent={serve.absent} />
        : serve.loading && !lanes.length ? <p className="ct-cluster-note ct-faint">Reading the server…</p>
        : <LaneGroups gpus={serve.gpus} lanes={lanes} />}
    </div>
    <footer className="ct-cluster-card-foot">
      <span className="ct-label ct-faint cms-foot-sum">{!computer.reachable ? 'READINGS RESUME WHEN ONLINE' : serve.absent ? 'NO MODEL SERVER' : `${lanes.length} LANE${lanes.length === 1 ? '' : 'S'} · ${slots} SLOT${slots === 1 ? '' : 'S'} · ${engines} ENGINE${engines === 1 ? '' : 'S'} · ${millions(tokensHour)} M/H`}</span>
      <button type="button" className="ct-link" onClick={onSelect}>{serve.absent ? 'VIEW' : 'EDIT'} <span aria-hidden>→</span></button>
    </footer>
  </article>;
}

/** `GPU 0 · RTX 5090 · 32 GB`, a Mac's `UNIFIED MEMORY · 192 GB`. */
export function gpuTitle(gpu: Gpu, count: number): string {
  if (gpu.unified) return `UNIFIED MEMORY · ${gbWhole(gpu.memory_bytes)}`;
  return `${count > 1 || gpu.index > 0 ? `GPU ${gpu.index}` : 'GPU'} · ${gpu.name.replace(/^NVIDIA (GeForce )?/i, '')} · ${gbWhole(gpu.memory_bytes)}`;
}

/** The lanes grouped by their GPU: a header with a hairline memory bar, then one row per lane. */
function LaneGroups({ gpus, lanes }: { gpus: Gpu[]; lanes: LaneView[] }) {
  const indices = [...new Set([...gpus.map((gpu) => gpu.index), ...lanes.map((lane) => lane.settings.gpu ?? 0)])].sort((a, b) => a - b);
  if (!indices.length) return <p className="ct-cluster-note ct-faint">No GPU reported and no lanes yet.</p>;
  return <>{indices.map((index) => {
    const gpu = gpus.find((entry) => entry.index === index) ?? { index, name: '', backend: '', memory_bytes: 0 };
    const own = lanes.filter((lane) => (lane.settings.gpu ?? 0) === index);
    const used = gpu.used_bytes || own.reduce((sum, lane) => sum + (lane.memory_bytes ?? 0), 0);
    return <div key={index} className="cms-gpu">
      <div className="cms-gpu-head">
        <span className="ct-label">{gpu.name || gpu.memory_bytes ? gpuTitle(gpu, gpus.length) : `GPU ${index}`}</span>
        <span className="cms-hairbar" aria-hidden><span style={{ width: `${gpu.memory_bytes ? Math.min(100, (used / gpu.memory_bytes) * 100) : 0}%` }} /></span>
        <span className="ct-label ct-faint cms-num">{gpu.memory_bytes ? `${gb(used)} / ${gbWhole(gpu.memory_bytes)}` : '–'}</span>
        <span className="ct-label cms-num cms-util">{gpu.util != null ? pct(gpu.util) : '–'}</span>
      </div>
      {own.length === 0 && <div className="cms-lane cms-lane-empty"><span /><span className="ct-faint">no lanes</span></div>}
      {own.map((lane) => <LaneRow key={lane.settings.id || lane.name} lane={lane} />)}
    </div>;
  })}</>;
}

function LaneRow({ lane }: { lane: LaneView }) {
  const tone = laneTone(lane.state, lane.loaded, lane.error);
  const slots = lane.slots ?? lane.settings.slots ?? 0;
  const busy = lane.slots_busy ?? 0;
  return <div className={`cms-lane is-${tone}`} title={`${TONE_LABEL[tone]} · ${lane.state}${lane.error ? ` · ${lane.error}` : ''}`}>
    <span className={`cms-pip ${tone}`} aria-label={TONE_LABEL[tone]} />
    <span className="cms-lane-name">{lane.name || lane.settings.recipe}</span>
    <span className="cms-lane-engine ct-faint">{lane.engine}</span>
    <span className="cms-num">{tone === 'serving' || busy ? `${busy}/${slots} slots` : tone === 'loading' ? 'loading' : tone === 'error' ? 'error' : 'idle'}</span>
    <span className="cms-num ct-faint">{ctxLabel(lane.ctx || lane.settings.ctx)}</span>
    <span className="cms-num">{busy ? `${tokS(lane.tok_s)} tok/s` : '—'}</span>
    <span className="cms-num ct-faint">{lane.tokens_hour ? `${millions(lane.tokens_hour)} M/h` : '—'}</span>
  </div>;
}

/** A computer without cortexmodelserver: the models its mesh card lends, read-only. */
export function CardModels({ computer, absent }: { computer: Computer; absent: boolean }) {
  const models = computer.card?.models ?? [];
  return <>
    <p className="ct-cluster-note ct-faint">{absent ? `No model server here${computer.card?.software ? ` (${computer.card.software})` : ''}: its models, read-only.` : 'Offline: the models it last reported.'}</p>
    {models.length === 0 && <div className="cms-lane cms-lane-empty"><span /><span className="ct-faint">no models</span></div>}
    {models.map((model) => <CardModelRow key={model.plugin} model={model} />)}
  </>;
}

function CardModelRow({ model }: { model: ModelCard }) {
  const tone = model.loaded === false ? 'unloaded' : model.slots_busy ? 'serving' : 'unloaded';
  return <div className={`cms-lane is-${tone}`}>
    <span className={`cms-pip ${model.loaded ? 'serving' : 'unloaded'}`} aria-hidden />
    <span className="cms-lane-name">{modelName(model)}</span>
    <span className="cms-lane-engine ct-faint">{model.engine || model.plugin}</span>
    <span className="cms-num">{model.slots_busy || model.loaded ? `${model.slots_busy}/${model.slots} slots` : 'idle'}</span>
    <span className="cms-num ct-faint">{ctxLabel(model.ctx_total)}</span>
    <span className="cms-num">{model.slots_busy && model.tok_s ? `${tokS(model.tok_s)} tok/s` : '—'}</span>
    <span className="cms-num ct-faint">{model.tokens_hour ? `${millions(model.tokens_hour)} M/h` : '—'}</span>
  </div>;
}

