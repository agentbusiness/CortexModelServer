/**
 * The server editor in the right pane (BUILD-PLAN §4.4): one computer's lanes, edited in place. Every
 * call goes to that computer (`runtimeFor(node)`, a mesh member through `mesh.link`).
 *
 * Edits are staged here and sent on APPLY: removals first (they free memory), then changes, then
 * new lanes (`serve.lanes.delete` / `edit` / `create`, one at a time); each lane's live state shows
 * as it drains and reloads. The memory bars recompute from `serve.fit.get` for every staged lane as
 * you edit. RESET TO PLAN shows `serve.plan.get` as a diff to accept (`serve.plan.run`: lanes set
 * by hand become the plan's again); CALIBRATE runs `serve.calibrate.run`.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Menu, errorMessage, runtimeFor, type MenuItem } from '@cortex/sdk';
import { CardModels, gpuTitle } from './Grid';
import { CONTEXTS, ctxLabel, gb, gbWhole, laneTone, TONE_LABEL, tokS } from './format';
import { serveOf, useServe } from './serve-data';
import type { Calibration, CalibrationResult, Computer, Fit, Gpu, LaneSettings, LaneView, LanesView, Plan, Recipe } from './types';

interface Staged {
  key: string;
  /** The lane it edits; none: a new lane. */
  id?: string;
  settings: LaneSettings;
  removed?: boolean;
  more?: boolean;
  /** The person chose CUSTOM for the cache (kept even when the number matches a preset). */
  customCache?: boolean;
}

type Step = 'sending' | 'done' | string;

const CACHE_PRESETS = [
  { id: 'off', label: 'OFF', mib: 0 },
  { id: 'small', label: 'SMALL', mib: 1024 },
  { id: 'large', label: 'LARGE', mib: 6144 },
] as const;
const KV_TYPES = ['f16', 'q8_0', 'q4_0', 'int8'];
const IDLE = [{ s: 0, label: 'NEVER' }, { s: 300, label: '5 MIN' }, { s: 900, label: '15 MIN' }, { s: 3600, label: '1 H' }];

const toStaged = (lane: LaneView): Staged => ({ key: lane.settings.id || lane.name, id: lane.settings.id, settings: { ...lane.settings } });
const comparable = (settings: LaneSettings) => JSON.stringify({ ...settings, tags: settings.tags ?? [], manual: undefined, lend: settings.lend !== false, pinned: Boolean(settings.pinned) });
let newKey = 0;

export function Editor({ computer }: { computer: Computer }) {
  const node = computer.node;
  const client = runtimeFor(node || null).client;
  const serve = useServe(node, computer.reachable);
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [recipesProblem, setRecipesProblem] = useState<string | null>(null);
  const [staged, setStaged] = useState<Staged[] | null>(null);
  const [fits, setFits] = useState<Record<string, Fit | string>>({});
  const fitCache = useRef(new Map<string, Fit | string>());
  const [steps, setSteps] = useState<Record<string, Step>>({});
  const [applying, setApplying] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [planBusy, setPlanBusy] = useState(false);
  const [calibration, setCalibration] = useState<Calibration[] | null>(null);
  const [calibrating, setCalibrating] = useState(false);
  const [adding, setAdding] = useState<{ recipe: string; fits: Record<number, Fit | string> } | null>(null);

  useEffect(() => {
    if (!computer.reachable || serve.absent) return;
    let stopped = false;
    client.call<{ recipes: Recipe[] }>('serve.recipes.list', {}, { signal: AbortSignal.timeout(10_000) })
      .then((result) => { if (!stopped) { setRecipes(result?.recipes ?? []); setRecipesProblem(null); } })
      .catch((cause) => { if (!stopped) setRecipesProblem(errorMessage(cause)); });
    return () => { stopped = true; };
  }, [client, computer.reachable, serve.absent]);

  const base = useMemo(() => serve.lanes.map(toStaged), [serve.lanes]);
  const rows = staged ?? base;
  const baseById = new Map(base.map((row) => [row.id, row]));
  const changedRow = (row: Staged) => row.removed || !row.id || comparable(row.settings) !== comparable(baseById.get(row.id)?.settings ?? { recipe: '' });
  const dirty = staged !== null && (rows.some(changedRow) || base.some((row) => !rows.some((other) => other.key === row.key)));
  const recipeOf = (id: string) => recipes.find((recipe) => recipe.id === id);
  const gpus: Gpu[] = serve.gpus.length ? serve.gpus : serve.status?.gpus ?? [];
  const liveOf = (id?: string) => (id ? serve.lanes.find((lane) => lane.settings.id === id) : undefined);

  const edit = (key: string, change: (row: Staged) => Staged) => setStaged((current) => (current ?? base).map((row) => (row.key === key ? change(row) : row)));
  const set = (key: string, settings: Partial<LaneSettings>) => edit(key, (row) => ({ ...row, settings: { ...row.settings, ...settings } }));

  // --- fit: every staged lane's memory, debounced, cached by its settings -------------------------
  const fitKey = (row: Staged) => JSON.stringify([row.settings, row.id ?? '']);
  const wanted = rows.filter((row) => !row.removed);
  const wantedKey = JSON.stringify(wanted.map(fitKey));
  useEffect(() => {
    if (!computer.reachable || serve.absent || !wanted.length) return;
    let stopped = false;
    const timer = setTimeout(() => {
      void Promise.all(wanted.map(async (row) => {
        const key = fitKey(row);
        let fit = fitCache.current.get(key);
        if (!fit) {
          fit = await client.call<Fit>('serve.fit.get', { settings: sendable(row.settings), replaces: row.id ?? '' }, { signal: AbortSignal.timeout(8000) })
            .catch((cause) => errorMessage(cause));
          if (typeof fit !== 'string') fitCache.current.set(key, fit);
        }
        return [row.key, fit] as const;
      })).then((entries) => { if (!stopped) setFits(Object.fromEntries(entries)); });
    }, 250);
    return () => { stopped = true; clearTimeout(timer); };
    // The key stands for the staged settings.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wantedKey, client, computer.reachable, serve.absent]);
  const fitOf = (row: Staged): Fit | undefined => { const fit = fits[row.key]; return typeof fit === 'object' ? fit : undefined; };

  // --- apply: removals, changes, then new lanes, one at a time ------------------------------------
  const apply = async () => {
    setApplying(true);
    setProblem(null);
    const failed = new Set<string>();
    const step = (key: string, value: Step) => setSteps((current) => ({ ...current, [key]: value }));
    const run = async (row: Staged, call: () => Promise<unknown>) => {
      step(row.key, 'sending');
      try { await call(); step(row.key, 'done'); } catch (cause) { failed.add(row.key); step(row.key, errorMessage(cause)); }
    };
    const removedBase = base.filter((row) => !rows.some((other) => other.key === row.key));
    for (const row of [...rows.filter((entry) => entry.removed && entry.id), ...removedBase]) {
      await run(row, () => client.call('serve.lanes.delete', { id: row.id }, { signal: AbortSignal.timeout(60_000) }));
    }
    for (const row of rows.filter((entry) => !entry.removed && entry.id && changedRow(entry))) {
      await run(row, () => client.call<LaneView>('serve.lanes.edit', { id: row.id, settings: { ...sendable(row.settings), id: row.id, manual: true } }, { signal: AbortSignal.timeout(120_000) }));
    }
    for (const row of rows.filter((entry) => !entry.removed && !entry.id)) {
      await run(row, () => client.call<LaneView>('serve.lanes.create', { ...sendable(row.settings), id: '', manual: true }, { signal: AbortSignal.timeout(120_000) }));
    }
    await serve.refresh();
    if (failed.size) {
      // Keep only what did not take; the rest is the server's now.
      const fresh = serveOf(node).lanes.map(toStaged);
      const kept = rows.filter((row) => failed.has(row.key));
      setStaged([...fresh.map((row) => kept.find((entry) => entry.id === row.id) ?? row), ...kept.filter((row) => !row.id)]);
      setProblem(`${failed.size} change${failed.size === 1 ? '' : 's'} did not take: see the lane${failed.size === 1 ? '' : 's'} below.`);
    } else setStaged(null);
    setApplying(false);
    setTimeout(() => setSteps((current) => Object.fromEntries(Object.entries(current).filter(([, value]) => value !== 'done'))), 8000);
  };

  const askPlan = async () => {
    setPlanBusy(true);
    setProblem(null);
    try { setPlan(await client.call<Plan>('serve.plan.get', {}, { signal: AbortSignal.timeout(30_000) })); } catch (cause) { setProblem(`No plan: ${errorMessage(cause)}`); }
    setPlanBusy(false);
  };
  const acceptPlan = async () => {
    setPlanBusy(true);
    try {
      await client.call<LanesView>('serve.plan.run', {}, { signal: AbortSignal.timeout(300_000) });
      setPlan(null);
      setStaged(null);
      await serve.refresh();
    } catch (cause) { setProblem(`The plan did not run: ${errorMessage(cause)}`); }
    setPlanBusy(false);
  };
  const calibrate = async () => {
    setCalibrating(true);
    setProblem(null);
    try {
      const result = await client.call<CalibrationResult>('serve.calibrate.run', { lane: '' }, { signal: AbortSignal.timeout(900_000) });
      setCalibration(result?.results ?? []);
    } catch (cause) { setProblem(`Calibration failed: ${errorMessage(cause)}`); }
    setCalibrating(false);
  };

  const startAdd = (recipe: string) => {
    setAdding({ recipe, fits: {} });
    const targets = gpus.length ? gpus : [{ index: 0 } as Gpu];
    void Promise.all(targets.map(async (gpu) => [gpu.index, await client.call<Fit>('serve.fit.get', { settings: { recipe, gpu: gpu.index, slots: 0, ctx: 0 } }, { signal: AbortSignal.timeout(8000) })
      .catch((cause) => errorMessage(cause))] as const))
      .then((entries) => setAdding((current) => (current?.recipe === recipe ? { recipe, fits: Object.fromEntries(entries) } : current)));
  };
  const addOn = (recipe: string, gpu: number) => {
    const key = `new-${++newKey}`;
    setStaged((current) => [...(current ?? base), { key, settings: { recipe, gpu, slots: 0, ctx: 0, lend: true, pinned: false, tags: [] } }]);
    setAdding(null);
  };

  // --- what is drawn ------------------------------------------------------------------------------
  const head = (
    <header className="cms-ed-head">
      <div className="cms-ed-title">
        <span className="ct-serif-title">{computer.name}</span>
        <span className="grow" />
      </div>
      <div className="ct-label ct-faint">{[computer.card?.chip, serve.status?.platform, computer.node ? (computer.reachable ? 'MESH' : 'OFFLINE') : 'LOCAL'].filter(Boolean).join(' · ')}</div>
    </header>
  );

  if (!computer.reachable) {
    return <div className="cms-editor">{head}<div className="ct-machine-offline" role="status">
      <span className="ct-reading-name">{computer.name} is offline</span>
      <p className="ct-hint">Its server can be edited when it reconnects.</p>
    </div></div>;
  }
  if (serve.absent) {
    return <div className="cms-editor">{head}
      <p className="ct-hint">This computer has no model server (cortexmodelserver). The models it lends to the mesh are listed read-only.</p>
      <div className="cms-lanes"><CardModels computer={computer} absent /></div>
    </div>;
  }

  const indices = [...new Set([...gpus.map((gpu) => gpu.index), ...rows.map((row) => row.settings.gpu ?? 0)])].sort((a, b) => a - b);
  const addRecipe = adding ? recipeOf(adding.recipe) : undefined;
  const addFits = adding ? Object.entries(adding.fits).map(([gpu, fit]) => ({ gpu: Number(gpu), fit })) : [];
  const fitting = addFits.filter((entry) => typeof entry.fit === 'object' && entry.fit.fits);
  const recipeItems = (onPick: (id: string) => void, current?: string): MenuItem[] => recipes.map((recipe) => ({
    key: recipe.id, label: recipeLabel(recipe), selected: recipe.id === current, disabled: recipe.available === false,
    note: recipe.available === false ? recipe.reason || 'not here' : recipe.downloaded === false ? 'downloads' : undefined,
    onSelect: () => onPick(recipe.id),
  }));

  return (
    <div className="cms-editor">
      {head}
      <div className="cms-ed-bars">
        {gpus.map((gpu) => <MemoryBar key={gpu.index} gpu={gpu} count={gpus.length}
          rows={rows.filter((row) => !row.removed && (row.settings.gpu ?? 0) === gpu.index).map((row) => ({ fit: fitOf(row), memory: liveOf(row.id)?.memory_bytes ?? 0 }))} />)}
        {!gpus.length && <p className="ct-hint">{serve.loading ? 'Reading the GPUs…' : 'No GPU reported.'}</p>}
      </div>

      <div className="cms-ed-actions">
        <button type="button" className="detent solid" disabled={!dirty || applying} onClick={() => void apply()}>{applying ? 'APPLYING…' : 'APPLY'}</button>
        <button type="button" className="detent" disabled={!dirty || applying} onClick={() => { setStaged(null); setSteps({}); setProblem(null); }}>DISCARD</button>
        <span className="grow" />
        <button type="button" className="detent blue" disabled={planBusy || applying} onClick={() => void askPlan()}>{planBusy && !plan ? 'PLANNING…' : 'RESET TO PLAN'}</button>
        <button type="button" className="detent" disabled={calibrating || applying} onClick={() => void calibrate()}>{calibrating ? 'CALIBRATING…' : 'CALIBRATE'}</button>
      </div>
      {dirty && !applying && <p className="ct-label ct-warn cms-ed-staged">{rows.filter(changedRow).length + base.filter((row) => !rows.some((other) => other.key === row.key)).length} CHANGE(S) STAGED · CHANGED LANES DRAIN AND RELOAD ONE AT A TIME</p>}
      {problem && <p className="inline-error" role="alert">{problem}</p>}
      {serve.problem && <p className="inline-error" role="status">The server does not answer: {serve.problem}</p>}
      {recipesProblem && <p className="inline-error">Recipes unavailable: {recipesProblem}</p>}

      {plan && <section className="cms-ed-panel" aria-label="Automatic plan">
        <div className="cms-ed-panel-head"><span className="ct-label">THE PLAN</span><span className="grow" />
          <button type="button" className="detent sm solid" disabled={planBusy} onClick={() => void acceptPlan()}>{planBusy ? 'RUNNING…' : 'ACCEPT'}</button>
          <button type="button" className="detent sm" disabled={planBusy} onClick={() => setPlan(null)}>REJECT</button></div>
        {plan.changes.length === 0 && <p className="ct-hint">The plan matches the lanes now. Accepting keeps following it (AUTO).</p>}
        <ul className="cms-diff">
          {plan.changes.map((change, index) => {
            const recipe = recipeOf(change.lane.recipe);
            return <li key={index} className={`cms-diff-${change.op}`}>
              <span className="ct-label cms-diff-op">{change.op.toUpperCase()}</span>
              <span className="cms-diff-what">{recipe?.name ?? change.lane.recipe}<span className="ct-faint"> · GPU {change.lane.gpu ?? 0} · {change.lane.slots || recipe?.slots_default || '–'} slots · {ctxLabel(change.lane.ctx || recipe?.ctx_default)}</span></span>
              <span className="cms-diff-why ct-faint">{change.why}</span>
            </li>;
          })}
        </ul>
        {plan.notes.map((note, index) => <p key={index} className="ct-hint">{note}</p>)}
      </section>}

      {calibration && <section className="cms-ed-panel" aria-label="Calibration">
        <div className="cms-ed-panel-head"><span className="ct-label">CALIBRATION</span><span className="grow" /><button type="button" className="ct-x" aria-label="Close calibration" onClick={() => setCalibration(null)}>×</button></div>
        {calibration.length === 0 && <p className="ct-hint">No chat lane to measure.</p>}
        {calibration.length > 0 && <div className="cms-cal">
          <span className="ct-label ct-faint">LANE</span><span className="ct-label ct-faint">DECODE</span><span className="ct-label ct-faint">PREFILL</span><span className="ct-label ct-faint">ALL SLOTS</span><span className="ct-label ct-faint">TTFT</span>
          {calibration.map((result) => <CalRow key={result.lane} result={result} name={liveOf(result.lane)?.name ?? recipeOf(result.recipe)?.name ?? result.lane} />)}
        </div>}
      </section>}

      <section className="cms-ed-lanes" aria-label="Lanes">
        {indices.map((index) => {
          const gpu = gpus.find((entry) => entry.index === index);
          const own = rows.filter((row) => (row.settings.gpu ?? 0) === index);
          return <div key={index} className="cms-ed-gpu">
            <div className="cms-gpu-head"><span className="ct-label">{gpu ? gpuTitle(gpu, gpus.length) : `GPU ${index}`}</span><span className="grow" /><span className="ct-label ct-faint">{own.filter((row) => !row.removed).length} LANE{own.length === 1 ? '' : 'S'}</span></div>
            {own.length === 0 && <p className="ct-hint">No lanes on this GPU.</p>}
            {own.map((row) => <LaneEditor key={row.key} row={row} recipe={recipeOf(row.settings.recipe)} live={liveOf(row.id)} fit={fits[row.key]} step={steps[row.key]}
              changed={changedRow(row)} recipeItems={(pick) => recipeItems(pick, row.settings.recipe)} disabled={applying}
              onSet={(settings) => set(row.key, settings)} onRow={(change) => edit(row.key, change)}
              onRemove={() => (row.id ? edit(row.key, (entry) => ({ ...entry, removed: true })) : setStaged((current) => (current ?? base).filter((entry) => entry.key !== row.key)))} />)}
          </div>;
        })}
        <div className="cms-ed-add">
          <Menu variant="detent" label="ADD LANE" disabled={applying || !recipes.length} title={recipes.length ? 'Pick a recipe, then a GPU it fits on' : 'No recipes'}
            items={recipeItems(startAdd, adding?.recipe)} />
          {adding && <div className="cms-ed-addfit">
            <div className="cms-ed-panel-head"><span className="ct-label">{addRecipe ? recipeLabel(addRecipe) : adding.recipe}</span><span className="grow" /><button type="button" className="ct-x" aria-label="Cancel adding" onClick={() => setAdding(null)}>×</button></div>
            {addFits.length === 0 && <p className="ct-hint">Checking where it fits…</p>}
            {fitting.map(({ gpu, fit }) => {
              const card = gpus.find((entry) => entry.index === gpu);
              return <button key={gpu} type="button" className="detent cms-ed-fitpick" onClick={() => addOn(adding.recipe, gpu)}>
                {card?.unified ? 'UNIFIED MEMORY' : `GPU ${gpu}`} <span className="k">{typeof fit === 'object' ? fit.note : ''}</span>
              </button>;
            })}
            {addFits.length > 0 && fitting.length === 0 && <>
              <p className="inline-error">It fits on no GPU here.</p>
              {addFits.map(({ gpu, fit }) => <p key={gpu} className="ct-hint">GPU {gpu}: {typeof fit === 'object' ? fit.note : fit}</p>)}
            </>}
            {addRecipe?.downloaded === false && fitting.length > 0 && <p className="ct-hint">Its weights are not here yet: they download when the lane starts.</p>}
          </div>}
        </div>
      </section>
    </div>
  );
}

function recipeLabel(recipe: Recipe): string {
  return `${recipe.name} · ${recipe.engine}${recipe.origin ? ` (${recipe.origin})` : ''}`;
}

/** What goes on the wire: no empty id on create, the defaults left out. */
function sendable(settings: LaneSettings): LaneSettings {
  const out: LaneSettings = { ...settings };
  for (const key of Object.keys(out) as (keyof LaneSettings)[]) if (out[key] === undefined) delete out[key];
  return out;
}

/** One GPU: weights, KV, buffers and free, stacked, from the staged lanes' fits. */
function MemoryBar({ gpu, count, rows }: { gpu: Gpu; count: number; rows: { fit?: Fit; memory: number }[] }) {
  const total = gpu.memory_bytes || 1;
  const weights = rows.reduce((sum, row) => sum + (row.fit ? row.fit.weights_bytes : row.memory), 0);
  const kv = rows.reduce((sum, row) => sum + (row.fit?.kv_bytes ?? 0), 0);
  const buffers = rows.reduce((sum, row) => sum + (row.fit?.compute_bytes ?? 0), 0);
  const used = weights + kv + buffers;
  const free = gpu.memory_bytes - used;
  const width = (bytes: number) => `${Math.max(0, Math.min(100, (bytes / total) * 100))}%`;
  return <div className="cms-ed-bar">
    <div className="cms-gpu-head"><span className="ct-label">{gpuTitle(gpu, count)}</span><span className="grow" />
      <span className={`ct-label cms-num${free < 0 ? ' ct-warn' : ' ct-faint'}`}>{free < 0 ? `OVER BY ${gb(-free)} GB` : `${gb(used)} / ${gbWhole(gpu.memory_bytes)}`}</span></div>
    <div className={`cms-stack${free < 0 ? ' over' : ''}`} role="img" aria-label={`weights ${gb(weights)} GB, KV ${gb(kv)} GB, buffers ${gb(buffers)} GB, free ${gb(Math.max(0, free))} GB`}>
      <span className="w" style={{ width: width(weights) }} /><span className="kv" style={{ width: width(kv) }} /><span className="buf" style={{ width: width(buffers) }} />
    </div>
    <div className="ct-tile-note ct-memkey cms-stack-key">
      <span className="app">weights {gb(weights)}</span><span className="comp">kv {gb(kv)}</span><span className="swap">buffers {gb(buffers)}</span><span className="ct-faint">free {gb(Math.max(0, free))} GB</span>
    </div>
  </div>;
}

function CalRow({ result, name }: { result: Calibration; name: string }) {
  if (result.error) return <><span className="cms-cal-name">{name}</span><span className="cms-cal-error inline-error">{result.error}</span></>;
  return <>
    <span className="cms-cal-name">{name}</span>
    <span className="cms-num">{tokS(result.decode_tok_s)}</span>
    <span className="cms-num">{tokS(result.prefill_tok_s)}</span>
    <span className="cms-num">{tokS(result.aggregate_tok_s)}</span>
    <span className="cms-num">{Math.round(result.ttft_ms)} ms</span>
  </>;
}

function LaneEditor({ row, recipe, live, fit, step, changed, recipeItems, disabled, onSet, onRow, onRemove }: {
  row: Staged; recipe?: Recipe; live?: LaneView; fit?: Fit | string; step?: Step; changed: boolean;
  recipeItems: (pick: (id: string) => void) => MenuItem[]; disabled: boolean;
  onSet: (settings: Partial<LaneSettings>) => void; onRow: (change: (row: Staged) => Staged) => void; onRemove: () => void;
}) {
  const s = row.settings;
  const fitted = typeof fit === 'object' ? fit : undefined;
  const tone = live ? laneTone(live.state, live.loaded, live.error) : null;
  const slots = s.slots || live?.slots || recipe?.slots_default || 1;
  const maxSlots = Math.max(1, fitted?.max_slots || recipe?.slots_max || 64);
  const ctx = s.ctx || live?.ctx || recipe?.ctx_default || 0;
  const ctxLimit = Math.min(fitted?.max_ctx || Infinity, recipe?.ctx_max || Infinity);
  const contexts = [...new Set([...CONTEXTS.filter((value) => value <= ctxLimit), ...(s.ctx ? [s.ctx] : [])])].sort((a, b) => a - b);
  const cache = s.cache_ram_mib === undefined && !row.customCache ? 'default' : row.customCache ? 'custom' : CACHE_PRESETS.find((preset) => preset.mib === s.cache_ram_mib)?.id ?? 'custom';
  const stepText = step === 'sending' ? 'SENDING…' : step === 'done' ? 'SENT' : step ? 'FAILED' : '';

  if (row.removed) {
    return <div className="cms-ed-lane is-removed">
      <div className="cms-ed-line"><span className="cms-pip unloaded" aria-hidden /><span className="cms-ed-gone">{live?.name ?? recipe?.name ?? s.recipe}</span><span className="grow" />
        {stepText && <span className={`ct-label ${step === 'done' ? 'ct-blue' : step === 'sending' ? 'ct-faint' : 'ct-warn'}`}>{stepText}</span>}
        <span className="ct-label ct-warn">REMOVED ON APPLY</span>
        <button type="button" className="ct-link" disabled={disabled} onClick={() => onRow((entry) => ({ ...entry, removed: false }))}>UNDO</button></div>
      {step && step !== 'sending' && step !== 'done' && <p className="inline-error">{step}</p>}
    </div>;
  }

  return <div className={`cms-ed-lane${changed ? ' is-changed' : ''}`}>
    <div className="cms-ed-line">
      <span className={`cms-pip ${tone ?? 'unloaded'}`} title={live ? `${TONE_LABEL[tone!]} · ${live.state}` : 'new'} aria-hidden />
      <Menu className="cms-ed-recipe" label={<span className="cms-ed-recipe-l">{recipe ? recipeLabel(recipe) : s.recipe || 'PICK A RECIPE'}</span>} disabled={disabled} title="Recipe: model, engine and the app's settings"
        items={recipeItems((id) => onSet({ recipe: id, slots: 0, ctx: 0 }))} />
      <button type="button" className="detent sm warn" disabled={disabled} onClick={onRemove} aria-label="Remove this lane" title="Remove this lane">REMOVE</button>
    </div>
    <div className="cms-ed-line cms-ed-controls">
      <span className="cms-step" aria-label="Slots">
        <span className="ct-label ct-faint">SLOTS</span>
        <button type="button" className="detent sm" disabled={disabled || slots <= 1} onClick={() => onSet({ slots: slots - 1 })} aria-label="Fewer slots">−</button>
        <span className="num cms-step-v">{slots}</span>
        <button type="button" className="detent sm" disabled={disabled || slots >= maxSlots} onClick={() => onSet({ slots: slots + 1 })} aria-label="More slots" title={`At most ${maxSlots} fit`}>+</button>
      </span>
      <Menu label="CTX" value={s.ctx ? ctxLabel(s.ctx) : `${ctxLabel(ctx)}`} disabled={disabled} title={Number.isFinite(ctxLimit) ? `Up to ${ctxLabel(ctxLimit)} fits` : 'Context per lane'}
        items={[{ key: '0', label: `DEFAULT${recipe?.ctx_default ? ` (${ctxLabel(recipe.ctx_default)})` : ''}`, selected: !s.ctx, onSelect: () => onSet({ ctx: 0 }) },
          ...contexts.map((value) => ({ key: String(value), label: ctxLabel(value), selected: s.ctx === value, onSelect: () => onSet({ ctx: value }) }))]} />
      <Menu label="CACHE" value={cache === 'default' ? 'DEFAULT' : cache === 'custom' ? `${s.cache_ram_mib ?? 0} MiB` : cache.toUpperCase()} disabled={disabled} title="Prompt cache in RAM and context checkpoints"
        items={[
          { key: 'default', label: 'DEFAULT', note: "the recipe's", selected: cache === 'default', onSelect: () => onRow((entry) => ({ ...entry, customCache: false, settings: { ...entry.settings, cache_ram_mib: undefined, checkpoints: undefined } })) },
          { key: 'off', label: 'OFF', note: 'no cache, no checkpoints', selected: cache === 'off', onSelect: () => onRow((entry) => ({ ...entry, customCache: false, settings: { ...entry.settings, cache_ram_mib: 0, checkpoints: 0 } })) },
          { key: 'small', label: 'SMALL', note: '1 GB', selected: cache === 'small', onSelect: () => onRow((entry) => ({ ...entry, customCache: false, settings: { ...entry.settings, cache_ram_mib: 1024, checkpoints: entry.settings.checkpoints || undefined } })) },
          { key: 'large', label: 'LARGE', note: '6 GB', selected: cache === 'large', onSelect: () => onRow((entry) => ({ ...entry, customCache: false, settings: { ...entry.settings, cache_ram_mib: 6144, checkpoints: entry.settings.checkpoints || undefined } })) },
          { key: 'custom', label: 'CUSTOM', note: 'MiB and checkpoints', selected: cache === 'custom', onSelect: () => onRow((entry) => ({ ...entry, customCache: true, settings: { ...entry.settings, cache_ram_mib: entry.settings.cache_ram_mib ?? 2048 } })) },
        ]} />
      <span className="cms-sw-l"><span className="ct-label ct-faint">LEND</span>
        <button type="button" className="sw" aria-pressed={s.lend !== false} aria-label="Lend to the mesh" disabled={disabled} onClick={() => onSet({ lend: s.lend === false })} /></span>
      <button type="button" className="ct-more" aria-expanded={Boolean(row.more)} onClick={() => onRow((entry) => ({ ...entry, more: !entry.more }))}>{row.more ? 'LESS' : 'MORE'}</button>
    </div>
    {cache === 'custom' && <div className="cms-ed-line cms-ed-controls">
      <NumberField label="CACHE MiB" value={s.cache_ram_mib} disabled={disabled} onChange={(value) => onSet({ cache_ram_mib: value ?? 0 })} />
      <NumberField label="CHECKPOINTS" value={s.checkpoints} disabled={disabled} onChange={(value) => onSet({ checkpoints: value })} />
    </div>}
    {row.more && <div className="cms-ed-more">
      <Menu label="KV" value={s.kv_type ?? 'DEFAULT'} disabled={disabled}
        items={[{ key: '', label: 'DEFAULT', selected: !s.kv_type, onSelect: () => onSet({ kv_type: undefined }) }, ...KV_TYPES.map((kv) => ({ key: kv, label: kv, selected: s.kv_type === kv, onSelect: () => onSet({ kv_type: kv }) }))]} />
      <span className="cms-sw-l"><span className="ct-label ct-faint">SPECULATIVE</span>
        <span className="seg-set" role="group" aria-label="Speculative decoding">
          {([['DEFAULT', undefined], ['ON', true], ['OFF', false]] as const).map(([label, value]) => (
            <button key={label} type="button" disabled={disabled || (value === true && recipe?.speculative === '')} aria-pressed={s.speculative === value} onClick={() => onSet({ speculative: value })}>{label}</button>
          ))}
        </span></span>
      <Menu label="IDLE UNLOAD" value={s.idle_unload_s === undefined ? 'DEFAULT' : IDLE.find((entry) => entry.s === s.idle_unload_s)?.label ?? `${s.idle_unload_s} S`} disabled={disabled}
        items={[{ key: 'd', label: 'DEFAULT', selected: s.idle_unload_s === undefined, onSelect: () => onSet({ idle_unload_s: undefined }) }, ...IDLE.map((entry) => ({ key: String(entry.s), label: entry.label, selected: s.idle_unload_s === entry.s, onSelect: () => onSet({ idle_unload_s: entry.s }) }))]} />
      <span className="cms-sw-l"><span className="ct-label ct-faint">PIN</span>
        <button type="button" className="sw" aria-pressed={Boolean(s.pinned)} aria-label="Keep loaded" title="Keep it loaded: never unloaded when idle" disabled={disabled} onClick={() => onSet({ pinned: !s.pinned })} /></span>
      <label className="field cms-ed-tags"><span className="ct-label ct-faint">TAGS</span>
        <input type="text" defaultValue={(s.tags ?? []).join(', ')} placeholder="office, coding" disabled={disabled} spellCheck={false}
          onBlur={(event) => onSet({ tags: event.target.value.split(',').map((tag) => tag.trim()).filter(Boolean) })} /></label>
      {recipe && <p className="ct-hint cms-ed-recipe-note">{[recipe.kind, recipe.family, recipe.vision ? 'vision' : '', recipe.speculative ? `drafter ${recipe.speculative}` : ''].filter(Boolean).join(' · ')}</p>}
    </div>}
    <div className="cms-ed-status">
      {live && <span className={`ct-label ${tone === 'error' ? 'ct-warn' : tone === 'serving' ? 'ct-blue' : 'ct-faint'}`}>{live.state.toUpperCase()}{live.loaded ? ` · ${live.slots_busy ?? 0}/${live.slots ?? 0} BUSY` : ''}{live.tok_s ? ` · ${tokS(live.tok_s)} TOK/S` : ''}</span>}
      {!live && <span className="ct-label ct-faint">NEW · STARTS ON APPLY</span>}
      {stepText && <span className={`ct-label ${step === 'done' ? 'ct-blue' : step === 'sending' ? 'ct-faint' : 'ct-warn'}`}>{stepText}</span>}
      <span className="grow" />
      {fitted && <span className={`ct-label ${fitted.fits ? 'ct-faint' : 'ct-warn'}`}>{fitted.note || (fitted.fits ? 'FITS' : 'DOES NOT FIT')}</span>}
    </div>
    {typeof fit === 'string' && <p className="inline-error">Fit unknown: {fit}</p>}
    {live?.error && <p className="inline-error">{live.error}</p>}
    {step && step !== 'sending' && step !== 'done' && <p className="inline-error">{step}</p>}
  </div>;
}

function NumberField({ label, value, disabled, onChange }: { label: string; value?: number; disabled: boolean; onChange: (value: number | undefined) => void }) {
  return <label className="field cms-ed-num"><span className="ct-label ct-faint">{label}</span>
    <input type="number" min={0} value={value ?? ''} disabled={disabled} placeholder="default"
      onChange={(event) => onChange(event.target.value === '' ? undefined : Math.max(0, Math.round(Number(event.target.value))))} /></label>;
}
