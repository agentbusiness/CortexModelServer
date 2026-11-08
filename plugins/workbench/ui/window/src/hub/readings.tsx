/**
 * The machine readings every server card shows, as in ct's machine pane: six tiles (CPU, MEM, GPU,
 * NET, DISK, TEMP) from `system.metrics` and the `system.sample` stream, each with a 60-sample
 * sparkline. Copied from ct's Sidebar (the parts live in ct's bundle, so they are copied, not
 * imported).
 */

import { useEffect, useState } from 'react';
import { errorMessage, useRuntime } from '@cortex/sdk';
import type { Metrics } from './api';
import { GB, rate, size } from './api';

const HISTORY = 60;
export function Spark({ values, max, warn }: { values: number[]; max?: number; warn?: boolean }) {
  if (values.length < 2) return <svg className="ct-spark" viewBox="0 0 120 22" preserveAspectRatio="none" aria-hidden />;
  const top = max ?? Math.max(1, ...values);
  const points = values.map((value, index) => `${(index / (values.length - 1)) * 120},${21 - (Math.min(value, top) / top) * 19}`).join(' ');
  return (
    <svg className="ct-spark" viewBox="0 0 120 22" preserveAspectRatio="none" aria-hidden>
      <polyline points={points} fill="none" stroke={warn ? 'var(--rubric-lift)' : 'var(--counter)'} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** Memory over time, stacked: what apps hold, then what is compressed, up to all the RAM; swap
 *  (on disk) as a dashed line on the same scale. */
function MemChart({ history, total }: { history: Metrics[]; total: number }) {
  if (history.length < 2 || !total) return <svg className="ct-memchart" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden />;
  const x = (index: number) => (index / (history.length - 1)) * 120;
  const y = (bytes: number) => 33 - (Math.min(bytes, total) / total) * 32;
  const line = (value: (m: Metrics) => number) => history.map((m, index) => `${x(index)},${y(value(m))}`).join(' ');
  const app = (m: Metrics) => m.memory_used - m.memory_compressed;
  const area = (top: (m: Metrics) => number, bottom: (m: Metrics) => number) =>
    `${line(top)} ${history.map((m, index) => `${x(index)},${y(bottom(m))}`).reverse().join(' ')}`;
  return (
    <svg className="ct-memchart" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden>
      <polygon points={area(app, () => 0)} fill="color-mix(in srgb, var(--counter) 22%, transparent)" />
      <polygon points={area((m) => m.memory_used, app)} fill="color-mix(in srgb, var(--rubric) 26%, transparent)" />
      <polyline points={line(app)} fill="none" stroke="var(--counter)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
      <polyline points={line((m) => m.memory_used)} fill="none" stroke="var(--rubric)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
      <polyline points={line((m) => m.swap_used)} fill="none" stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="3 2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** CPU (blue) and GPU (orange) temperature over time, on a 30–100 °C scale. */
function TempChart({ history }: { history: Metrics[] }) {
  if (history.length < 2) return <svg className="ct-memchart" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden />;
  const x = (index: number) => (index / (history.length - 1)) * 120;
  const y = (celsius: number) => 33 - ((Math.min(Math.max(celsius, 30), 100) - 30) / 70) * 32;
  const line = (value: (m: Metrics) => number | null | undefined) => history.filter((m) => value(m) != null).length < 2 ? '' :
    history.map((m, index) => (value(m) != null ? `${x(index)},${y(value(m)!)}` : '')).filter(Boolean).join(' ');
  const cpu = line((m) => m.cpu_temp);
  const gpu = line((m) => m.gpu_temp);
  return (
    <svg className="ct-memchart" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden>
      <line x1="0" x2="120" y1={y(90)} y2={y(90)} stroke="var(--rule-2)" strokeDasharray="2 3" vectorEffect="non-scaling-stroke" />
      {gpu && <polyline points={gpu} fill="none" stroke="var(--rubric)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />}
      {cpu && <polyline points={cpu} fill="none" stroke="var(--counter)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />}
    </svg>
  );
}

/** Samples kept across mounts, per computer, so switching the side pane away and back keeps the curves. */
const kept = new Map<string, Metrics[]>();

export function useMetrics(enabled = true) {
  const { client, node } = useRuntime();
  const key = node ?? '';
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [history, setHistory] = useState<Metrics[]>([]);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    let stopped = false;
    const samples = kept.get(key) ?? [];
    setHistory(samples);
    setMetrics(samples[samples.length - 1] ?? null);
    setProblem(null);
    if (!enabled) return;
    const accept = (sample: Metrics) => {
      if (stopped || !sample?.at) return;
      const previous = kept.get(key) ?? [];
      if ((previous.at(-1)?.at ?? 0) > sample.at) return;
      const next = previous.at(-1)?.at === sample.at ? previous : [...previous.slice(-(HISTORY - 1)), sample];
      kept.set(key, next);
      setMetrics(sample);
      setHistory(next);
      setProblem(null);
    };
    const read = () => client.call<Metrics>('system.metrics', {}, { signal: AbortSignal.timeout(8000) })
      .then(accept).catch((cause) => { if (!stopped) setProblem(errorMessage(cause)); });
    const off = client.on('system.sample', (_topic, data) => accept(data as Metrics));
    void read();
    // Refresh after a missed event or reconnect as well as receiving the live stream.
    const timer = setInterval(() => void read(), 10_000);
    return () => { stopped = true; off(); clearInterval(timer); };
  }, [client, key, enabled]);
  return { metrics, history, problem };
}

/** The same measurements in the machine pane and in every cluster card. */
export function MachineReadings({ metrics, history }: { metrics: Metrics | null; history: Metrics[] }) {
  return <>
      <div className="ct-tiles three">
        <Tile label="CPU" value={metrics ? `${Math.round(metrics.cpu)}%` : '–'} values={history.map((m) => m.cpu)} max={100}
          note={metrics && metrics.p_cores ? `P ${metrics.p_cores} · E ${metrics.e_cores} · load ${metrics.load[0].toFixed(1)}` : metrics ? `load ${metrics.load[0].toFixed(1)}` : ''} />
        <div className="ct-tile">
          <div className="ct-tile-head" title={metrics ? `Memory pressure: ${metrics.pressure}` : undefined}><span className={metrics && metrics.pressure !== 'normal' ? 'ct-warn' : 'ct-faint'}>MEM</span><span className="grow" /><span>{metrics ? `${(metrics.memory_used / GB).toFixed(1)}/${Math.round(metrics.memory_total / GB)} GB` : '–'}</span></div>
          <MemChart history={history} total={metrics?.memory_total ?? 0} />
          <div className="ct-tile-note ct-memkey">{metrics ? <>
            <span className="app" title={`Apps: ${size(metrics.memory_used - metrics.memory_compressed)}`}>a:{((metrics.memory_used - metrics.memory_compressed) / GB).toFixed(1)}</span>
            <span className="comp" title={`Compressed: ${size(metrics.memory_compressed)}`}>c:{(metrics.memory_compressed / GB).toFixed(1)}</span>
            {metrics.swap_used > 0 && <span className="swap" title={`Swap: ${size(metrics.swap_used)}`}>s:{(metrics.swap_used / GB).toFixed(1)}</span>}
          </> : ''}</div>
        </div>
        <Tile label="GPU" value={metrics?.gpu != null ? `${Math.round(metrics.gpu)}%` : '–'} values={history.map((m) => m.gpu ?? 0)} max={100}
          note={metrics ? [metrics.gpu_active != null ? `awake ${Math.round(metrics.gpu_active)}%` : '', metrics.gpu_power != null ? `${metrics.gpu_power.toFixed(1)} W` : '', metrics.ane_power ? `ANE ${metrics.ane_power.toFixed(1)} W` : ''].filter(Boolean).join(' · ') : ''} />
      </div>
      <div className="ct-tiles three">
        <Tile label="NET" value={metrics ? `↓${rate(metrics.net_rx)}` : '–'} values={history.map((m) => m.net_rx + m.net_tx)} note={metrics ? `↑${rate(metrics.net_tx)}` : ''} />
        <Tile label="DISK" value={metrics ? `${Math.round(metrics.disk_free / GB)} GB free` : '–'} values={history.map((m) => m.disk_write)}
          note={metrics ? `r ${rate(metrics.disk_read)} · w ${rate(metrics.disk_write)}` : ''} />
        <div className="ct-tile">
          <div className="ct-tile-head"><span className="ct-faint">TEMP</span><span className="grow" /><span className={(metrics?.cpu_temp ?? 0) >= 90 ? 'ct-warn' : ''}>{metrics?.cpu_temp != null ? `${Math.round(metrics.cpu_temp)}°C` : '–'}</span></div>
          <TempChart history={history} />
          <div className="ct-tile-note ct-memkey">{metrics ? <>
            {metrics.cpu_temp != null && <span className="app" title="CPU">c:{Math.round(metrics.cpu_temp)}°</span>}
            {metrics.gpu_temp != null && <span className="comp" title="GPU">g:{Math.round(metrics.gpu_temp)}°</span>}
            <span className="swap" title="Fan">f:{metrics.fan_rpm ? `${metrics.fan_rpm} rpm` : '–'}</span>
          </> : ''}</div>
        </div>
      </div>
  </>;
}


export function Tile({ label, value, values, max, note }: { label: string; value: string; values: number[]; max?: number; note?: string }) {
  return (
    <div className="ct-tile">
      <div className="ct-tile-head"><span className="ct-faint">{label}</span><span className="grow" /><span>{value}</span></div>
      <Spark values={values} max={max} />
      {note && <div className="ct-faint ct-tile-note">{note}</div>}
    </div>
  );
}
