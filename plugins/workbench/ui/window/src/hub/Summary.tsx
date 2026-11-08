/**
 * The cluster summary over the grid (BUILD-PLAN §4.1): the mesh's GPU load, memory, throughput,
 * requests, slots, servers and power, each a tile of the cards' anatomy (head, sparkline, note),
 * larger. Live values come from `serve.cluster.get` (every 5 s) with the cards' own live samples
 * summed over it; the sparklines from `serve.history.read {scope: "cluster"}` over the chosen
 * range. A tile pressed sorts the grid by its number.
 */

import { useEffect, useRef, useState } from 'react';
import { errorMessage, runtimeFor, useShared } from '@cortex/sdk';
import { Spark } from './readings';
import { gb, gbWhole, millions, noServer, pct, tokS } from './format';
import { isServer, sumTotals, totalsOf, type SortKey } from './mesh';
import { serveOf, useServeVersion } from './serve-data';
import type { Cluster, Computer, History, HistoryPoint, Totals } from './types';

const RANGES = [{ s: 3600, label: '1H' }, { s: 86_400, label: '24H' }, { s: 604_800, label: '7D' }] as const;
/** Live points kept here when the server keeps no history (one per 5 s poll). */
const LIVE_POINTS = 120;

export function Summary({ computers, sort, onSort }: { computers: Computer[]; sort: SortKey | null; onSort: (key: SortKey | null) => void }) {
  useServeVersion();
  const client = runtimeFor(null).client;
  const [range, setRange] = useShared<number>('cms.summary.range', 3600);
  const [cluster, setCluster] = useState<Cluster | null>(null);
  const [clusterProblem, setClusterProblem] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryPoint[] | null>(null);
  const live = useRef<HistoryPoint[]>([]);
  const [, setTick] = useState(0);

  useEffect(() => {
    let stopped = false;
    const read = () => client.call<Cluster>('serve.cluster.get', {}, { signal: AbortSignal.timeout(8000) })
      .then((value) => {
        if (stopped) return;
        setCluster(value);
        setClusterProblem(null);
        live.current = [...live.current.slice(-(LIVE_POINTS - 1)), { at_ms: value.updated_ms || Date.now(), totals: value.totals }];
        setTick((n) => n + 1);
      })
      .catch((cause) => { if (!stopped) { setCluster(null); setClusterProblem(noServer(cause) ? 'NO MODEL SERVER HERE · ADDED UP FROM THE CARDS' : errorMessage(cause)); } });
    void read();
    const timer = setInterval(read, 5000);
    return () => { stopped = true; clearInterval(timer); };
  }, [client]);

  useEffect(() => {
    let stopped = false;
    const read = () => client.call<History>('serve.history.read', { scope: 'cluster', id: '', range_s: range }, { signal: AbortSignal.timeout(15_000) })
      .then((value) => { if (!stopped) setHistory(value?.points ?? []); })
      .catch(() => { if (!stopped) setHistory(null); });
    void read();
    const timer = setInterval(read, range > 3600 ? 300_000 : 60_000);
    return () => { stopped = true; clearInterval(timer); };
  }, [client, range]);

  // The cards' live samples, summed: newer than the cluster's 5 s poll, for what moves fast.
  const sampled = computers.filter((computer) => computer.reachable && serveOf(computer.node).sampleAt > Date.now() - 6000);
  const cardSum = sumTotals(computers.map(totalsOf));
  const totals: Totals = cluster
    ? sampled.length >= Math.max(1, cluster.reporting)
      ? { ...cluster.totals, ...pick(sumTotals(sampled.map(totalsOf)), FAST) }
      : cluster.totals
    : cardSum;
  const servers = computers.filter(isServer);
  const online = cluster ? cluster.servers_online : servers.filter((computer) => computer.reachable).length;
  const total = cluster ? cluster.servers : servers.length;
  const gpus = cluster?.gpus ?? computers.reduce((sum, computer) => sum + (serveOf(computer.node).gpus.length || computer.card?.gpus?.length || 0), 0);

  const points = history && history.length > 1 ? history : live.current;
  const series = (value: (t: Totals) => number) => points.map((point) => value(point.totals));

  const tiles: { key: SortKey; label: string; value: string; values: number[]; max?: number; note: string; warn?: boolean }[] = [
    { key: 'gpu_load', label: 'GPU LOAD', value: pct(totals.gpu_util), values: series((t) => t.gpu_util * 100), max: 100, note: `${gpus} GPU${gpus === 1 ? '' : 's'} · by memory` },
    { key: 'gpu_memory', label: 'GPU MEMORY', value: totals.gpu_bytes ? `${gb(totals.gpu_used_bytes)}/${gbWhole(totals.gpu_bytes)}` : '–', values: series((t) => t.gpu_used_bytes), max: totals.gpu_bytes || undefined, note: totals.gpu_bytes ? `${pct(totals.gpu_used_bytes / totals.gpu_bytes)} in use` : '' },
    { key: 'tok_s', label: 'TOKENS/S', value: tokS(totals.tok_s), values: series((t) => t.tok_s), note: `prefill ${tokS(totals.prefill_tok_s)}` },
    { key: 'tokens_hour', label: 'TOKENS/HOUR', value: `${millions(totals.tokens_hour)} M`, values: series((t) => t.tokens_hour), note: `${millions(totals.tokens_day)} M last 24 h` },
    { key: 'requests', label: 'REQUESTS/MIN', value: totals.requests_min ? totals.requests_min.toFixed(totals.requests_min >= 10 ? 0 : 1) : '0', values: series((t) => t.requests_min), note: totals.ttft_ms ? `TTFT ${Math.round(totals.ttft_ms)} ms median` : 'TTFT –' },
    { key: 'slots', label: 'SLOTS', value: `${totals.slots_busy}/${totals.slots}`, values: series((t) => t.slots_busy), max: totals.slots || undefined, note: `${totals.queued} queued`, warn: totals.queued > 0 },
    { key: 'servers', label: 'SERVERS', value: `${online}/${total}`, values: series((t) => t.lanes_serving), note: `lanes ${totals.lanes_serving} serving · ${totals.lanes_loading} loading · ${totals.lanes_failed} failed`, warn: totals.lanes_failed > 0 },
    { key: 'power', label: 'POWER', value: totals.power_w ? `${Math.round(totals.power_w)} W` : '–', values: series((t) => t.power_w), note: totals.power_w ? 'GPUs that report it' : 'not reported' },
  ];
  const sortedLabel = tiles.find((tile) => tile.key === sort)?.label;

  return (
    <section className="cms-summary" aria-label="Cluster summary">
      <div className="cms-summary-head">
        <span className="ct-label">CLUSTER</span>
        <span className="ct-label ct-faint">{online} ONLINE · {computers.length} COMPUTER{computers.length === 1 ? '' : 'S'}</span>
        {clusterProblem && <span className="ct-label ct-faint" title={clusterProblem}>{clusterProblem.length > 60 ? 'CLUSTER TOTALS UNAVAILABLE · ADDED UP FROM THE CARDS' : clusterProblem}</span>}
        <span className="grow" />
        {sortedLabel && <button type="button" className="ct-link" onClick={() => onSort(null)} title="Back to your order">SORTED BY {sortedLabel} ×</button>}
        <div className="seg-set" role="group" aria-label="Sparkline range">
          {RANGES.map((entry) => <button key={entry.s} type="button" aria-pressed={range === entry.s} onClick={() => setRange(entry.s)}>{entry.label}</button>)}
        </div>
      </div>
      <div className="cms-summary-tiles">
        {tiles.map((tile) => (
          <button key={tile.key} type="button" className="ct-tile cms-sum-tile" aria-pressed={sort === tile.key}
            title={`Sort the servers by ${tile.label.toLowerCase()}`} onClick={() => onSort(sort === tile.key ? null : tile.key)}>
            <span className="ct-tile-head"><span className="ct-faint">{tile.label}</span><span className="grow" /></span>
            <span className={`cms-sum-value${tile.warn ? ' ct-warn' : ''}`}>{tile.value}</span>
            <Spark values={tile.values} max={tile.max} warn={tile.warn} />
            <span className="ct-faint ct-tile-note">{tile.note}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

/** The fields the cards' 2 s samples know better than the cluster's 5 s poll. */
const FAST: (keyof Totals)[] = ['gpu_util', 'gpu_used_bytes', 'tok_s', 'prefill_tok_s', 'slots_busy', 'queued', 'power_w'];
function pick(totals: Totals, keys: (keyof Totals)[]): Partial<Totals> {
  return Object.fromEntries(keys.map((key) => [key, totals[key]]));
}
