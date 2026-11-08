/**
 * The left sidebar: the mesh as a mixing desk.
 *
 * - LIVE: the mesh's generated tokens per second, large, over a seismograph tape of the last
 *   four minutes (one 2px bar per sample, scrolling left), and one line of what it adds up to.
 * - CHANNELS: one strip per model, wherever it runs: its name and tokens per second, and a
 *   segmented level meter of its slots in use across the mesh, with a peak-hold tick (the
 *   highest level of the last 30 s) and its queue in red at the meter's end. A click outlines
 *   the servers that run it.
 * - LOG: the last lane events, one line each, the time in a narrow column.
 *
 * The servers are the grid's cards; this column says what the mesh is doing.
 */

import { useEffect, useRef, useState } from 'react';
import { runningOf, sumTotals, totalsOf } from './mesh';
import { useActivity, useServeVersion } from './serve-data';
import { millions, tokS } from './format';
import type { Computer } from './types';

export type ServerFilter = 'all' | 'serving' | 'idle' | 'offline';

const SAMPLE_MS = 2000;
/** Four minutes of samples on the tape. */
const TAPE = 120;
/** Thirty seconds of samples for a meter's peak hold. */
const PEAK = 15;
const SEGMENTS = 20;
const LOG_SHOWN = 6;

interface Channel { name: string; servers: number; slots: number; busy: number; queued: number; tok_s: number; tokens_hour: number; loaded: boolean }

function channelsOf(computers: Computer[]): Channel[] {
  const map = new Map<string, Channel & { nodes: Set<string> }>();
  for (const computer of computers) {
    if (!computer.reachable) continue;
    for (const run of runningOf(computer)) {
      const entry = map.get(run.name) ?? { name: run.name, servers: 0, slots: 0, busy: 0, queued: 0, tok_s: 0, tokens_hour: 0, loaded: false, nodes: new Set<string>() };
      entry.nodes.add(computer.node);
      entry.servers = entry.nodes.size;
      entry.slots += run.slots;
      entry.busy += run.busy;
      entry.queued += run.queued;
      entry.tok_s += run.tok_s;
      entry.tokens_hour += run.tokens_hour;
      entry.loaded ||= run.loaded;
      map.set(run.name, entry);
    }
  }
  return [...map.values()].sort((a, b) => b.tok_s - a.tok_s || b.slots - a.slots || a.name.localeCompare(b.name));
}

const level = (channel: Channel) => (channel.slots > 0 ? Math.min(1, channel.busy / channel.slots) : 0);

/** The tape and the peaks, sampled every 2 s from what the cards report (kept while the page lives). */
function useHistory(computers: Computer[]) {
  const latest = useRef(computers);
  latest.current = computers;
  const [history, setHistory] = useState<{ tape: number[]; peaks: Map<string, number[]> }>(() => ({ tape: [], peaks: new Map() }));
  useEffect(() => {
    const take = () => setHistory((current) => {
      const live = latest.current.filter((computer) => computer.reachable);
      const tok = sumTotals(live.map(totalsOf)).tok_s;
      const tape = [...current.tape, tok].slice(-TAPE);
      const peaks = new Map(current.peaks);
      for (const channel of channelsOf(latest.current)) peaks.set(channel.name, [...(peaks.get(channel.name) ?? []), level(channel)].slice(-PEAK));
      return { tape, peaks };
    });
    take();
    const timer = setInterval(take, SAMPLE_MS);
    return () => clearInterval(timer);
  }, []);
  return history;
}

export function Nav({ computers, highlight, onPick, onHighlight }: {
  computers: Computer[]; selected: string | null; filter: ServerFilter; highlight: string | null;
  onFilter: (filter: ServerFilter) => void; onPick: (node: string) => void; onHighlight: (model: string | null) => void;
}) {
  useServeVersion();
  const activity = useActivity().slice(0, LOG_SHOWN);
  const history = useHistory(computers);
  const totals = sumTotals(computers.filter((computer) => computer.reachable).map(totalsOf));
  const channels = channelsOf(computers);

  return (
    <div className="cx-sidebar-scroll cms-desk">
      <section className="cms-desk-block cms-live" aria-label="Live">
        <header className="cms-desk-head">
          <span className="ct-label">LIVE</span>
          <span className="grow" />
          <span className="ct-label ct-faint num">{totals.slots_busy}/{totals.slots} SLOTS</span>
        </header>
        <div className="cms-live-value">
          <span className="num">{tokS(totals.tok_s) === '—' ? '0' : tokS(totals.tok_s)}</span>
          <span className="ct-label ct-faint">TOKENS / S</span>
        </div>
        <Tape samples={history.tape} />
        <div className="ct-label ct-faint cms-live-line num">
          {totals.tokens_hour ? millions(totals.tokens_hour) : '0'} M/H · {totals.requests_min ? totals.requests_min.toFixed(1) : '0'} REQ/MIN{totals.ttft_ms ? ` · ${Math.round(totals.ttft_ms)} MS FIRST` : ''}
        </div>
      </section>

      <section className="cms-desk-block" aria-label="Channels">
        <header className="cms-desk-head">
          <span className="ct-label">CHANNELS</span>
          <span className="grow" />
          <span className="ct-label ct-faint num">{channels.length}</span>
        </header>
        <ul className="cms-channels">
          {channels.map((channel) => {
            const on = highlight === channel.name;
            const peak = Math.max(0, ...(history.peaks.get(channel.name) ?? [0]));
            return (
              <li key={channel.name}>
                <button type="button" className={`cms-channel${on ? ' on' : ''}${channel.loaded ? '' : ' cold'}`} aria-pressed={on} onClick={() => onHighlight(on ? null : channel.name)}
                  title={`${channel.name}: ${channel.busy} of ${channel.slots} slots busy${channel.queued ? `, ${channel.queued} waiting` : ''} on ${channel.servers} server${channel.servers === 1 ? '' : 's'} · ${millions(channel.tokens_hour)} M tokens this hour`}>
                  <span className="cms-channel-top">
                    <span className="cms-channel-name">{channel.name}</span>
                    <span className={`ct-label num cms-channel-val${channel.tok_s > 0 ? '' : ' ct-faint'}`}>{channel.tok_s > 0 ? `${tokS(channel.tok_s)} T/S` : channel.loaded ? 'READY' : 'COLD'}</span>
                  </span>
                  <span className="cms-channel-bottom">
                    <Meter value={level(channel)} peak={peak} queued={channel.queued} />
                    <span className="ct-label ct-faint num cms-channel-slots">{channel.busy}/{channel.slots}</span>
                  </span>
                </button>
              </li>
            );
          })}
          {!channels.length && <li className="cms-desk-none">No model is served on the mesh yet.</li>}
        </ul>
      </section>

      <section className="cms-desk-block cms-desk-log" aria-label="Log">
        <header className="cms-desk-head"><span className="ct-label">LOG</span></header>
        <ul className="cms-log">
          {activity.map((event) => (
            <li key={event.id}>
              <button type="button" className="cms-log-row" onClick={() => onPick(event.node)} title={`${event.computer} · ${event.lane}: ${event.text}`}>
                <span className={`cms-log-mark ${event.tone}`} aria-hidden />
                <span className="cms-log-text"><b>{event.lane}</b> {event.text}</span>
                <span className="ct-label ct-faint num cms-log-at">{since(event.at)}</span>
              </button>
            </li>
          ))}
          {!activity.length && <li className="cms-desk-none">Lane events appear here.</li>}
        </ul>
      </section>
    </div>
  );
}

/** The seismograph: one bar per sample, the newest at the right, scaled to the window's peak. */
function Tape({ samples }: { samples: number[] }) {
  const width = TAPE * 3;
  const height = 36;
  const top = Math.max(1, ...samples);
  const offset = TAPE - samples.length;
  return (
    <svg className="cms-tape" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden>
      <line x1="0" y1={height - 0.5} x2={width} y2={height - 0.5} className="cms-tape-base" />
      {samples.map((value, index) => {
        // Idle samples still tick, so the tape is seen running at zero.
        const h = value > 0 ? Math.max(3, (value / top) * (height - 2)) : 2;
        const last = index === samples.length - 1;
        return <rect key={index} x={(offset + index) * 3} y={height - 1 - h} width="2" height={h} className={value > 0 ? (last ? 'now' : undefined) : 'idle'} />;
      })}
    </svg>
  );
}

/** A level meter: lit segments up to the level, a peak tick, the queue in red at the end. */
function Meter({ value, peak, queued }: { value: number; peak: number; queued: number }) {
  const lit = Math.round(value * SEGMENTS);
  const peakAt = Math.min(SEGMENTS - 1, Math.max(0, Math.round(peak * SEGMENTS) - 1));
  const waiting = Math.min(4, queued);
  return (
    <span className="cms-meter" aria-hidden>
      {Array.from({ length: SEGMENTS }, (_, index) => {
        const cls = index >= SEGMENTS - waiting ? 'q' : index < lit ? (index >= SEGMENTS * 0.8 ? 'hot' : 'on') : peak > 0 && index === peakAt ? 'peak' : '';
        return <i key={index} className={cls || undefined} />;
      })}
    </span>
  );
}

/** `now`, `40s`, `4m`, `2h`. */
function since(at: number): string {
  const s = Math.max(0, Math.round((Date.now() - at) / 1000));
  if (s < 10) return 'NOW';
  if (s < 60) return `${s}S`;
  if (s < 3600) return `${Math.round(s / 60)}M`;
  return `${Math.round(s / 3600)}H`;
}
