/**
 * The hub: CortexModelServer's window (BUILD-PLAN §4). ct's Desk layout, with the model servers of
 * the mesh in it: the sidebar | the cluster summary over the server grid | the server editor, each
 * divider draggable (double-click: default width), each side able to fold (⌘B, ⌘⇧K) and its width
 * kept in the workbench's durable state.
 *
 * The grid has one card per model server: this one, then every mesh member running cortexmodelserver
 * (`mesh.peers`, local=false, its card's software).
 * A card renders inside its computer's `ComputerScope`, so its calls go to that frame (a member
 * through `mesh.link`). Clicking a card opens its server in the editor on the right.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ComputerScope, Divider, runtimeFor, useHost, useShared } from '@cortex/sdk';
import { ServerGrid, useClusterLayout } from './Grid';
import { Summary } from './Summary';
import { Editor } from './Editor';
import { Nav, type ServerFilter } from './Nav';
import { LOCAL_COMPUTER, runningOf, sortValue, stateOf, type SortKey } from './mesh';
import { nameComputer, useServe, useServeSelect } from './serve-data';
import type { Card, Computer, Peer } from './types';
import './sidebar.css';
import './desk.css';
import './cluster.css';
import './hub.css';

const NAV = { def: 220, min: 160, max: 420, fold: 110 };
const SIDE = { def: 420, min: 320, max: 760, fold: 220 };
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

/** The three columns' widths and folds, in the workbench's durable state (they survive restarts). */
function useColumns() {
  const work = useRef<HTMLDivElement>(null);
  const [nav, setNav] = useShared<number>('cms.layout.nav', NAV.def);
  const [side, setSide] = useShared<number>('cms.layout.side', SIDE.def);
  const [navOpen, setNavOpen] = useShared<boolean>('cms.layout.navOpen', true);
  const [sideOpen, setSideOpen] = useShared<boolean>('cms.layout.sideOpen', true);
  const [drag, setDrag] = useState<{ nav?: number; side?: number }>({});
  const box = () => work.current?.getBoundingClientRect();
  const navWidth = drag.nav ?? nav;
  const sideWidth = drag.side ?? side;
  return {
    work,
    navOpen, sideOpen,
    toggleNav: () => setNavOpen((open) => !open),
    toggleSide: () => setSideOpen((open) => !open),
    openSide: () => setSideOpen(true),
    style: { gridTemplateColumns: `${navOpen ? navWidth : 0}px ${navOpen ? 1 : 0}px minmax(0,1fr) ${sideOpen ? 1 : 0}px ${sideOpen ? sideWidth : 0}px` },
    moveNav: (x: number) => { const rect = box(); if (rect) setDrag({ nav: clamp(x - rect.left, NAV.fold - 20, NAV.max) }); },
    endNav: () => {
      const width = drag.nav;
      setDrag({});
      if (width === undefined) return;
      if (width < NAV.fold) setNavOpen(false);
      else setNav(clamp(width, NAV.min, NAV.max));
    },
    resetNav: () => { setNav(NAV.def); setNavOpen(true); },
    moveSide: (x: number) => { const rect = box(); if (rect) setDrag({ side: clamp(rect.right - x, SIDE.fold - 20, SIDE.max) }); },
    endSide: () => {
      const width = drag.side;
      setDrag({});
      if (width === undefined) return;
      if (width < SIDE.fold) setSideOpen(false);
      else setSide(clamp(width, SIDE.min, SIDE.max));
    },
    resetSide: () => { setSide(SIDE.def); setSideOpen(true); },
  };
}

/** This computer and every mesh member, with their cards; refreshed on `mesh.changed` and every 10 s. */
function useComputers(): { computers: Computer[]; reread: () => void } {
  const client = runtimeFor(null).client;
  const [peers, setPeers] = useState<Computer[]>([]);
  const [localCard, setLocalCard] = useState<Card | null>(null);
  const read = useCallback(() => {
    client.call<{ peers: Peer[] }>('mesh.peers', {}, { signal: AbortSignal.timeout(8000) }).then((list) => setPeers((list?.peers ?? [])
      // Only model servers: another app on this machine is not another computer, and apps that
      // are not cortexmodelserver (CortexCode, CortexTerm, the knowledge app) have no runners to manage.
      .filter((peer) => !peer.member.local && peer.card?.software === 'cortexmodelserver')
      .map((peer) => ({ node: peer.member.node, name: peer.card?.name || peer.member.name, reachable: peer.reachable, path: peer.path, card: peer.card ?? null }))))
      .catch(() => undefined);
    client.call<Card>('mesh.card', {}, { signal: AbortSignal.timeout(8000) }).then(setLocalCard).catch(() => undefined);
  }, [client]);
  useEffect(() => {
    const off = client.on('mesh.changed', () => read());
    read();
    const timer = setInterval(read, 10_000);
    return () => { off(); clearInterval(timer); };
  }, [read, client]);
  const computers = useMemo(() => [{ ...LOCAL_COMPUTER, card: localCard }, ...peers], [localCard, peers]);
  useEffect(() => { for (const computer of computers) nameComputer(computer.node, computer.name); }, [computers]);
  return { computers, reread: read };
}

export function Hub() {
  const host = useHost();
  const columns = useColumns();
  const { computers } = useComputers();
  const cluster = useClusterLayout(computers);
  const [settings, setSettings] = useShared<string | null>('settings.open', null);
  const [filter, setFilter] = useShared<ServerFilter>('cms.nav.filter', 'all');
  const [selected, setSelected] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<string | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; order: string[] } | null>(null);
  const local = useServe('', true);

  const membership = computers.map((computer) => computer.node).join('\n');
  const sortBy = useCallback((key: SortKey | null) => {
    if (!key) { setSort(null); return; }
    // Sorted once, when asked: the cards stay put while their numbers move.
    const order = [...computers].sort((a, b) => sortValue(b, key) - sortValue(a, key) || a.name.localeCompare(b.name)).map((computer) => computer.node);
    setSort({ key, order });
  }, [computers]);
  useEffect(() => { if (sort) sortBy(sort.key); /* a computer came or went */ }, [membership]); // eslint-disable-line react-hooks/exhaustive-deps

  // Which cards the filter and the highlighted model pick, re-read as the servers report.
  const shownKey = useServeSelect(() => computers.filter((computer) => filter === 'all' || stateOf(computer) === filter).map((computer) => computer.node).join('\n'));
  const highlightKey = useServeSelect(() => (highlight ? computers.filter((computer) => runningOf(computer).some((run) => run.name === highlight)).map((computer) => computer.node).join('\n') : ''));
  const shownComputers = useMemo(() => { const nodes = new Set(shownKey.split('\n')); return computers.filter((computer) => nodes.has(computer.node)); }, [shownKey, computers]);
  const highlighted = useMemo(() => new Set(highlightKey ? highlightKey.split('\n') : []), [highlightKey]);

  const pick = useCallback((node: string) => {
    setSelected(node);
    columns.openSide();
    requestAnimationFrame(() => document.querySelector(`[data-computer="${CSS.escape(node || 'local')}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }));
  }, [columns]);

  const editing = selected === null ? null : computers.find((computer) => computer.node === selected) ?? null;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const meta = event.metaKey || event.ctrlKey;
      if (!meta) return;
      const key = event.key.toLowerCase();
      if (key === ',') { event.preventDefault(); setSettings((open) => (open === null ? '' : null)); }
      else if (key === 'b') { event.preventDefault(); columns.toggleNav(); }
      else if (key === 'k' && event.shiftKey) { event.preventDefault(); columns.toggleSide(); }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [columns, setSettings]);

  const bandTarget = document.getElementById('ct-band');
  const servingLanes = local.lanes.filter((lane) => lane.state === 'active').length;
  // Status readout and outlined actions in the band; the settings control stays at the right.
  const band = (
    <>
      <span className="grow" />
      <span className={`ct-band-note${!local.absent && !local.loading ? ' blue' : ''}`} title="This computer's model server">
        {local.absent ? 'NO MODEL SERVER HERE' : local.loading ? 'SERVER …' : `${servingLanes}/${local.lanes.length} LANES ACTIVE`}
      </span>
      <button type="button" className="detent ct-band-gear" onClick={() => setSettings((open) => (open === null ? '' : null))} title="Settings (⌘,)" aria-label="Settings">
        <svg viewBox="0 0 24 24" width="13" height="13" aria-hidden fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z" />
        </svg>
      </button>
    </>
  );

  return (
    <div className="ct-desk">
      {bandTarget && createPortal(band, bandTarget)}
      <div ref={columns.work} className="ct-work" style={columns.style}>
        <nav className="ct-nav cx-sidebar" aria-label="Servers and models" hidden={!columns.navOpen}>
          <Nav computers={computers} selected={selected} filter={filter} highlight={highlight}
            onFilter={setFilter} onPick={pick} onHighlight={setHighlight} />
        </nav>
        <Divider name="ct-nav" label="Sidebar width" onMove={columns.moveNav} onEnd={columns.endNav} onReset={columns.resetNav} className={columns.navOpen ? '' : 'ct-hidden'} />
        <main className="ct-main cms-main">
          <Summary computers={computers} sort={sort?.key ?? null} onSort={sortBy} />
          <div className="ct-cluster-scroll">
            <ServerGrid computers={shownComputers} selected={columns.sideOpen ? selected : null} highlighted={highlighted}
              layout={cluster.layout} order={sort?.order ?? cluster.order}
              onLayout={cluster.setLayout} onReordered={() => setSort(null)} onSelect={pick} />
          </div>
        </main>
        <Divider name="ct-side" label="Editor width" onMove={columns.moveSide} onEnd={columns.endSide} onReset={columns.resetSide} className={columns.sideOpen ? '' : 'ct-hidden'} />
        {columns.sideOpen ? (
          <aside className="ct-side" aria-label="Server editor">
            {editing
              ? <ComputerScope key={editing.node || 'local'} node={editing.node || null}><Editor computer={editing} /></ComputerScope>
              : <div className="ct-machine-offline"><span className="ct-reading-name">No server open</span>
                  <p className="ct-hint">Click a card to edit that computer's lanes.</p>
                  <button type="button" className="detent" onClick={() => pick('')}>THIS COMPUTER</button></div>}
          </aside>
        ) : <div />}
      </div>
      {settings !== null && <host.Slot name="settings.host@1" className="cx-settings-layer" props={{ onClose: () => setSettings(null) }} />}
    </div>
  );
}
