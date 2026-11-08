/**
 * settings.mesh@1 — joining computers. Nothing to set: an invitation (one hour, used once) to copy
 * for another computer, a box to paste one into, and the joined computers with Remove. What runs
 * where (a session's model, an agent, a command) is decided by the host from what each computer
 * reports; the list shows it as it happens. CortexModelServer: every member may work here (the
 * model servers administer each other; there is no switch), and each member is one line: whether
 * it is reachable, its name, its app, its computer and models, how it is reached, Remove.
 */

import { useState } from 'react';
import { definePlugin, errorMessage, useClient, useHost, useNow, useQuery } from '@cortex/sdk';
import './mesh.css';

interface Member { node: string; name: string; added_ms: number; by: string; local?: boolean }
interface ModelCard { plugin: string; model_id: string; variant: string; family: string; tags?: string[]; slots: number; slots_busy: number; queued: number; tok_s: number }
interface Card { software: string; chip: string; memory_bytes: number; free_memory_bytes: number; on_mains: boolean; user_active: boolean; models?: ModelCard[]; languages: { id: string; version: string }[]; docker: boolean; agents_running: number }
interface Peer { member: Member; card?: Card | null; reachable: boolean; path: string; rtt_ms: number; serving: string[]; computer?: string }
interface PairCode { code: string; expires_ms: number; invitation: string }
interface Placed { session_id: string; node: string; name: string }
interface Network { id: string; founder: string; founded_ms: number; members: number }
interface Status { node: string; name: string; members: number; reachable: number; network?: Network | null; pairing?: PairCode | null; serving: [string, string][]; placed: Placed[]; allowed: string[] }

function gb(bytes: number): string {
  const value = bytes / (1024 * 1024 * 1024);
  return `${value < 10 ? value.toFixed(1).replace(/\.0$/, '') : Math.round(value)} GB`;
}

/** The app identifies itself in its card; model tags do not identify the app. */
function softwareName(software?: string): string {
  if (software === 'cortexcode') return 'CortexCode';
  if (software === 'cortexterm') return 'CortexTerm';
  if (software === 'cortexmodelserver') return 'Model server';
  return software || '';
}

function where(path: string): string {
  if (path === 'lan') return 'LAN';
  if (path === 'relay') return 'RELAY';
  if (path === 'direct') return 'DIRECT';
  return '';
}

/** One computer: its apps on the network, together (Remove takes the computer out as a whole). */
interface Computer { key: string; name: string; here: boolean; peers: Peer[] }

/** The members by computer: the machine key the log ties them to; a member without one (an older
 *  app) goes with the computer of its name. */
function byComputer(peers: Peer[]): Computer[] {
  const groups = new Map<string, Computer>();
  const nameKey = new Map<string, string>();
  for (const peer of peers) if (peer.computer) nameKey.set(peer.member.name, peer.computer);
  for (const peer of peers) {
    const key = peer.member.local ? 'here' : peer.computer || nameKey.get(peer.member.name) || `name:${peer.member.name}`;
    const group = groups.get(key) ?? { key, name: peer.member.name, here: Boolean(peer.member.local), peers: [] };
    group.peers.push(peer);
    groups.set(key, group);
  }
  const rank = (computer: Computer) => (computer.here ? 0 : computer.peers.some((peer) => peer.reachable) ? 1 : 2) * 2
    + (computer.peers.some((peer) => peer.card?.software === 'cortexmodelserver') ? 0 : 1);
  return [...groups.values()].sort((a, b) => rank(a) - rank(b) || a.name.localeCompare(b.name));
}

function MeshPanel() {
  const client = useClient();
  const host = useHost();
  const status = useQuery<Status>('mesh.status', (c) => c.call('mesh.status', {}), { topics: ['mesh.changed', 'plugin.swapped'] });
  const peers = useQuery<{ peers: Peer[] }>('mesh.peers', (c) => c.call('mesh.peers', {}), { topics: ['mesh.changed'] });
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [token, setToken] = useState('');
  const [copied, setCopied] = useState(false);
  const now = useNow(1000, Boolean(status.data?.pairing));

  const run = async (what: string, work: () => Promise<string | void>) => {
    setBusy(what);
    setError(null);
    try {
      const said = await work();
      if (said) host.notify(said);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(null);
      await Promise.all([status.refresh(), peers.refresh()]);
    }
  };

  const data = status.data;
  if (status.error && !data) {
    return (
      <section className="cx-mesh" data-build={__CORTEX_BUILD__}>
        <p className="cx-mesh-note">Joining computers is not available: {status.error}</p>
      </section>
    );
  }
  if (!data) return null;
  const pairing = data.pairing && data.pairing.expires_ms > now ? data.pairing : null;
  const minutesLeft = pairing ? Math.max(1, Math.round((pairing.expires_ms - now) / 60000)) : 0;
  const others = (peers.data?.peers ?? []).filter((peer) => peer.member.node !== data.node);

  const invite = () => void run('pair', async () => {
    const code = pairing ?? (await client.call<PairCode>('mesh.pair.start', {}));
    try {
      await navigator.clipboard?.writeText(code.invitation);
      setCopied(true);
      return 'Invitation copied: paste it on the other computer';
    } catch {
      return 'Invitation ready below: copy it to the other computer';
    }
  });

  return (
    <section className="cx-mesh" data-build={__CORTEX_BUILD__}>
      <div className="cx-mesh-grid">
        <span className="ap">This computer</span>
        <span className="t">{data.name}</span>
        <span />

        <span className="ap">Network</span>
        {data.network ? (
          <>
            <span className="v">started by {data.network.founder} on {new Date(data.network.founded_ms).toLocaleDateString()} · {data.reachable} of {others.length} other{others.length === 1 ? '' : 's'} reachable</span>
            <button type="button" className="detent tiny warn" data-action="leave" disabled={busy !== null} onClick={() => void run('leave', async () => { await client.call('mesh.leave', {}); return 'Left the network'; })}>Leave the network</button>
          </>
        ) : (
          <>
            <span className="v faint">none yet: make an invitation, or join with one</span>
            <span />
          </>
        )}

        <span className="ap">Add a computer</span>
        <span className="cx-mesh-cell">
          <button type="button" className="detent tiny send" data-action="pair" disabled={busy !== null} onClick={invite}>{pairing ? (copied ? 'Copy the invitation again' : 'Copy the invitation') : 'Make an invitation'}</button>
          {pairing && <span className="v faint">good for {minutesLeft} min, one computer</span>}
        </span>
        {pairing ? <button type="button" className="detent tiny" disabled={busy !== null} onClick={() => void run('cancel', async () => { await client.call('mesh.pair.cancel', {}); setCopied(false); })}>Withdraw</button> : <span />}
        <span />
        {pairing ? (
          <div className="cx-mesh-inv" data-testid="invitation">{pairing.invitation}</div>
        ) : (
          <span className="cx-mesh-note">Paste it on the other computer: in its Settings → Mesh, or with <code>cx mesh join '…'</code>.</span>
        )}
        <span />

        <span className="ap">Join</span>
        <span className="cx-mesh-cell cx-mesh-join">
          <label className="field cx-mesh-field"><input value={token} placeholder="paste an invitation from another computer" aria-label="Invitation" onChange={(event) => setToken(event.target.value)} /></label>
          <button type="button" className="detent tiny send" data-action="join" disabled={busy !== null || !token.trim()} onClick={() => void run('join', async () => {
            const member = await client.call<Member>('mesh.pair.join', { code: token.trim() });
            setToken('');
            return `Joined as ${member.name}`;
          })}>{busy === 'join' ? 'Joining…' : 'Join'}</button>
        </span>
      </div>

      {others.length > 0 && (
        <div className="cx-mesh-members">
          <div className="cx-mesh-members-head">
            <span className="ap">Computers</span>
            <span className="grow" />
            <span className="ap faint">{byComputer(others).filter((computer) => computer.peers.some((peer) => peer.reachable)).length} of {byComputer(others).length} reachable</span>
          </div>
          <ul className="cx-mesh-list">
            {byComputer(others).map((computer) => {
              const live = computer.peers.filter((peer) => peer.reachable);
              const card = live.find((peer) => peer.card)?.card ?? computer.peers.find((peer) => peer.card)?.card ?? null;
              const apps = [...new Set(live.map((peer) => softwareName(peer.card?.software)).filter(Boolean))];
              const away = computer.peers.length - live.length;
              const models = [...new Set(live.flatMap((peer) => (peer.card?.models ?? []).map((model) => model.variant || model.model_id)))];
              const best = [...live].sort((a, b) => (a.rtt_ms || 9999) - (b.rtt_ms || 9999))[0];
              const server = computer.peers.some((peer) => peer.card?.software === 'cortexmodelserver');
              const detail = live.length && card ? [card.chip, gb(card.memory_bytes), models.join(', ')].filter(Boolean).join(' · ') : 'not reachable now';
              return (
                <li key={computer.key} className={`cx-mesh-row${live.length ? '' : ' away'}${server ? ' server' : ''}`}
                  title={[computer.name, apps.join(', '), detail, away ? `${away} app${away === 1 ? '' : 's'} not reachable` : ''].filter(Boolean).join(' · ')}>
                  <span className={`cx-mesh-pip${live.length ? ' on' : ''}`} aria-hidden />
                  <span className="cx-mesh-row-name">{computer.here ? 'This computer' : computer.name}</span>
                  <span className="ap cx-mesh-row-app">{apps.join(' · ')}{away && live.length ? <span className="faint"> +{away}</span> : null}</span>
                  <span className="cx-mesh-row-detail">{detail}</span>
                  <span className="ap faint cx-mesh-row-where">{best ? [where(best.path), best.rtt_ms ? `${best.rtt_ms} MS` : ''].filter(Boolean).join(' · ') : ''}</span>
                  {computer.here ? <span /> : (
                    <button type="button" className="detent tiny warn cx-mesh-row-remove" data-action="revoke" disabled={busy !== null} aria-label={`Remove ${computer.name}`}
                      onClick={() => void run('revoke', async () => {
                        // Removing one app takes its whole computer out; members of an older app
                        // that the log does not tie to it go one by one.
                        for (const peer of computer.peers) {
                          try { await client.call('mesh.revoke', { node: peer.member.node }); } catch { /* already gone with its computer */ }
                        }
                        return `${computer.name} removed`;
                      })}>Remove</button>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {error && <p className="inline-error">{error}</p>}
    </section>
  );
}

export default definePlugin({ slot: 'settings.mesh@1', component: MeshPanel });
