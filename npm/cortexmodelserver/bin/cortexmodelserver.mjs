#!/usr/bin/env node
// `npx cortexmodelserver`: start CortexModelServer, the model server of the Cortex apps. The first run
// downloads the app for this machine from the official releases (the catalog), checks its checksum
// and unpacks it into ~/.cortexmodelserver/app/<version>/; later runs start the installed app at once,
// without looking at the catalog: the app updates its frame and plugins itself, so a newer app
// release is only for machines that have none. The app runs on its own: closing the terminal does
// not quit it (its start output goes to ~/.cortexmodelserver/logs/start.log). Arguments go to the app
// (`--headless`, `--safe`). Every desktop start also puts a CortexModelServer starter in /Applications
// (Linux: the app menu).
//
// `cortexmodelserver service` keeps it serving models in the background without a window: launchd on
// macOS, systemd on Linux (boot and SSH logout). `service join <code>` installs it, joins the mesh
// and applies the plan; `service allow <computer>` lets a member work here, `service status` says
// how it is, `service off` stops it. `runner` is the same command (CortexTerm's name for it).
// Development uses the same installer via ./cortexmodelserver service.
//
// The day-to-day commands (`status`, `cluster`, `peers`, `recipes`, `plan`, `lane`, `models`,
// `calibrate`) call the running frame's methods, as the window does: serve.*, mesh.peers and
// frame.lanes.list, through the frame's `POST /api/rpc` with its token. `--on <server>` calls another
// cortexmodelserver of the mesh through `mesh.link` (a local tunnel to that computer's frame).
// Node built-ins only: the launcher has no dependencies.

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { accessSync, chmodSync, closeSync, constants, createWriteStream, existsSync, lstatSync, mkdirSync, openSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';

const REPO = process.env.CORTEXMODELSERVER_REPO || 'agentbusiness/CortexModelServer';
// CORTEXMODELSERVER_CATALOG: another catalog (a mirror, or a release under test).
const CATALOG_URL = process.env.CORTEXMODELSERVER_CATALOG || `https://github.com/${REPO}/releases/download/catalog/catalog.json`;
const HOME = resolve(process.env.CORTEX_HOME || join(homedir(), '.cortexmodelserver'));
const DEV_ROOT = process.env.CORTEXMODELSERVER_DEV_ROOT ? resolve(process.env.CORTEXMODELSERVER_DEV_ROOT) : null;
const APPS = join(HOME, 'app');
const READY = '.ready';
const START_LOG = join(HOME, 'logs', 'start.log');
const SERVICE_LOG = join(HOME, 'logs', 'service.log');
const KEEP = 2;
// The models folder every Cortex app shares (shared-models): `<org>-<repo>/<revision>/<file>`.
const MODELS = join(homedir(), '.cortex-models');

const say = (text) => process.stderr.write(`cortexmodelserver: ${text}\n`);
const out = (text = '') => process.stdout.write(`${text}\n`);

class UsageError extends Error {}

function platform() {
  const key = `${process.platform}-${process.arch}`;
  const known = { 'darwin-arm64': 'darwin-arm64', 'linux-x64': 'linux-x64', 'linux-arm64': 'linux-arm64' };
  if (!known[key]) throw new Error(`CortexModelServer runs on Apple Silicon Macs and on Linux (x64, arm64), not on ${key}`);
  return known[key];
}

/** `0.3.7` as a comparable triple, or null. */
function version(text) {
  const match = /^v?(\d+)\.(\d+)\.(\d+)(\+.*)?$/.exec(String(text ?? '').trim());
  return match ? [Number(match[1]), Number(match[2]), Number(match[3])] : null;
}

function compare(a, b) {
  for (let i = 0; i < 3; i += 1) if (a[i] !== b[i]) return a[i] - b[i];
  return 0;
}

/** The response to a GET of `url`, after redirects (GitHub sends release files from another host). */
function get(url, redirects = 5) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith('https:') ? https : http;
    const request = client.get(url, { headers: { 'user-agent': 'cortexmodelserver-npx' } }, (response) => {
      const status = response.statusCode ?? 0;
      if (status >= 300 && status < 400 && response.headers.location && redirects > 0) {
        response.resume();
        resolve(get(new URL(response.headers.location, url).toString(), redirects - 1));
        return;
      }
      if (status !== 200) {
        response.resume();
        reject(new Error(`${url} answered ${status}`));
        return;
      }
      resolve(response);
    });
    request.on('error', reject);
  });
}

/** The release catalog: the current version of the app and every part, with their files. */
async function catalog() {
  const response = await fetch(CATALOG_URL, { headers: { 'user-agent': 'cortexmodelserver-npx' } });
  if (!response.ok) throw new Error(`${CATALOG_URL} answered ${response.status}`);
  const parsed = await response.json();
  if (parsed.format !== 'cortexmodelserver-catalog/1') throw new Error(`unknown catalog format ${parsed.format}`);
  return parsed;
}

/** Installed app versions, newest first. */
function installed() {
  if (!existsSync(APPS)) return [];
  return readdirSync(APPS)
    .filter((name) => version(name) && existsSync(join(APPS, name, READY)))
    .sort((a, b) => compare(version(b), version(a)));
}

function executable(dir) {
  return process.platform === 'darwin' ? join(dir, 'CortexModelServer.app', 'Contents', 'MacOS', 'CortexModelServer') : join(dir, 'cortexmodelserver', 'CortexModelServer');
}

/** Download the app archive, check size and SHA-256, unpack it beside the others. */
async function install(app, asset) {
  mkdirSync(APPS, { recursive: true });
  const archive = join(APPS, `.download-${app.version}.tar.gz`);
  const staging = join(APPS, `.staging-${app.version}`);
  say(`downloading CortexModelServer ${app.version} (${Math.round(asset.size / 1e6)} MB)`);
  // node:http rather than fetch: fetch's stream fails when a server closes the connection while the
  // file write is behind.
  const body = await get(asset.url);
  const hash = createHash('sha256');
  let received = 0;
  let reported = 0;
  body.on('data', (chunk) => {
    hash.update(chunk);
    received += chunk.length;
    const percent = asset.size ? Math.floor((received * 100) / asset.size) : 0;
    if (percent >= reported + 10) {
      reported = percent;
      process.stderr.write(`\rcortexmodelserver: ${percent}%`);
    }
  });
  await pipeline(body, createWriteStream(archive));
  process.stderr.write('\n');
  if (received !== asset.size || hash.digest('hex') !== String(asset.sha256).toLowerCase()) {
    rmSync(archive, { force: true });
    throw new Error(`${asset.name} does not match the catalog (size or checksum); it was not used`);
  }
  rmSync(staging, { recursive: true, force: true });
  mkdirSync(staging, { recursive: true });
  await run('tar', ['-xzf', archive, '-C', staging]);
  rmSync(archive, { force: true });
  const target = join(APPS, app.version);
  rmSync(target, { recursive: true, force: true });
  renameSync(staging, target);
  writeFileSync(join(target, READY), '');
  say(`CortexModelServer ${app.version} is installed in ${target}`);
}

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'ignore', 'inherit'] });
    child.on('error', reject);
    child.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`${command} failed (${code})`))));
  });
}

/** `<home>/run/frame.json` of a frame running for this home (pid, port; the token on Linux), or null. */
function frameInfo() {
  try {
    const info = JSON.parse(readFileSync(join(HOME, 'run', 'frame.json'), 'utf8'));
    process.kill(info.pid, 0);
    return info;
  } catch {
    return null;
  }
}

/** The pid of a CortexModelServer running for this home, if any. */
function running() {
  return frameInfo()?.pid ?? null;
}

/** Keep the newest versions (and the one starting now). */
function prune(keep) {
  installed()
    .slice(KEEP)
    .filter((name) => name !== keep)
    .forEach((name) => rmSync(join(APPS, name), { recursive: true, force: true }));
}

/** The installed app to start (downloaded first on a machine that has none), or null. */
async function ensureApp() {
  const key = platform();
  // The installed app updates itself (frame and plugins): only a machine without one downloads.
  if (installed().length === 0) {
    try {
      const listed = await catalog();
      const files = listed.app?.files?.[key];
      if (listed.app && files?.archive) await install(listed.app, files.archive);
      else say(`no CortexModelServer app is released for ${key} yet`);
    } catch (error) {
      say(`cannot download CortexModelServer (${error.message})`);
    }
  }
  const chosen = installed()[0];
  if (!chosen) {
    say('CortexModelServer is not installed and could not be downloaded');
    return null;
  }
  prune(chosen);
  return chosen;
}

/** The installed frame for the service; an installed app whose frame does not know `--service`
 * yet is replaced by the current release before the service is installed. */
async function serviceApp() {
  let chosen = await ensureApp();
  if (!chosen) return null;
  let app = executable(join(APPS, chosen));
  if ((await serviceCommand(app, ['--help'], true)).stdout.includes('--service')) return app;
  say('the installed app predates service mode; downloading the current app');
  const listed = await catalog();
  const asset = listed.app?.files?.[platform()]?.archive;
  if (!asset) throw new Error('the release catalog has no app with service mode for this machine yet');
  await install(listed.app, asset);
  chosen = listed.app.version;
  app = executable(join(APPS, chosen));
  if (!(await serviceCommand(app, ['--help'], true)).stdout.includes('--service')) {
    throw new Error('the released app does not support --service yet');
  }
  return app;
}

// ─── the background service ─────────────────────────────────────────────────────────────────────
// The service starts the installed frame directly, independently of npx's temporary cache, with
// `--service`: no window, the updater on, and its engines serving lanes as in the window.
const LABEL = 'com.cortexmodelserver.service';
const AGENT = join(homedir(), 'Library', 'LaunchAgents', `${LABEL}.plist`);
const UNIT_NAME = 'cortexmodelserver.service';
const SYSTEM_SERVICE = process.platform === 'linux' && process.getuid() === 0;
const UNIT_DIR = SYSTEM_SERVICE ? '/etc/systemd/system' : join(process.env.XDG_CONFIG_HOME || join(homedir(), '.config'), 'systemd', 'user');
const UNIT = join(UNIT_DIR, UNIT_NAME);
// What earlier builds of this launcher installed (CortexTerm's runner, renamed): taken out when the
// service is installed or turned off.
const OLD_LABEL = 'com.cortexmodelserver.runner';
const OLD_AGENT = join(homedir(), 'Library', 'LaunchAgents', `${OLD_LABEL}.plist`);
const OLD_UNIT_NAME = 'cortexmodelserver-runner.service';
const OLD_UNIT = join(UNIT_DIR, OLD_UNIT_NAME);
const xml = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const journal = () => `journalctl ${SYSTEM_SERVICE ? '' : '--user '}-u ${UNIT_NAME}`;

function serviceEnv() {
  const env = { CORTEX_HOME: HOME, HOME: homedir(), PATH: process.env.PATH || '/usr/local/bin:/usr/bin:/bin' };
  if (DEV_ROOT) { env.CORTEXMODELSERVER_DEV_ROOT = DEV_ROOT; env.CORTEX_NO_HANDOVER = '1'; }
  if (process.env.SHELL) env.SHELL = process.env.SHELL;
  // Explicit release mirrors (offline or private installations), the log level, and the GPUs the
  // engines may use, as they were set for the installer.
  for (const key of ['CORTEXMODELSERVER_REPO', 'CORTEXMODELSERVER_CATALOG', 'CORTEX_LOG', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH']) {
    if (process.env[key]) env[key] = process.env[key];
  }
  return env;
}

function agentPlist(app) {
  const env = Object.entries(serviceEnv()).map(([key, value]) => `<key>${xml(key)}</key><string>${xml(value)}</string>`).join('');
  const log = xml(SERVICE_LOG);
  // Interactive: launchd does not throttle the engines as background work. The process group is
  // not abandoned: an engine left behind by a frame that died goes with it.
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>${LABEL}</string>
<key>ProgramArguments</key><array><string>${xml(app)}</string><string>--service</string></array>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><true/>
<key>ThrottleInterval</key><integer>5</integer>
<key>ProcessType</key><string>Interactive</string>
<key>ExitTimeOut</key><integer>120</integer>
<key>StandardOutPath</key><string>${log}</string>
<key>StandardErrorPath</key><string>${log}</string>
<key>EnvironmentVariables</key><dict>${env}</dict>
</dict></plist>
`;
}

// systemd interprets percent specifiers and, in ExecStart, dollar expansion even inside quotes.
function unitQuote(value, command = false) {
  const text = String(value);
  if (/[\0\r\n]/.test(text)) throw new Error('service paths and environment values cannot contain line breaks or NUL');
  return '"' + text.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/%/g, '%%').replace(/\$/g, command ? '$$$$' : '$$') + '"';
}

function serviceUnit(app) {
  const env = Object.entries(serviceEnv()).map(([key, value]) => `Environment=${unitQuote(`${key}=${value}`)}`).join('\n');
  return `[Unit]
Description=CortexModelServer model server
${SYSTEM_SERVICE ? 'After=network-online.target\nWants=network-online.target\n' : ''}StartLimitIntervalSec=0

[Service]
Type=simple
ExecStart=${unitQuote(app, true)} --service
${env}
Restart=always
RestartSec=5
# The frame drains its lanes and stops its engines; what is left when it exits (an engine of a
# frame that died) is killed with it, so no stray engine holds GPU memory across a restart.
KillMode=mixed
TimeoutStopSec=120
UMask=0077

[Install]
WantedBy=${SYSTEM_SERVICE ? 'multi-user.target' : 'default.target'}
`;
}

function serviceCommand(command, args, capture = false) {
  return new Promise((resolve) => {
    let stdout = '';
    let stderr = '';
    const env = { ...process.env };
    if (process.platform === 'linux' && process.getuid() !== 0 && command === 'systemctl') {
      env.XDG_RUNTIME_DIR ||= `/run/user/${process.getuid()}`;
      env.DBUS_SESSION_BUS_ADDRESS ||= `unix:path=${env.XDG_RUNTIME_DIR}/bus`;
    }
    const child = spawn(command, args, { env, stdio: ['ignore', capture ? 'pipe' : 'ignore', 'pipe'] });
    child.stdout?.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', (error) => resolve({ ok: false, stdout, stderr: error.message }));
    child.on('exit', (code) => resolve({ ok: code === 0, stdout: stdout.trim(), stderr: stderr.trim() }));
  });
}

const systemctl = (args, capture = false) => serviceCommand('systemctl', [...(SYSTEM_SERVICE ? [] : ['--user']), '--no-ask-password', ...args], capture);

async function requireService(result, action) {
  const outcome = await result;
  if (!outcome.ok) throw new Error(`${action}: ${outcome.stderr || 'service manager unavailable'}`);
  return outcome;
}

/** `/health` of the frame running for this home, or null. */
async function health(info = frameInfo(), timeout = 1000) {
  if (!info) return null;
  try {
    const response = await fetch(`http://127.0.0.1:${info.port}/health`, { signal: AbortSignal.timeout(timeout) });
    return await response.json();
  } catch {
    return null;
  }
}

async function waitForService() {
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    const info = frameInfo();
    // The frame reports service mode as `runner` (the name its health kept).
    const state = await health(info, Math.max(1, Math.min(1000, deadline - Date.now())));
    if (state?.ok && state.runner && state.pid === info.pid) return info.pid;
    await new Promise((resolve) => setTimeout(resolve, Math.min(500, Math.max(0, deadline - Date.now()))));
  }
  throw new Error(`the service did not become ready in 60 seconds; see ${process.platform === 'darwin' ? SERVICE_LOG : journal()}`);
}

/** Stop and remove one launchd job / systemd unit (the service, or what an earlier launcher made). */
async function removeService(label, agent, unitName, unit) {
  if (process.platform === 'darwin') {
    const job = `gui/${process.getuid()}/${label}`;
    if ((await serviceCommand('/bin/launchctl', ['print', job])).ok) {
      await requireService(serviceCommand('/bin/launchctl', ['bootout', job]), 'cannot stop the service');
    }
    rmSync(agent, { force: true });
    return;
  }
  const present = existsSync(unit);
  const active = await systemctl(['is-active', '--quiet', unitName]);
  if (present || active.ok) {
    await requireService(systemctl(['stop', unitName]), 'cannot stop the service');
    if (present) await requireService(systemctl(['disable', unitName]), 'cannot disable the service');
    rmSync(unit, { force: true });
    await requireService(systemctl(['daemon-reload']), 'cannot reload services');
  }
}

function needServicePlatform() {
  if (!['darwin', 'linux'].includes(process.platform)) throw new Error('the service supports macOS and Linux');
}

/** Install the service and start it; resolves to its pid. */
async function serviceOn() {
  needServicePlatform();
  const current = await health();
  if (current?.ok && current.runner) {
    say(`the service already runs (pid ${current.pid})`);
    return current.pid;
  }
  if (running()) {
    // Do not take over a desktop or another checkout sharing this home.
    throw new Error('CortexModelServer already runs for this home (not as the service); quit it before installing the service');
  }
  let app;
  if (DEV_ROOT) {
    if (!existsSync(join(DEV_ROOT, 'cortexmodelserver'))) throw new Error(`no development launcher in ${DEV_ROOT}`);
    await run(join(DEV_ROOT, 'cortexmodelserver'), ['--build-only']);
    app = process.platform === 'darwin'
      ? join(DEV_ROOT, 'build', 'CortexModelServer.app', 'Contents', 'MacOS', 'CortexModelServer')
      : join(DEV_ROOT, 'build', 'cortexmodelserver', 'CortexModelServer');
  } else {
    app = await serviceApp();
    if (!app) throw new Error('CortexModelServer is not installed and could not be downloaded');
  }
  await removeService(OLD_LABEL, OLD_AGENT, OLD_UNIT_NAME, OLD_UNIT);
  if (process.platform === 'linux' && !SYSTEM_SERVICE) {
    // Linger keeps the user manager alive after SSH logout and starts it at boot.
    const linger = await serviceCommand('loginctl', ['show-user', String(process.getuid()), '--property=Linger', '--value'], true);
    if (linger.stdout !== 'yes') {
      await requireService(serviceCommand('loginctl', ['--no-ask-password', 'enable-linger', String(process.getuid())]),
        `cannot keep the service alive after logout; an administrator must run loginctl enable-linger ${process.getuid()}`);
    }
  }
  mkdirSync(join(HOME, 'logs'), { recursive: true, mode: 0o700 });
  if (process.platform === 'darwin') {
    mkdirSync(dirname(AGENT), { recursive: true });
    writeFileSync(AGENT, agentPlist(app));
    const domain = `gui/${process.getuid()}`;
    await serviceCommand('/bin/launchctl', ['bootout', `${domain}/${LABEL}`]);
    await requireService(serviceCommand('/bin/launchctl', ['bootstrap', domain, AGENT]), 'cannot start the service');
  } else {
    mkdirSync(UNIT_DIR, { recursive: true });
    writeFileSync(UNIT, serviceUnit(app));
    await requireService(systemctl(['daemon-reload']), 'cannot load the service');
    await requireService(systemctl(['enable', UNIT_NAME]), 'cannot enable the service');
    await requireService(systemctl(['restart', UNIT_NAME]), 'cannot start the service');
  }
  const pid = await waitForService();
  say(`the ${DEV_ROOT ? 'development ' : ''}service is on (pid ${pid}): it serves its lanes, updates itself and restarts on its own`);
  say(process.platform === 'darwin' ? 'starts at login; you can close this terminal' : 'starts at boot and stays running after SSH logout');
  return pid;
}

async function serviceOff() {
  needServicePlatform();
  await removeService(LABEL, AGENT, UNIT_NAME, UNIT);
  await removeService(OLD_LABEL, OLD_AGENT, OLD_UNIT_NAME, OLD_UNIT);
  say('the service is off: its lanes stopped; models, lanes and the mesh membership are kept');
  return 0;
}

/** Mac names carry a curly apostrophe (Sam’s Mac); a typed straight one names the same computer. */
const plain = (text) => String(text ?? '').replace(/[\u2018\u2019]/g, "'").trim().toLowerCase();

async function service(args, options) {
  const [what = 'on', ...rest] = args;
  if (what === 'on' || what === 'install') {
    await serviceOn();
    return 0;
  }
  if (what === 'off') return serviceOff();
  if (what === 'status') return serviceStatus(options);
  if (what === 'join') {
    const code = rest.join(' ').trim();
    if (!code) throw new UsageError('service join <code>: the four words (or the invitation) from Settings > Mesh on another computer');
    const current = await health();
    if (current?.ok && !current.runner) say('CortexModelServer runs here with its window; joining through it instead of installing the service');
    else await serviceOn();
    const target = await localTarget();
    const member = await ready(() => call(target, 'mesh.pair.join', { code }));
    say(`joined the network as ${member.name}`);
    // The plan runs by itself: lanes from this computer's GPUs and what the Cortex apps here use.
    const view = await ready(() => call(target, 'serve.lanes.list', {}));
    say(`serving: ${view.lanes.length} lane${view.lanes.length === 1 ? '' : 's'}`);
    if (view.lanes.length) printLanes(view.lanes);
    say('every computer of the mesh may now work on this one');
    return 0;
  }
  if (what === 'allow') {
    // The service has no Settings window: this is Settings > Mesh's "may work here" switch.
    const wanted = rest.join(' ').trim();
    if (!wanted) throw new UsageError('service allow <computer>: the name of a computer in your network');
    const target = await localTarget();
    const { members } = await call(target, 'mesh.members', {});
    const status = await call(target, 'mesh.status', {});
    const others = members.filter((member) => member.node !== status.node);
    const named = others.filter((member) => plain(member.name) === plain(wanted) || member.node.startsWith(wanted));
    if (!named.length) throw new Error(`no computer named ${wanted} in the network; members: ${others.map((member) => member.name).join(', ') || 'none'}`);
    for (const member of named) await call(target, 'mesh.allow', { node: member.node, allowed: true });
    say(`${named[0].name} may now work on this computer`);
    return 0;
  }
  throw new UsageError(`unknown service command ${what}; use service, service join <code>, service allow <computer>, service status or service off`);
}

async function serviceStatus(options) {
  needServicePlatform();
  let manager;
  if (process.platform === 'darwin') {
    const job = await serviceCommand('/bin/launchctl', ['print', `gui/${process.getuid()}/${LABEL}`], true);
    const state = /^\s*state = (.+)$/m.exec(job.stdout)?.[1];
    manager = existsSync(AGENT) ? `LaunchAgent ${LABEL}: ${job.ok ? state || 'loaded' : 'not loaded'}` : `no LaunchAgent (${AGENT})`;
  } else {
    const active = await systemctl(['is-active', UNIT_NAME], true);
    const enabled = await systemctl(['is-enabled', UNIT_NAME], true);
    manager = existsSync(UNIT) ? `${SYSTEM_SERVICE ? 'system' : 'user'} unit ${UNIT_NAME}: ${active.stdout || 'unknown'}, ${enabled.stdout || 'not enabled'}` : `no systemd unit (${UNIT})`;
  }
  const info = frameInfo();
  const state = await health(info);
  const result = { service: manager, frame: state, mesh: null, serve: null };
  if (state?.ok) {
    try {
      const target = await localTarget();
      result.mesh = await call(target, 'mesh.status', {}).catch(() => null);
      result.serve = await call(target, 'serve.status.get', {}).catch(() => null);
    } catch (error) {
      result.error = error.message;
    }
  }
  if (options.json) return json(result);
  out(manager);
  if (!state?.ok) {
    out('frame: not running for this home');
  } else {
    out(`frame: ${state.version} (pid ${state.pid}), ${state.runner ? 'service mode' : state.headless ? 'headless' : 'with its window'}, up ${duration(state.uptime_ms)}`);
  }
  if (result.mesh) out(`mesh: ${result.mesh.name}, ${result.mesh.reachable}/${result.mesh.members} member${result.mesh.members === 1 ? '' : 's'} reachable`);
  if (result.serve) {
    const totals = result.serve.totals ?? {};
    out(`serving: ${result.serve.lanes.length} lane${result.serve.lanes.length === 1 ? '' : 's'}, ${num(totals.tok_s)} tok/s, slots ${totals.slots_busy ?? 0}/${totals.slots ?? 0}`);
  }
  out(`log: ${process.platform === 'darwin' ? SERVICE_LOG : journal()}`);
  return 0;
}

// ─── the frame's API ────────────────────────────────────────────────────────────────────────────
// `POST /api/rpc` with the token (the same envelope routing as a WebSocket client, one call per
// request; stream items come back together). It needs no WebSocket library, and it goes through
// `mesh.link`'s tunnel the same way, since the tunnel carries any TCP connection to the frame's port.

/** The frame's API token: CORTEX_FRAME_TOKEN, else `run/frame.json` (Linux), else the macOS
 * Keychain (frame/src/keychain.rs: service "CortexModelServer frame token", account the home). */
function frameToken(info) {
  if (process.env.CORTEX_FRAME_TOKEN) return Promise.resolve(process.env.CORTEX_FRAME_TOKEN);
  if (info.token) return Promise.resolve(info.token);
  if (process.platform !== 'darwin') return Promise.reject(new Error(`${join(HOME, 'run', 'frame.json')} has no token`));
  return new Promise((resolve, reject) => {
    let text = '';
    const child = spawn('/usr/bin/security', ['find-generic-password', '-s', 'CortexModelServer frame token', '-a', HOME, '-w'], { stdio: ['ignore', 'pipe', 'ignore'] });
    child.stdout.on('data', (chunk) => { text += chunk; });
    child.on('error', reject);
    child.on('exit', (code) => (code === 0 ? resolve(text.trim()) : reject(new Error('the frame token is not in the Keychain'))));
  });
}

/** This computer's frame. */
async function localTarget() {
  const info = frameInfo();
  if (!info) throw new Error('CortexModelServer is not running here: start it (cortexmodelserver) or install the service (cortexmodelserver service)');
  return { base: `http://127.0.0.1:${info.port}`, token: await frameToken(info), name: null };
}

class CallError extends Error {
  constructor(method, error) {
    super(error?.message ?? JSON.stringify(error));
    this.method = method;
    this.code = error?.code ?? 'error';
    this.retryable = Boolean(error?.retryable);
  }
}

/** One call on a frame. No timeout: a calibration or a plan that loads models takes minutes. */
function call(target, method, params = {}) {
  const body = Buffer.from(JSON.stringify({ method, params }));
  const url = new URL('/api/rpc', target.base);
  return new Promise((resolve, reject) => {
    const request = http.request(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'content-length': body.length, authorization: `Bearer ${target.token}` },
    }, (response) => {
      const chunks = [];
      response.on('data', (chunk) => chunks.push(chunk));
      response.on('error', reject);
      response.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        if (response.statusCode === 401) {
          reject(new Error(`${target.name ?? 'the frame'} refused the token (it may have restarted); try again`));
          return;
        }
        let answer;
        try {
          answer = JSON.parse(text);
        } catch {
          reject(new Error(`${method}: the frame answered ${response.statusCode} ${text.slice(0, 200)}`));
          return;
        }
        if (answer.ok) resolve(answer.result);
        else reject(new CallError(method, answer.error));
      });
    });
    request.on('error', (error) => reject(new Error(`cannot reach ${target.name ?? 'the frame'} (${error.message})`)));
    request.end(body);
  });
}

/** `fn` again while the plugin that answers it is still starting (a service that just started). */
async function ready(fn, ms = 120_000) {
  const deadline = Date.now() + ms;
  for (;;) {
    try {
      return await fn();
    } catch (error) {
      const starting = error instanceof CallError && (error.retryable || ['unavailable', 'unknown_method', 'not_ready', 'starting'].includes(error.code));
      if (!starting || Date.now() > deadline) throw error;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
}

/** The member `wanted` names (its name, as Settings > Mesh shows it, or the start of its node key). */
async function findMember(local, wanted) {
  const { peers } = await call(local, 'mesh.peers', {});
  const exact = peers.filter((peer) => plain(peer.member.name) === plain(wanted) || (peer.card && plain(peer.card.name) === plain(wanted)));
  const found = exact.length ? exact : peers.filter((peer) => wanted.length >= 4 && peer.member.node.startsWith(wanted.toLowerCase()));
  if (found.length === 1) return found[0];
  if (found.length > 1) throw new Error(`${wanted} names ${found.length} computers (${found.map((peer) => `${peer.member.name} ${peer.member.node.slice(0, 8)}`).join(', ')}); give the start of its key`);
  const names = peers.map((peer) => peer.member.name).join(', ');
  throw new Error(`no server named ${wanted} in the mesh${names ? `; members: ${names}` : ' (this computer is on no network)'}`);
}

/** The frame a command talks to: this computer's, or with `--on` a member's, through `mesh.link`
 * (a local address that leads to its frame, and the token it takes), as the window's runtimeFor. */
async function target(options) {
  const local = await localTarget();
  if (!options.on) return local;
  const status = await call(local, 'mesh.status', {}).catch(() => null);
  if (status && plain(status.name) === plain(options.on)) return local;
  const peer = await findMember(local, options.on);
  if (!peer.reachable) throw new Error(`${peer.member.name} is not reachable now`);
  const link = await call(local, 'mesh.link', { node: peer.member.node });
  return { base: link.url.replace(/\/+$/, ''), token: link.token, name: peer.member.name };
}

// ─── formatting ─────────────────────────────────────────────────────────────────────────────────

function json(value) {
  out(JSON.stringify(value, null, 2));
  return 0;
}

function num(value, digits = 1) {
  const n = Number(value ?? 0);
  return Number.isFinite(n) ? n.toFixed(digits) : '-';
}

function bytes(value) {
  const n = Number(value ?? 0);
  if (!n) return '-';
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)} TB`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.round(n / 1e3)} kB`;
}

function count(value) {
  const n = Number(value ?? 0);
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(0)}k`;
  return String(n);
}

/** 32768 -> 32k. */
function ctx(value) {
  const n = Number(value ?? 0);
  if (!n) return '-';
  if (n % 1048576 === 0) return `${n / 1048576}m`;
  if (n % 1024 === 0) return `${n / 1024}k`;
  return String(n);
}

const percent = (share) => (share === undefined || share === null ? '-' : `${Math.round(Number(share) * 100)}%`);

function duration(ms) {
  const s = Math.floor(Number(ms ?? 0) / 1000);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}

/** A plain table: `columns` are [header, row -> text]; numbers right-aligned. */
function table(rows, columns) {
  if (!rows.length) return;
  const cells = rows.map((row) => columns.map(([, cell]) => String(cell(row) ?? '-')));
  const heads = columns.map(([head]) => head);
  const widths = heads.map((head, i) => Math.max(head.length, ...cells.map((line) => line[i].length)));
  const right = heads.map((_, i) => cells.every((line) => /^[-+]?[\d.,]+\s?(%|[kKMGT]B?|ms|W|°C|m|k|G|M)?$|^-$/.test(line[i])));
  const line = (values) => values.map((value, i) => (right[i] ? value.padStart(widths[i]) : value.padEnd(widths[i]))).join('  ').trimEnd();
  out(line(heads));
  for (const values of cells) out(line(values));
}

function printGpus(gpus) {
  table(gpus, [
    ['GPU', (gpu) => gpu.index],
    ['NAME', (gpu) => gpu.name],
    ['BACKEND', (gpu) => gpu.backend + (gpu.unified ? ' (unified)' : '')],
    ['MEMORY', (gpu) => bytes(gpu.memory_bytes)],
    ['USED', (gpu) => bytes(gpu.used_bytes)],
    ['UTIL', (gpu) => percent(gpu.util)],
    ['POWER', (gpu) => (gpu.power_w ? `${Math.round(gpu.power_w)} W` : '-')],
    ['TEMP', (gpu) => (gpu.temp_c ? `${Math.round(gpu.temp_c)}°C` : '-')],
  ]);
}

function printLanes(lanes, instances = new Map()) {
  table(lanes, [
    ['LANE', (lane) => lane.settings.id],
    ['MODEL', (lane) => lane.name],
    ['ENGINE', (lane) => lane.engine],
    ['KIND', (lane) => lane.kind],
    ['GPU', (lane) => lane.settings.gpu],
    ['STATE', (lane) => lane.state + (lane.loaded ? '' : lane.state === 'active' ? ' (unloaded)' : '') + (lane.error ? `: ${lane.error}` : '')],
    ['SLOTS', (lane) => `${lane.slots_busy}/${lane.slots}${lane.queued ? ` +${lane.queued}` : ''}`],
    ['CTX', (lane) => ctx(lane.ctx || lane.settings.ctx)],
    ['TOK/S', (lane) => num(lane.tok_s)],
    ['TOKENS/H', (lane) => count(lane.tokens_hour)],
    ['MEMORY', (lane) => bytes(lane.memory_bytes)],
    ['SET', (lane) => [lane.settings.manual ? 'by hand' : 'plan', lane.settings.pinned ? 'pinned' : '', lane.settings.lend ? '' : 'not lent'].filter(Boolean).join(',')],
    ...(instances.size ? [['PID', (lane) => instances.get(lane.settings.id)?.pid || '-'], ['RESTARTS', (lane) => instances.get(lane.settings.id)?.restarts ?? '-']] : []),
  ]);
}

function printTotals(totals) {
  const t = totals ?? {};
  out(`  throughput  ${num(t.tok_s)} tok/s generated, ${num(t.prefill_tok_s, 0)} tok/s prefill, first token ${t.ttft_ms ? `${Math.round(t.ttft_ms)} ms` : '-'}`);
  out(`  load        slots ${t.slots_busy ?? 0}/${t.slots ?? 0} busy, ${t.queued ?? 0} queued, ${num(t.requests_min)} requests/min`);
  out(`  tokens      ${count(t.tokens_hour)} last hour, ${count(t.tokens_day)} last day`);
  out(`  lanes       ${t.lanes_serving ?? 0} serving, ${t.lanes_loading ?? 0} loading, ${t.lanes_failed ?? 0} failed`);
  out(`  gpu         ${percent(t.gpu_util)} busy, ${bytes(t.gpu_used_bytes)} of ${bytes(t.gpu_bytes)}${t.power_w ? `, ${Math.round(t.power_w)} W` : ''}`);
}

// ─── arguments ──────────────────────────────────────────────────────────────────────────────────

const VALUE_FLAGS = new Set(['on', 'gpu', 'slots', 'ctx', 'cache', 'kv', 'speculative', 'idle', 'tags', 'recipe', 'checkpoints']);
const BOOL_FLAGS = new Set(['json', 'reset', 'all', 'pin', 'unpin', 'lend', 'no-lend', 'yes', 'local']);

/** `[words, options]`: `--name value`, `--name=value` and boolean flags. */
function parse(args) {
  const words = [];
  const options = {};
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--') { words.push(...args.slice(i + 1)); break; }
    const match = /^--([a-z-]+)(?:=(.*))?$/.exec(arg);
    if (!match) { words.push(arg); continue; }
    const [, name, inline] = match;
    if (VALUE_FLAGS.has(name)) {
      const value = inline ?? args[++i];
      if (value === undefined) throw new UsageError(`--${name} needs a value`);
      options[name] = value;
    } else if (BOOL_FLAGS.has(name)) {
      options[name] = true;
    } else {
      throw new UsageError(`unknown option --${name}`);
    }
  }
  return [words, options];
}

function integer(name, text, min = 0) {
  if (!/^\d+$/.test(String(text))) throw new UsageError(`--${name} takes a whole number, not ${text}`);
  const n = Number(text);
  if (n < min) throw new UsageError(`--${name} is at least ${min}`);
  return n;
}

/** `32k` -> 32768, `1m` -> 1048576, `8192` -> 8192 (tokens). */
function parseCtx(text) {
  const match = /^(\d+(?:\.\d+)?)\s*([kKmM]?)$/.exec(String(text).trim());
  if (!match) throw new UsageError(`--ctx takes tokens such as 32k, 128k or 8192, not ${text}`);
  const scale = { '': 1, k: 1024, m: 1048576 }[match[2].toLowerCase()];
  return Math.round(Number(match[1]) * scale);
}

/** Host-RAM prompt cache, MiB: off 0, small 1024, large 6144, a number; `default`: the recipe's. */
function parseCache(text) {
  const presets = { off: 0, none: 0, small: 1024, large: 6144 };
  const key = String(text).trim().toLowerCase();
  if (key === 'default') return null;
  if (key in presets) return presets[key];
  if (/^\d+$/.test(key)) return Number(key);
  const match = /^(\d+(?:\.\d+)?)\s*(g|gb|gib|m|mb|mib)$/.exec(key);
  if (match) return Math.round(Number(match[1]) * (match[2].startsWith('g') ? 1024 : 1));
  throw new UsageError(`--cache takes off, small, large, default or MiB (2048, 4g), not ${text}`);
}

function onOff(name, text) {
  const key = String(text).toLowerCase();
  if (['on', 'yes', 'true', '1'].includes(key)) return true;
  if (['off', 'no', 'false', '0'].includes(key)) return false;
  if (key === 'default') return null;
  throw new UsageError(`--${name} takes on, off or default`);
}

/** The lane settings the options name, over `base` (an edit keeps what is not named). */
function laneSettings(options, base) {
  const settings = { ...base };
  const set = (key, value) => { if (value === null) delete settings[key]; else settings[key] = value; };
  if (options.gpu !== undefined) settings.gpu = integer('gpu', options.gpu);
  if (options.slots !== undefined) settings.slots = integer('slots', options.slots);
  if (options.ctx !== undefined) settings.ctx = options.ctx === 'default' ? 0 : parseCtx(options.ctx);
  if (options.cache !== undefined) set('cache_ram_mib', parseCache(options.cache));
  if (options.checkpoints !== undefined) set('checkpoints', options.checkpoints === 'default' ? null : integer('checkpoints', options.checkpoints));
  if (options.kv !== undefined) set('kv_type', options.kv === 'default' ? null : options.kv);
  if (options.speculative !== undefined) set('speculative', onOff('speculative', options.speculative));
  if (options.idle !== undefined) set('idle_unload_s', options.idle === 'default' ? null : integer('idle', options.idle));
  if (options.pin) settings.pinned = true;
  if (options.unpin) settings.pinned = false;
  if (options.lend) settings.lend = true;
  if (options['no-lend']) settings.lend = false;
  if (options.tags !== undefined) settings.tags = options.tags.split(',').map((tag) => tag.trim()).filter(Boolean);
  return settings;
}

/** The recipe `wanted` names: its id, its id with spaces for dashes ("qwen3.8-27b uzu"), its
 * name, or the one recipe whose id, name and engine hold every word. */
function findRecipe(recipes, wanted) {
  const text = plain(wanted);
  const slug = text.replace(/\s+/g, '-');
  const exact = recipes.filter((recipe) => recipe.id === text || recipe.id === slug || plain(recipe.name) === text);
  const available = (list) => (list.length > 1 && list.some((recipe) => recipe.available) ? list.filter((recipe) => recipe.available) : list);
  let found = available(exact);
  if (!found.length) {
    const words = text.split(/\s+/).filter(Boolean);
    found = available(recipes.filter((recipe) => {
      const hay = plain(`${recipe.id} ${recipe.name} ${recipe.engine} ${recipe.engine.replace(/^model-/, '')}`);
      return words.every((word) => hay.includes(word));
    }));
  }
  if (found.length === 1) return found[0];
  if (found.length > 1) throw new Error(`${wanted} names ${found.length} recipes: ${found.map((recipe) => recipe.id).join(', ')}`);
  throw new Error(`no recipe ${wanted}; see cortexmodelserver recipes`);
}

/** The lane `wanted` names: its id, or the one lane whose id starts with it. */
function findLane(lanes, wanted) {
  const exact = lanes.find((lane) => lane.settings.id === wanted);
  if (exact) return exact;
  const found = lanes.filter((lane) => lane.settings.id.startsWith(wanted));
  if (found.length === 1) return found[0];
  if (found.length > 1) throw new Error(`${wanted} names ${found.length} lanes: ${found.map((lane) => lane.settings.id).join(', ')}`);
  throw new Error(`no lane ${wanted}; lanes: ${lanes.map((lane) => lane.settings.id).join(', ') || 'none'}`);
}

const where = (t) => (t.name ? ` on ${t.name}` : '');

// ─── the day-to-day commands ────────────────────────────────────────────────────────────────────

async function status(words, options) {
  const t = await target(options);
  const serve = await call(t, 'serve.status.get', {});
  const frame = await call(t, 'frame.lanes.list', {}).catch(() => ({ lanes: [] }));
  if (options.json) return json(serve);
  const totals = serve.totals ?? {};
  out(`${serve.name} (${serve.platform}) · ${num(totals.tok_s)} tok/s · slots ${totals.slots_busy ?? 0}/${totals.slots ?? 0} · ${totals.queued ?? 0} queued`);
  out();
  if (serve.gpus.length) printGpus(serve.gpus);
  else out('no GPU found');
  out();
  if (serve.lanes.length) printLanes(serve.lanes, new Map(frame.lanes.map((lane) => [lane.id, lane])));
  else out('no lanes yet (the plan adds them as the GPUs allow; or lane add <recipe>)');
  if (serve.engines.length) {
    out();
    out(`engines: ${serve.engines.map((engine) => `${engine.id} ${engine.version}${engine.enabled ? '' : ' (off)'}`).join(', ')}`);
  }
  if (serve.wants.length) {
    out();
    table(serve.wants, [
      ['APP', (want) => want.app],
      ['USES', (want) => want.model_id],
      ['TAGS', (want) => want.tags.join(',')],
      ['SERVED BY', (want) => want.recipe || 'nothing here'],
    ]);
  }
  return 0;
}

async function cluster(words, options) {
  const t = await target(options);
  const result = await call(t, 'serve.cluster.get', {});
  if (options.json) return json(result);
  out(`${result.servers_online}/${result.servers} servers online, ${result.gpus} GPU${result.gpus === 1 ? '' : 's'} (${result.reporting} reporting readings)`);
  printTotals(result.totals);
  return 0;
}

async function peers(words, options) {
  const t = await target(options);
  const result = await call(t, 'mesh.peers', {});
  const own = await call(t, 'mesh.card', {}).catch(() => null);
  if (options.json) return json(result);
  const rows = [];
  if (own) rows.push({ name: own.name, reach: 'this computer', card: own });
  // The model servers only: other apps on the mesh have no runners to manage (--all lists them).
  for (const peer of result.peers) {
    if (!options.all && (peer.member.local || peer.card?.software !== 'cortexmodelserver')) continue;
    rows.push({
      name: peer.member.name,
      reach: peer.reachable ? `${peer.path || 'connected'}${peer.rtt_ms ? ` ${peer.rtt_ms} ms` : ''}` : 'offline',
      card: peer.card,
    });
  }
  if (!rows.length) {
    out('no servers (this computer is on no network: cortexmodelserver service join <code>)');
    return 0;
  }
  const gpuText = (card) => {
    const gpus = card?.gpus ?? [];
    if (!gpus.length) return '-';
    return `${gpus.length}× ${bytes(gpus.reduce((sum, gpu) => sum + gpu.memory_bytes, 0))}`;
  };
  const util = (card) => {
    const gpus = card?.gpus ?? [];
    const memory = gpus.reduce((sum, gpu) => sum + gpu.memory_bytes, 0);
    return memory ? percent(gpus.reduce((sum, gpu) => sum + gpu.util * gpu.memory_bytes, 0) / memory) : '-';
  };
  table(rows, [
    ['SERVER', (row) => row.name],
    ['REACH', (row) => row.reach],
    ['SOFTWARE', (row) => (row.card ? `${row.card.software} ${row.card.version}` : '-')],
    ['GPUS', (row) => gpuText(row.card)],
    ['UTIL', (row) => util(row.card)],
    ['LANES', (row) => (row.card?.serving ? `${row.card.serving.lanes_serving}${row.card.serving.lanes_loading ? `+${row.card.serving.lanes_loading}` : ''}${row.card.serving.lanes_failed ? ` (${row.card.serving.lanes_failed} failed)` : ''}` : '-')],
    ['MODELS', (row) => (row.card?.models?.length ? row.card.models.map((model) => model.recipe || model.model_id).join(', ') : '-')],
    ['TOK/S', (row) => (row.card?.serving ? num(row.card.serving.tok_s) : '-')],
    ['TOKENS/H', (row) => (row.card?.serving ? count(row.card.serving.tokens_hour) : '-')],
  ]);
  return 0;
}

async function recipes(words, options) {
  const t = await target(options);
  const result = await call(t, 'serve.recipes.list', {});
  if (options.json) return json(result);
  table(result.recipes, [
    ['RECIPE', (recipe) => recipe.id],
    ['NAME', (recipe) => recipe.name],
    ['ENGINE', (recipe) => recipe.engine],
    ['KIND', (recipe) => recipe.kind],
    ['FROM', (recipe) => recipe.origin],
    ['WEIGHTS', (recipe) => bytes(recipe.weights_bytes)],
    ['CTX', (recipe) => `${ctx(recipe.ctx_default)}${recipe.ctx_max && recipe.ctx_max !== recipe.ctx_default ? `..${ctx(recipe.ctx_max)}` : ''}`],
    ['SLOTS', (recipe) => recipe.slots_default || '-'],
    ['HERE', (recipe) => (recipe.available ? (recipe.downloaded ? 'yes, downloaded' : 'yes') : `no: ${recipe.reason || 'not here'}`)],
  ]);
  return 0;
}

function printPlanChanges(plan) {
  if (!plan.changes.length) out('the lanes are as planned: no change');
  table(plan.changes, [
    ['CHANGE', (change) => change.op],
    ['LANE', (change) => change.lane.id || change.lane.recipe],
    ['RECIPE', (change) => change.lane.recipe],
    ['GPU', (change) => change.lane.gpu],
    ['SLOTS', (change) => change.lane.slots || 'default'],
    ['CTX', (change) => (change.lane.ctx ? ctx(change.lane.ctx) : 'default')],
    ['WHY', (change) => change.why],
  ]);
  for (const note of plan.notes) out(`note: ${note}`);
}

async function plan(words, options) {
  const t = await target(options);
  if (!options.reset) {
    const result = await call(t, 'serve.plan.get', {});
    if (options.json) return json(result);
    out(`plan${where(t)}: it runs by itself; lanes set by hand stay as set`);
    printPlanChanges(result);
    if (result.changes.length) out('back to the plan (lanes set by hand become the plan\'s again): cortexmodelserver plan --reset');
    return 0;
  }
  if (!options.json) say(`back to the plan${where(t)}; lanes that change load their models`);
  const view = await call(t, 'serve.plan.run', {});
  if (options.json) return json(view);
  if (view.lanes.length) printLanes(view.lanes);
  else out('no lanes');
  return 0;
}

async function lane(words, options) {
  const [what = 'list', ...rest] = words;
  const t = await target(options);
  if (what === 'list') {
    const view = await call(t, 'serve.lanes.list', {});
    if (options.json) return json(view);
    out(`lanes${where(t)}`);
    if (view.lanes.length) printLanes(view.lanes);
    else out('no lanes');
    return 0;
  }
  if (what === 'add') {
    const wanted = rest.join(' ').trim();
    if (!wanted) throw new UsageError('lane add <recipe> [--gpu N] [--slots N] [--ctx 32k] [--cache off|small|large|<MiB>]');
    const { recipes: list } = await call(t, 'serve.recipes.list', {});
    const recipe = findRecipe(list, wanted);
    if (!recipe.available) throw new Error(`${recipe.id} cannot run${where(t) || ' here'}: ${recipe.reason || 'not available'}`);
    const settings = laneSettings(options, { id: '', recipe: recipe.id, gpu: 0, slots: 0, ctx: 0, pinned: false, lend: true, tags: [], manual: true });
    if (!recipe.downloaded && !options.json) say(`${recipe.name} is not downloaded yet (${bytes(recipe.weights_bytes)}): the lane downloads it as it starts`);
    const view = await call(t, 'serve.lanes.create', settings);
    if (options.json) return json(view);
    out(`lane ${view.settings.id} added${where(t)}: ${view.name} on GPU ${view.settings.gpu}, ${view.state}`);
    out('this lane stays as set; cortexmodelserver plan --reset goes back to the plan');
    return 0;
  }
  if (what === 'edit') {
    const [id] = rest;
    if (!id) throw new UsageError('lane edit <lane> [--slots N] [--ctx 32k] [--cache ...] [--gpu N] [--recipe <recipe>]');
    const view = await call(t, 'serve.lanes.list', {});
    const current = findLane(view.lanes, id);
    const settings = laneSettings(options, current.settings);
    if (options.recipe) {
      const { recipes: list } = await call(t, 'serve.recipes.list', {});
      settings.recipe = findRecipe(list, options.recipe).id;
    }
    const changed = await call(t, 'serve.lanes.edit', { id: current.settings.id, settings });
    if (options.json) return json(changed);
    out(`lane ${changed.settings.id} changed${where(t)}: ${changed.name} on GPU ${changed.settings.gpu}, slots ${changed.slots || changed.settings.slots || 'default'}, ctx ${ctx(changed.ctx || changed.settings.ctx)}, ${changed.state}`);
    return 0;
  }
  if (what === 'remove' || what === 'rm' || what === 'delete') {
    const [id] = rest;
    if (!id) throw new UsageError('lane remove <lane>');
    const view = await call(t, 'serve.lanes.list', {});
    const current = findLane(view.lanes, id);
    const result = await call(t, 'serve.lanes.delete', { id: current.settings.id });
    if (options.json) return json(result);
    out(`lane ${current.settings.id} removed${where(t)}: it drains its requests and stops`);
    return 0;
  }
  throw new UsageError(`unknown lane command ${what}; use lane list, lane add, lane edit or lane remove`);
}

/** Bytes of the files under `dir` (links not followed). */
function folderBytes(dir) {
  let total = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) total += folderBytes(path);
    else if (entry.isFile()) total += lstatSync(path).size;
  }
  return total;
}

const weightsFolder = (recipe) => (recipe.repo && recipe.revision ? join(MODELS, recipe.repo.replace(/\//g, '-'), recipe.revision) : null);

async function models(words, options) {
  const [what = 'list', ...rest] = words;
  if (what === 'list') {
    const t = await target(options);
    const { recipes: list } = await call(t, 'serve.recipes.list', {});
    const { lanes } = await call(t, 'serve.lanes.list', {});
    const downloaded = list.filter((recipe) => recipe.downloaded);
    if (options.json) return json({ models: downloaded });
    if (!downloaded.length) {
      out(`no models downloaded${where(t) || ' here'}`);
      return 0;
    }
    // The folder's size is measured here only: another computer's disk is not read.
    table(downloaded, [
      ['RECIPE', (recipe) => recipe.id],
      ['NAME', (recipe) => recipe.name],
      ['ENGINE', (recipe) => recipe.engine],
      ['ON DISK', (recipe) => {
        const dir = weightsFolder(recipe);
        if (t.name || !dir || !existsSync(dir)) return bytes(recipe.weights_bytes);
        try { return bytes(folderBytes(dir)); } catch { return bytes(recipe.weights_bytes); }
      }],
      ['LANES', (recipe) => lanes.filter((lane) => lane.settings.recipe === recipe.id).map((lane) => lane.settings.id).join(', ') || '-'],
      ['FOLDER', (recipe) => (t.name ? '-' : weightsFolder(recipe) ?? '-')],
    ]);
    return 0;
  }
  if (what === 'remove' || what === 'rm') {
    if (options.on) throw new UsageError('models remove works on this computer only; run it there');
    const wanted = rest.join(' ').trim();
    if (!wanted) throw new UsageError('models remove <recipe>');
    const t = await localTarget();
    const { recipes: list } = await call(t, 'serve.recipes.list', {});
    const { lanes } = await call(t, 'serve.lanes.list', {});
    const recipe = findRecipe(list, wanted);
    const dir = weightsFolder(recipe);
    if (!dir || !existsSync(dir)) throw new Error(`${recipe.name} is not downloaded here`);
    // Other recipes of the same weights (one model on two engines) share the folder.
    const sharing = list.filter((other) => weightsFolder(other) === dir).map((other) => other.id);
    const using = lanes.filter((lane) => sharing.includes(lane.settings.recipe));
    if (using.length) throw new Error(`lane${using.length === 1 ? '' : 's'} ${using.map((lane) => lane.settings.id).join(', ')} use${using.length === 1 ? 's' : ''} it; remove ${using.length === 1 ? 'it' : 'them'} first (cortexmodelserver lane remove <lane>)`);
    if (readdirSync(dir).some((name) => name.endsWith('.part'))) throw new Error(`${recipe.name} is downloading now; try again when it is done`);
    if (!options.yes) {
      out(`${dir} (${bytes(folderBytes(dir))}) is in the models folder every Cortex app shares: another app on this computer may use it and would download it again.`);
      out(`remove it: cortexmodelserver models remove ${recipe.id} --yes`);
      return 1;
    }
    rmSync(dir, { recursive: true, force: true });
    // The repository folder goes with its last revision.
    const parent = dirname(dir);
    try { if (!readdirSync(parent).length) rmSync(parent, { recursive: true, force: true }); } catch { /* Gone already. */ }
    out(`removed ${recipe.name} (${dir})`);
    return 0;
  }
  throw new UsageError(`unknown models command ${what}; use models list or models remove <recipe>`);
}

async function calibrate(words, options) {
  const t = await target(options);
  const lane = words[0] ? findLane((await call(t, 'serve.lanes.list', {})).lanes, words[0]).settings.id : '';
  if (!options.json) say(`calibrating ${lane || 'every chat lane'}${where(t)}: each lane is measured alone and reloaded at each slot count and drafting setting, then each GPU's lanes run at once, and a full run applies what it finds (calls in progress finish first, this takes a while)`);
  const result = await call(t, 'serve.calibrate.run', { lane });
  if (options.json) return json(result);
  table(result.results, [
    ['LANE', (row) => row.lane],
    ['RECIPE', (row) => row.recipe],
    ['DECODE', (row) => (row.error || row.kind !== 'chat' ? '-' : `${num(row.decode_tok_s)} tok/s`)],
    ['PREFILL', (row) => (row.error || row.kind !== 'chat' ? '-' : `${num(row.prefill_tok_s, 0)} tok/s`)],
    ['ALL SLOTS', (row) => (row.error ? '-' : `${num(row.aggregate_tok_s)} tok/s`)],
    ['FIRST TOKEN', (row) => (row.error || row.kind !== 'chat' ? '-' : `${Math.round(row.ttft_ms)} ms`)],
    ['TOGETHER', (row) => (row.together_tok_s ? `${num(row.together_tok_s)} tok/s` : '-')],
    ['BEST SLOTS', (row) => (row.error ? '-' : (row.best_slots === row.current_slots ? `${row.best_slots}` : `${row.best_slots}, now ${row.current_slots}`))],
    ['DRAFT', (row) => row.best_draft || '-'],
    ['MOVED', (row) => (row.moved_to === undefined ? '' : `GPU ${row.moved_to}`)],
    ['TOTAL BY SLOTS', (row) => (row.error ? '-' : row.sweep.map((load) => `${load.slots}: ${num(load.tok_s)}`).join('  '))],
    ['ERROR', (row) => row.error || row.sweep_note || ''],
  ]);
  for (const share of result.gpus ?? []) out(`GPU ${share.gpu}, all lanes busy at once: ${share.error || `${Math.round(share.kept * 100)}% of their best alone gets done`}`);
  if (!result.results.length) out('no chat lanes to calibrate');
  return 0;
}

// ─── the app starter ────────────────────────────────────────────────────────────────────────────
// CortexModelServer in /Applications (~/Applications when that is not writable) on macOS, a .desktop
// entry on Linux, so the app starts from Launchpad, Spotlight or the app menu. It starts the newest
// app installed in ~/.cortexmodelserver/app (npx cortexmodelserver when there is none), or in
// development this checkout's ./cortexmodelserver. Every start writes it again, only where it changed.
const sh = (value) => `'${String(value).replace(/'/g, `'\\''`)}'`;

// The app starts detached (nohup ... &), so the starter ends at once: a starter that became the app's
// process would only be activated by the next open, never run again. When the app already runs, the
// starter brings its window forward (macOS: opening the frame's running Electron bundle activates it).
function starterScript() {
  const lines = ['#!/bin/sh', `export PATH=${sh(process.env.PATH || '/usr/local/bin:/usr/bin:/bin')}`, `export CORTEX_HOME=${sh(HOME)}`];
  const log = sh(START_LOG);
  lines.push(
    `pid="$(sed -n 's/.*"pid": *\\([0-9][0-9]*\\).*/\\1/p' ${sh(join(HOME, 'run', 'frame.json'))} 2>/dev/null)"`,
    'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then',
  );
  if (process.platform === 'darwin') {
    lines.push(
      `  shell="$(ps -axo ppid=,comm= | awk -v p="$pid" '$1 == p && /\\.app\\/Contents\\/MacOS\\// { sub(/^ *[0-9]+ +/, ""); sub(/\\/Contents\\/MacOS\\/.*/, ""); print; exit }')"`,
      '  [ -n "$shell" ] && open "$shell"',
    );
  }
  lines.push('  exit 0', 'fi', `mkdir -p ${sh(dirname(START_LOG))}`);
  if (DEV_ROOT) {
    lines.push(`nohup ${sh(join(DEV_ROOT, 'cortexmodelserver'))} >>${log} 2>&1 </dev/null &`);
  } else {
    const app = process.platform === 'darwin' ? 'CortexModelServer.app/Contents/MacOS/CortexModelServer' : 'cortexmodelserver/CortexModelServer';
    lines.push(
      `apps=${sh(APPS)}`,
      'for v in $(ls "$apps" 2>/dev/null | grep -E \'^[0-9]+\\.[0-9]+\\.[0-9]+$\' | sort -t. -k1,1nr -k2,2nr -k3,3nr); do',
      `  if [ -f "$apps/$v/${READY}" ]; then nohup "$apps/$v/${app}" >>${log} 2>&1 </dev/null & exit 0; fi`,
      'done',
      `nohup npx --yes cortexmodelserver >>${log} 2>&1 </dev/null &`,
    );
  }
  return `${lines.join('\n')}\n`;
}

/** Write `content` to `path` when it differs; true when it was written. */
function place(path, content, mode) {
  try {
    if (readFileSync(path).equals(Buffer.from(content))) return false;
  } catch { /* Not there yet. */ }
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, content, mode ? { mode } : undefined);
  if (mode) chmodSync(path, mode);
  return true;
}

/** The app icon: this checkout's, or the newest installed app's. */
function starterIcon(name) {
  if (DEV_ROOT) return join(DEV_ROOT, 'ui', 'shell', 'assets', name);
  const chosen = installed()[0];
  if (!chosen) return null;
  return process.platform === 'darwin'
    ? join(APPS, chosen, 'CortexModelServer.app', 'Contents', 'Resources', name)
    : join(APPS, chosen, 'cortexmodelserver', 'ui', 'shell', 'assets', name);
}

function writable(dir) {
  try { accessSync(dir, constants.W_OK); return true; } catch { return false; }
}

function installStarter() {
  try {
    if (process.platform === 'darwin') {
      const bundles = ['/Applications', join(homedir(), 'Applications')].map((folder) => join(folder, 'CortexModelServer.app'));
      const starter = (bundle) => {
        try { return readFileSync(join(bundle, 'Contents', 'Info.plist'), 'utf8').includes('ai.cortexmodelserver.starter'); } catch { return false; }
      };
      // A CortexModelServer.app that is not a starter (the disk image's app) is the user's own: leave it.
      if (bundles.some((bundle) => existsSync(join(bundle, 'Contents', 'Info.plist')) && !starter(bundle))) return;
      const bundle = bundles.find(starter) || (writable('/Applications') ? bundles[0] : bundles[1]);
      const folder = dirname(bundle);
      const contents = join(bundle, 'Contents');
      const icon = starterIcon('CortexModelServer.icns');
      const plist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>CortexModelServer</string>
<key>CFBundleDisplayName</key><string>CortexModelServer</string>
<key>CFBundleIdentifier</key><string>ai.cortexmodelserver.starter</string>
<key>CFBundleExecutable</key><string>CortexModelServer</string>
<key>CFBundleIconFile</key><string>CortexModelServer.icns</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>LSUIElement</key><true/>
</dict></plist>
`;
      let changed = place(join(contents, 'Info.plist'), plist);
      changed = place(join(contents, 'MacOS', 'CortexModelServer'), starterScript(), 0o755) || changed;
      if (icon && existsSync(icon)) changed = place(join(contents, 'Resources', 'CortexModelServer.icns'), readFileSync(icon)) || changed;
      if (changed) say(`CortexModelServer is in ${folder}`);
    } else if (process.platform === 'linux') {
      const script = join(HOME, 'starter', 'cortexmodelserver');
      const iconSource = starterIcon('cortexmodelserver-512.png');
      const icon = join(HOME, 'starter', 'cortexmodelserver.png');
      if (iconSource && existsSync(iconSource)) place(icon, readFileSync(iconSource));
      place(script, starterScript(), 0o755);
      const apps = join(process.env.XDG_DATA_HOME || join(homedir(), '.local', 'share'), 'applications');
      const entry = `[Desktop Entry]
Type=Application
Name=CortexModelServer
Comment=The model server of the Cortex apps
Exec="${script.replace(/(["`$\\])/g, '\\$1')}"
Icon=${icon}
Terminal=false
Categories=Development;System;
StartupWMClass=CortexModelServer
`;
      if (place(join(apps, 'cortexmodelserver.desktop'), entry)) say(`CortexModelServer is in the app menu (${apps})`);
    }
  } catch (error) {
    say(`cannot install the app starter (${error.message})`);
  }
}

async function foreground(args) {
  const appArgs = args.filter((arg) => arg !== '--foreground');
  let command;
  if (DEV_ROOT) {
    command = join(DEV_ROOT, 'cortexmodelserver');
  } else {
    const chosen = await ensureApp();
    if (!chosen) return 1;
    command = executable(join(APPS, chosen));
  }
  installStarter();
  return await new Promise((resolve) => {
    const child = spawn(command, appArgs, { stdio: 'inherit' });
    const forward = (signal) => { if (child.exitCode === null) child.kill(signal); };
    const term = () => forward('SIGTERM');
    const interrupt = () => forward('SIGINT');
    process.on('SIGTERM', term);
    process.on('SIGINT', interrupt);
    const done = (code) => {
      process.off('SIGTERM', term);
      process.off('SIGINT', interrupt);
      resolve(code);
    };
    child.once('error', (error) => { say(`cannot start CortexModelServer: ${error.message}`); done(1); });
    child.once('exit', (code) => done(code ?? 1));
  });
}

const HELP = `cortexmodelserver: the model server of the Cortex apps

Start and install
  cortexmodelserver [--headless] [--safe]      start it (the first run downloads it into ~/.cortexmodelserver/app)
  cortexmodelserver service                    install the background service (macOS login, Linux boot) and start it
  cortexmodelserver service join <code>        install the service, join the mesh (the code from Settings > Mesh), apply the plan
  cortexmodelserver service allow <computer>   let that computer of the mesh work here
  cortexmodelserver service status             how the service is
  cortexmodelserver service off                stop and remove the service (models and lanes are kept)
  (runner is the same command as service)

Every day                                      (each takes --on <server> and --json)
  cortexmodelserver status                     GPUs, engines, lanes, slots, tok/s
  cortexmodelserver cluster                    the whole mesh, added up
  cortexmodelserver peers                      every server of the mesh
  cortexmodelserver recipes                    what can run here
  cortexmodelserver plan [--reset]
                                               the automatic plan; apply it once, keep following it, or stop
  cortexmodelserver lane [list]
  cortexmodelserver lane add <recipe> [--gpu N] [--slots N] [--ctx 32k] [--cache off|small|large|<MiB>]
                     [--kv f16|q8_0] [--speculative on|off] [--idle <s>] [--pin] [--no-lend] [--tags a,b]
  cortexmodelserver lane edit <lane> [the same options] [--recipe <recipe>]   (default restores the recipe's value)
  cortexmodelserver lane remove <lane>
  cortexmodelserver models [list]              downloaded models
  cortexmodelserver models remove <recipe> [--yes]
  cortexmodelserver calibrate [<lane>]         measure every lane and, for all lanes, apply the best slots and drafting
`;

const COMMANDS = { status, cluster, peers, recipes, plan, lane, lanes: lane, models, calibrate };

async function main() {
  const args = process.argv.slice(2);
  const [first, ...rest] = args;
  if (args.includes('--help') || args.includes('-h') || first === 'help') {
    process.stdout.write(HELP);
    return 0;
  }
  if (first === 'service' || first === 'runner') {
    const [words, options] = parse(rest);
    return service(words, options);
  }
  if (first && Object.hasOwn(COMMANDS, first)) {
    const [words, options] = parse(rest);
    return COMMANDS[first](words, options);
  }
  if (first === 'starter') {
    installStarter();
    return 0;
  }
  if (first && !first.startsWith('-')) throw new UsageError(`unknown command ${first}; see cortexmodelserver --help`);
  if (args.includes('--foreground')) return foreground(args);
  const pid = running();
  if (pid) {
    installStarter();
    say(`CortexModelServer is already running (pid ${pid})`);
    return 0;
  }
  const chosen = await ensureApp();
  if (!chosen) return 1;
  installStarter();
  // Its own session, detached from this terminal: closing the terminal does not quit it.
  mkdirSync(join(HOME, 'logs'), { recursive: true });
  const log = openSync(START_LOG, 'a');
  const child = spawn(executable(join(APPS, chosen)), args, { detached: true, stdio: ['ignore', log, log] });
  closeSync(log);
  return await new Promise((resolve) => {
    child.on('error', (error) => {
      say(`cannot start CortexModelServer: ${error.message}`);
      resolve(1);
    });
    child.on('spawn', () => {
      child.unref();
      say(`CortexModelServer ${chosen} is starting; you can close this terminal (start output: ${START_LOG})`);
      resolve(0);
    });
  });
}

main().then(
  (code) => process.exit(code),
  (error) => {
    say(error.message);
    process.exit(error instanceof UsageError ? 2 : 1);
  },
);
