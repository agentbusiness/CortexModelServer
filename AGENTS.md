# CortexModelServer: notes for anyone working on this repository

CortexModelServer (`cortexmodelserver`) is the model-serving and administration hub of the cortex-mesh.
It is installed on every Mac and Linux server. On each computer it reads the GPUs, runs many engines
side by side, each with many models in one or more **lanes** (one recipe on one GPU, with its own
slots, context and cache), serves the models every Cortex app on that computer needs, and lends every
lane to the mesh. Its window shows the whole mesh as a cluster; its command line does everything the
window does. Plan: `docs/BUILD-PLAN.md`.

## Keep it simple

Pick the simplest design that does the job. Don't add sharing, sync, layers or options that nobody asked
for. When a plan grows a second mechanism for the same thing, drop one of them.

## Words

- **Engine**: one of our patched inference engines, each its own plugin of category `model`
  (`model-uzu`, `model-llama`, `model-llama-agent`, `model-bonsai`, `model-relex`).
- **Recipe**: one model on one engine with the exact settings of the app it comes from
  (`plugins/serve/data/recipes/<engine>.toml`, shipped data).
- **Lane**: one recipe running on one GPU: one frame instance of its engine plugin, named by the lane's
  id on the mesh card (the `plugin` of model calls). Many lanes run at once.

## The shared standard (CortexCode, CortexTerm, CortexModelServer, the knowledge app)

CortexModelServer is a separate app with its own frame, plugins, home (`~/.cortexmodelserver`), releases and
memory. Its code was copied from CortexTerm (`../CortexTerm`), which was copied from CortexCode
(`../CortexCode`), and reshaped. All of them follow the same standard:

- the plugin package: `plugin.toml` (`ccplugin/1`), `.ccplugin`, hello and lifecycle;
- CPP/1 and routing by method prefix;
- the wire shape of every contract the apps share (`model@1`, `mesh@1`, `notify@1`, `skills@1`);
- the mesh: ALPNs, pairing, membership log, card, claims, lending, `mesh.link`;
- model and resource sharing: model tags (`coding`, `office`, `smalloffice` for Gemma E2B, `largeoffice` for Gemma 26B, and any other an app lends under), a
  model already in memory on this computer is used before one is loaded (no second copy of one model
  on a computer), the serving order on one computer (each app says its own `sharing_priority` on its card, the highest first; no app knows another by name): an app hands a tag over to an app of a higher priority that serves the tag on the same computer, stops its own model for it, uses the other app's and does not lend it (a model an app does not run is not its to share), a model only on a GPU (Metal on macOS, CUDA or Vulkan on Linux), one models folder
  for every Cortex app (`~/.cortex-models`, `docs/SHARED-MODELS.md`; `shared-models/`);
- shared services (`protocol/src/contracts/service.rs`);
- where data lives (below) and the graph's contract: the memory plugin's `graphapi.rs`, `streams.rs`,
  `settings.rs` and `app.rs` the same in every repository, byte for byte;
- `sandbox-kit/`, `service-kit/`, `shared-models/`, `model-kit/`: the same code in every repository,
  byte for byte.

**The standard is followed 1:1. It is not optional and not a starting point.** The cortex-mesh
(transport, ALPNs, pairing, the membership log, cards, claims and their ranking, lending,
`mesh.link`, local members) and the plugin architecture behave here exactly as in CortexCode,
CortexTerm and the knowledge app, so every app on the network works with every other one:

- The mesh plugin's sources are CortexTerm's, byte for byte, except the app's name (`lib.rs`
  `SOFTWARE`, `APP`) and the additions listed below. Before changing a line of `plugins/mesh`,
  `protocol/`, the frame's routing or a shared kit, diff it against `../CortexTerm` and
  `../CortexCode`.
- A difference is allowed only when it is **additive on the wire** (new optional fields with
  `serde(default)`, new methods, new topics, which the other apps ignore) and **changes no standard
  behaviour** (who may connect, what is lent, how claims are ranked and granted, the log). App policy
  goes in this app's UI and plugins, never in the standard's code: the per-member switch is left out
  of Settings › Mesh, while `may_work` stays the standard's (every member allowed unless refused).
- Every difference is marked in the code (`// CortexModelServer:`), listed here, and listed in
  `docs/STANDARD-CHANGES.md` under "Pending in the other apps", so the other apps take it.
- **Any change to the plugin architecture, the mesh, or model and resource sharing is made in every
  app that carries the standard**, in the same piece of work, with the version it lands in.

What this copy adds to the standard today (all additive):

1. `protocol/src/contracts/mesh.rs`: `Card.gpus` (`GpuCard`), `Card.serving` (`ServingCard`),
   `ModelCard.kind` / `engine` / `recipe` / `gpu` / `tokens_hour`, `Peer.computer`.
2. `plugins/mesh/backend/src/card.rs`: fills those from `serve.status.get`; a lane set not to lend
   stays off the card (the standard's "a model not lent is not on the card", per lane).
3. `plugins/mesh/backend/src/service.rs`: borrowers may also call `model.metrics`,
   `model.extract`; `mesh.peers` names each member's computer (the machine key the log already holds).
4. `protocol/src/contracts/model.rs`: `model.extract`, `ModelInfo.kind`.
5. `protocol/src/contracts/frame.rs` and `frame/src/`: lanes (`frame.lanes.*`, routing a model call by
   its `plugin` param, `LaneSpec.gpu` and the GPU line: a person's call on a lane goes before other
   work on that GPU's other lanes, which waits at most 30 s; a lane without `gpu` is not in it).
6. `protocol/src/contracts/serve.rs`: `serve@1`, this app's own contract.

Embedding is not shared: every app runs its own embedder (IronGraph's, or the knowledge app's in process).

Every app states on its card the context it needs from whoever serves a model (`ModelCard.ctx_needed`, `UsedModel.ctx_needed`, by default the model's `ctx_total`); a lane runs the biggest any app states, and the same field is in every app.

A call claims context as well as a slot (`ClaimParams.tokens`, `claims.rs` in every app): a use holds its stated tokens of the model's shared pool, at least one slot's share. A call waits until the pool has room for it, and the calls behind it wait behind it.

The office model's pool is counted in 32k shares in every app (`--kv-unified`, `ctx = parallel × 32768`): one share per slot, a normal call books one, a bigger call books several (the knowledge app's 131k report call books four) and runs on the same lane. An app alone runs its own slots (CortexTerm 2, the knowledge app 2 plus the report's four); a server runs one share per slot plus the shares of the biggest call any app on its computer states (`ctx_needed`). No app has a model only for itself: every lane is lent, and a call goes to any computer whose pool has room for it. Nothing of ours caps the slots or the pool; memory does.

## Engines are copied, not rewritten

Each engine plugin is the origin app's code, copied bit for bit: its vendored engine (uzu `1763a366`
with P1–P8 from CortexCode; llama.cpp `468ad58` with CortexTerm's 4 patches; CortexAgent's llama.cpp
`468ad58` with its 4 (as CortexAgent vendors it now); PrismML's fork for Bonsai; the knowledge app's candle relex and ModernBERT), its catalog, its
settings code. A change this app needs is the smallest possible, marked in the code
(`// CortexModelServer patch:`) and listed in the plugin's README under "CortexModelServer changes".
Upgrading an engine is a standard change: every app that carries it takes the upgrade.

What each engine must match, byte for byte, and how to check it (run these before a release and
whenever an origin app changes its engine):

| Engine plugin | Origin | Must be identical | Check |
|---|---|---|---|
| `model-uzu` | CortexCode `plugins/model-uzu` | `backend/vendor/uzu` (uzu `1763a366`, P1–P8, `PATCHES.md`, `REVISION`) | `diff -rq -x target plugins/model-uzu/backend/vendor ../CortexCode/plugins/model-uzu/backend/vendor` |
| `model-bonsai` | CortexCode `plugins/model-bonsai` | `backend/vendor` (PrismML fork) and `backend/native` (C shim) | the same `diff -rq` on both folders |
| `model-llama` | CortexTerm `plugins/model-llama` (= the knowledge app's worker) | `backend/vendor` (llama.cpp `468ad58`, 4 patches); `server-context.cpp` also equals `model-worker/vendor/llama.cpp` in the knowledge app's repository | `diff -rq` against CortexTerm; `cmp` against the knowledge app's repository |
| `model-llama-agent` | CortexAgent `vendor/llama.cpp`, `crates/model`, `crates/llama-engine` | `backend/vendor/llama.cpp` as CortexAgent has it now (its `REVISION`, `PATCHES.md`); catalog, fitter and server flags as in `crates/model` | `diff -rq` against `../CortexAgent/vendor/llama.cpp` |
| `model-relex` | the knowledge app's `model-worker/src/relex` | `backend/src/relex/` (candle 0.11.0, tokenizers 0.22.2) | `diff -rq` against the knowledge app's repository; `Cargo.lock` versions |

Outside the vendored engines, a plugin's own code may differ from its origin only by a marked
patch (`// CortexModelServer patch:`, or the plumbing it needs) listed in its README: diff each
source file against the origin and read every hunk. When an origin moves (a new engine revision,
new patches, a changed catalog or setting), this app takes the same change in the same piece of
work. LFM2.5 runs on CortexCode's uzu (`1763a366`, P1–P8) with the knowledge app's exact LFM settings and
template; the knowledge app's own copy is an older checkout of the same patched uzu.

## Lanes

- The frame runs a plugin of category `model` only as lanes (`frame.lanes.set`, `frame/src/lanes.rs`):
  slot key `model/<lane id>`, the lane's config merged over the package's `[backend.config]`, its own
  environment (`CORTEX_LANE`, `CUDA_VISIBLE_DEVICES` / `GGML_VK_VISIBLE_DEVICES`) and data folder
  (`plugins-data/<exec>/lanes/<id>`). A model call is routed by its `plugin` param to the lane of
  that name. The category's own slot never starts.
- The serve plugin owns the lanes: they are graph records. A computer starts with none: they come from
  what the Cortex apps on this computer have selected (their cards, kept while an app is away) and
  from what was set by hand; it sets them in the frame at every start. A lane runs the app's own
  sizes and is never shrunk or grown to fit: a shared pool is one share per slot plus the shares of
  the biggest call any app states (`fit::slots_of`, `fit::ctx_of`); LFM on uzu is one conversation's
  room per slot, the 32k the knowledge app states; Qwen on uzu (`ctx_default` 0) is set no slots and
  no context and fits itself in the memory the lane is given, as CortexCode runs it. Slots and
  context of any lane can be set by hand. On a Mac every selected model that
  fits the GPU alone gets a lane: lanes that do not fit loaded together take turns, unloading after
  five idle minutes, as the knowledge app's own lanes do. On a discrete GPU a lane that does not
  fit beside the others is left out.
- A calibration measures a lane (one call, its slots busy at once) and tunes its drafting
  (`speculative`, `draft_max`), applied to a lane set by hand directly and to a planned lane through
  the plan. It never changes a lane's slots or context, never moves it, and never removes it.
- Every lane gets its memory as a limit (`memory_limit_bytes` where the engine has it): a lane its
  engine sizes gets the GPU less the other lanes on it, so it never plans for the whole GPU when
  other lanes share it.
- A lane's process runs from the start, but its model loads on the first real call and unloads after
  an hour without one (`idle_unload_s`), so it stays on the mesh card with `loaded: false`. Only a
  pinned lane is loaded ahead of a call. A lane's self-test does not load its model; reading its
  info or slots neither loads it nor counts as a use.

## Where data lives

**The graph is the only record**, exactly as in CortexTerm: settings, lanes, GPUs, engines, plans,
calibrations, metric summaries, served calls and lane events are in IronGraph, written through the
ordered outbox (`graph.rs`, the same file as CortexTerm's system plugin). Histories are streams with a
retention (`model.served`, `model.lanes.events`, `serve.minutes`, `serve.cluster.minutes`). Live
readings are streamed (`serve.sample`, `system.sample`) and not stored per sample. No JSON, TOML or
other data files hold records. Weights stay files in `~/.cortex-models`; recipes are shipped data.
See `docs/GRAPH-MODEL.md`.

## Remote administration

Every cortexmodelserver of the mesh administers every other one: membership is the trust, as for
lending models. The window and the command line reach another server through `mesh.link`; the
standard mesh allows every member unless refused, and this app offers no switch to refuse one.

## Use the theme's controls

Never use the browser's native controls: their pop-ups ignore the theme. A dropdown is the SDK's
`Menu` (`@cortex/sdk`, `.pick` with the theme's `.menu` list), never `<select>`. A switch is `.sw`,
a button `.detent`, a segmented choice `.seg-set`, never a native checkbox or radio. Inputs,
buttons and pickers on one row share one height. `theme-cortex` stays byte-identical to the other apps'.

## No builds unless asked

Write the code only. Never build, package, install, release or hot-swap the app or a plugin
(`build-app.sh`, release scripts, `./cortexmodelserver service`) until the owner asks for it in so many words.
`cargo check` to see that the code compiles is fine.

## Checking a change

As in CortexCode: no test suites, check scripts or benchmark folders in the repository. Check a change by
using the real app on real machines: a Mac, a Linux NVIDIA server, an 8 GB Mac.
