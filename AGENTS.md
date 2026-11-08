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
  (`model-uzu`, `model-llama`, `model-llama-agent`, `model-bonsai`, `model-relex`, `model-embed`).
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
- model and resource sharing: model tags (`coding`, `office`, and any other an app lends under), a
  model already in memory on this computer is used before one is loaded (no second copy of one model
  on a computer), a model only on a GPU (Metal on macOS, CUDA or Vulkan on Linux), one models folder
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
3. `plugins/mesh/backend/src/service.rs`: borrowers may also call `model.metrics`, `model.embed`,
   `model.extract`; `mesh.peers` names each member's computer (the machine key the log already holds).
4. `protocol/src/contracts/model.rs`: `model.embed`, `model.extract`, `ModelInfo.kind`.
5. `protocol/src/contracts/frame.rs` and `frame/src/`: lanes (`frame.lanes.*`, routing a model call by
   its `plugin` param).
6. `protocol/src/contracts/serve.rs`: `serve@1`, this app's own contract.

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
| `model-embed` | the knowledge app's adapters crate, `src/embedding` | `backend/src/embedding/` (candle 0.10.2, the knowledge app's `embed` feature; bekko only), apart from the renames in the plugin's README | `diff -rq` against the knowledge app's repository; `Cargo.lock` versions |

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
- The serve plugin owns the lanes: they are graph records, planned from the hardware, from what the
  Cortex apps on this computer use, and from what was set by hand; it sets them in the frame at every
  start.
- Every lane gets its memory as a limit (`memory_limit_bytes` where the engine has it): an engine
  never plans for the whole GPU when other lanes share it.

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
