# model-llama-agent

The engine plugin (`model@1`, category `model`) that serves Gemma 4 the way CortexAgent does: its
llama.cpp, its catalog, its per-machine fitter and its slot scheduler. It is ct's `model-llama`
(speaks model@1 over CPP/1, supervises one `llama-server` child on loopback HTTP) with those parts
replaced. macOS (Metal) only. The frame runs one instance per lane.

## Engine

- `backend/vendor/llama.cpp` is CortexAgent's `vendor/llama.cpp` byte for byte (`diff -rq` empty,
  resynced 2026-11-07): unslothai/llama.cpp commit `468ad5824c4afb245338f0ede29ef31680ba59c7`
  (build b11876, a 0.7.0 development build), the same source as CortexTerm's model-llama, with the
  four patch sites in `tools/server/server-context.cpp` described in `vendor/llama.cpp/PATCHES.md`
  (marked `GlobalCortex patch`). It replaced v0.6.0 `d8123504` (five sites; upstream now does the
  first itself).
- CortexAgent copied only what it builds: CPU, BLAS, ET and Metal backends, `tools/mtmd`, and the
  server core without upstream's `server.cpp`, `main.cpp`, `server-models`, `server-connect`,
  `tools/CMakeLists.txt`, `tools/server/CMakeLists.txt` or `tools/ui`. So the stock `llama-server` target cannot be built from this tree. `agent-server/`
  builds it instead, without touching the tree:
  - `CMakeLists.txt` adds the tree exactly as CortexAgent's engine shim does
    (`crates/llama-engine/native/CMakeLists.txt`, same cache settings, including its
    `LLAMA_BUILD_IS_DEV ON`, `LLAMA_BUILD_NUMBER 11876`, `LLAMA_BUILD_COMMIT 468ad58`, so
    `--version` reads `0.7.0-dev (build 11876, commit 468ad58)`), and links the server core
    CortexAgent links (it carries the patches) with the vendored `server-http.cpp` and cpp-httplib;
  - `agent_server.cpp` is the entry point. It parses arguments with llama.cpp's own parser
    (`LLAMA_EXAMPLE_SERVER`), loads the model and runs the task loop as upstream's `server.cpp`
    (468ad58) instance mode does for one model, including its `ex_wrapper` exception mapping and
    the memory breakdown printed at exit. Of upstream's `server_prepare_params` only the model alias
    is kept (gc_engine runs none of it; the auto `--parallel` and `--kv-unified-per-slot` sizing
    never trigger with CortexAgent's explicit `--parallel` and `--ctx-size`). Router mode, the
    download worker, `-hf` resolution, llama-connect, MCP, server tools and the resumable-stream
    routes are not built. It registers the routes CortexAgent's `gc_engine.cpp` exposes, plus
    `/v1/models`, `/props` and `/metrics`;
  - `ui/` is the empty UI table that upstream's `ui-assets.cmake` generates when no UI is built.
- What `gc_engine.cpp` adds that llama-server flags cannot reproduce:
  - `/tool-call-markers`: the loaded chat template's tool-call opening markers that are a single
    special token. It is ported as a route in `agent_server.cpp`, in CortexAgent's current form:
    the probe body is parsed into a `common_chat_session`, which is applied to sampling settings,
    and the markers are read from their `grammar_triggers`.
  - Everything else in `gc_engine.cpp` is transport. It calls llama-server's own route handlers
    in-process over a pipe, with no socket. The responses are the same bytes, so HTTP changes no
    output. Its 404 for unknown paths and its exception mapping match upstream's.
- Linux: the tree has no CUDA or Vulkan backend, so ct's Linux path does not apply. `build.sh`
  stops with a message.

## Catalog and settings (from CortexAgent)

| id | repo @ revision | files |
|---|---|---|
| `gemma_4_e4b_it_q4km` | `unsloth/gemma-4-E4B-it-qat-GGUF` @ `8c5a9e4f…` | UD-Q4_K_XL 4,215,695,776 · MTP 59,678,016 · mmproj-F16 990,372,672 |
| `gemma_4_26b_a4b_it_q4km` | `unsloth/gemma-4-26B-A4B-it-qat-GGUF` @ `7b92b5b2…` | UD-Q4_K_XL 14,249,047,104 · MTP 251,939,328 · mmproj-F16 1,193,058,784 |

- **Install.** Every file is SHA-256 checked into `~/.cortex-models/<org>-<repo>/<revision>/` (ct's
  installer).
- **Default variant.** CortexAgent's `recommend`: the largest option that fits comfortably and
  decodes interactively.
- **The fit** (`src/fit.rs`, CortexAgent's arithmetic) runs once at each start, from the memory free
  at that moment:
  - device budget = Metal `recommendedMaxWorkingSetSize` (from `llama-server --list-devices`) − 3 GiB;
  - RAM budget = available RAM − ¼ of physical RAM;
  - slots: start at 16 and halve until every slot gets 32k of the pool (at least 2);
  - context: up to the native context, in 1024-token steps;
  - KV cache: f16, or q8_0 with `--flash-attn on` when f16 cannot reach the native context;
  - checkpoints: up to 16 per slot;
  - prompt cache: `--cache-ram` up to 6 GiB.
- **Arguments** (CortexAgent's `arguments()`): `--parallel N --kv-unified --ctx-size C
  --n-gpu-layers 99 --ubatch-size 2048 --cache-type-k/-v T [--flash-attn on] --cache-ram M
  --ctx-checkpoints K --checkpoint-min-step 2048 --cache-reuse 0 --no-prefill-assistant
  --no-context-shift --jinja --slot-save-path <data>/slots --mmproj … --spec-type draft-mtp
  --spec-draft-model … --spec-draft-n-max 4`, then `--no-webui --host 127.0.0.1 --port P`.
- **Scheduler** (`src/scheduler.rs`, CortexAgent's `SlotScheduler`):
  - every request is pinned to a slot (`id_slot`);
  - requests are admitted by priority, and only while the shared pool has room;
  - each conversation keeps its own slot;
  - an idle background conversation is saved to disk and restored when it comes back;
  - a displaced foreground conversation is saved before its slot is reused.

## Lanes

The frame starts one instance per lane. Each instance gets these from the frame:

- `HelloAck.config`: `[backend.config]` merged with the lane config;
- its own `CORTEX_PLUGIN_DATA`;
- `CORTEX_LANE`.

Each instance keeps these under its own name, so lanes never share them:

- under its own `CORTEX_PLUGIN_DATA`:
  - the llama-server pid file (`llama-server.pid`) and log (`llama-server.log`);
  - `--slot-save-path` (`slots/`), with the files a start removes (`*.slot`);
- its own loopback port, picked free;
- its own `model.progress` request ids (`llama-agent-<lane>-N`);
- its own graph setting for the chosen variant (`model-llama-agent.<lane>.variant`).

Per-lane config keys (0 or "" leaves the value to the fit):

- `slots`: fixes the slot count, like CortexAgent's `CORTEXAGENT_LLAMA_SLOTS`;
- `max_slots`: caps the slot count the fit chooses;
- `ctx`: caps the context;
- `cache_ram` (MiB, at most 6144): caps the prompt cache;
- `checkpoints` (at most 16): caps the checkpoints per slot;
- `kv_type` (`f16` | `q8_0`): fixes the KV cache type;
- `memory_limit` (device bytes, like `CORTEXAGENT_DEVICE_MEMORY_LIMIT_BYTES`): caps the device memory;
- `variant`, `vision`, `draft_max`, `gpu_layers`, `ubatch`, `idle_unload_s`.

ct's `borrow` (using a member's model through the mesh) is removed: a lane never borrows.

## CortexModelServer changes (differences from CortexAgent)

Resynced with CortexAgent on 2026-11-07 (its vendored llama.cpp, `crates/llama-engine/native` and
`crates/model`, uncommitted on top of `3fe5411`). The catalog, the fit arithmetic and the arguments
are CortexAgent's unchanged. What differs is lane plumbing or a marked patch
(`// CortexModelServer patch:`):

- Engine host: llama-server over loopback HTTP instead of CortexAgent's in-process `gc_engine`
  (`agent-server/`, see Engine): the HTTP layer (`server-http.cpp`, cpp-httplib, the empty UI
  table), the model alias, and the read-only routes `/v1/models`, `/models`, `/props`, `/metrics`.
  `server.rs` adds `--no-webui --host 127.0.0.1 --port P` after CortexAgent's arguments.
- `--slot-save-path`: this lane's `CORTEX_PLUGIN_DATA/slots` instead of
  `~/.cortexagent/cache/inference/llama` (lanes never share slot files).
- Lane ceilings in the fit (`fit.rs`, `server.rs`): `slots` fixes the count (CortexAgent's
  `CORTEXAGENT_LLAMA_SLOTS`), `kv_type` fixes the type (`CORTEXAGENT_KV_CACHE_TYPE`), `memory_limit`
  is `CORTEXAGENT_DEVICE_MEMORY_LIMIT_BYTES`; `max_slots`, `ctx`, `cache_ram` and `checkpoints` can
  only lower CortexAgent's constants. All 0 / "" by default, which is CortexAgent's fit exactly.
- Scheduler (`scheduler.rs`): model@1's `agent` priority is a class between CortexAgent's foreground
  and background; a model@1 call's `cache_key` names the conversation before CortexAgent's
  first-message key.
- Not carried over: CortexAgent's prompt shaping, `primed` / `block_only` state and stranded-process
  records (see below), its in-process worker process, and its probes of the template.

## Deviations from ct's plugin

Each one is marked in the code with `// CortexModelServer: from CortexAgent` or
`// CortexModelServer patch:`.

- `plugin.toml`: id `model-llama-agent`, exec `cc-model-llama-agent`, macOS only, a lane-oriented
  `[backend.config]`.
- `backend/Cargo.toml`: the package and binary are renamed `cc-model-llama-agent`.
- `backend/build.sh`: builds `agent-server/` (see Engine) on macOS only. The build name comes from
  `agent-server/CMakeLists.txt` (CortexAgent's), not from a `-DLLAMA_BUILD_COMMIT` flag.
- `backend/vendor/`: CortexAgent's tree (the same 468ad58 source as ct's, trimmed to what CortexAgent
  builds, with CortexAgent's patches). ct's `vendor/PATCHES.md` and `vendor/REVISION` are gone; the
  tree carries its own.
- `backend/agent-server/*`: new (see Engine).
- `src/catalog.rs`: CortexAgent's two specs, each with its projector (3 files), and `recommend`.
  ct's `for_memory` (E4B / E2B) is removed.
- `src/config.rs`: the lane keys above. ct's fixed `ctx 32768` / `parallel 2` / `kv_type f16` are
  removed.
- `src/fit.rs`, `src/gguf_meta.rs`, `src/scheduler.rs`: ported from CortexAgent.
- `src/server.rs`:
  - the start fits first, then passes CortexAgent's arguments;
  - ct's `--device`, `-b`, `-ngl 999`, `--ctx-checkpoints 2` and `--cache-ram 1024` are gone;
  - it creates the slot folder and starts the spill task.
- `src/local.rs`:
  - each request is counted (`/v1/chat/completions/input_tokens`, or `/tokenize` for a completion)
    and leased a slot;
  - its `id_slot` is set;
  - image parts are sent as data URIs when the projector is loaded;
  - `model.slots` reports the scheduler's keys, queue and saved conversations.
- `src/engine.rs`:
  - borrowing and the quantized-KV trial and f16 fallback are removed;
  - the default variant is CortexAgent's recommendation;
  - `idle_unload_s` 0 (the default) never unloads;
  - `model.info` reports the fitted pool, slots and KV type, the vision projector, the lane and
    `kind`.
- `src/store.rs`: only the lane's variant setting is kept (no KV check).
- `src/progress.rs`: request ids carry the lane.
- `src/main.rs`: id `model-llama-agent`, and the probe has no borrowed-model branch.
- `src/borrow.rs`: removed.

Not carried over from CortexAgent:

- its prompt shaping (priming, the host block, tool-message anchoring and folding, the tool-marker
  ban grammar, the reasoning-first grammar);
- its `primed` and `block_only` scheduler state;
- its stranded-process records.

model@1 callers shape their own prompts.
