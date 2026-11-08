# model-llama: llama.cpp `468ad58` with CortexTerm's 4 patches (`model@1`)

CortexTerm's `plugins/model-llama`, copied and renamed (`cortexterm` -> `cortexmodelserver` in the
manifest, Cargo description, `cache_key` prefix and `holder`). `backend/vendor/llama.cpp` is
byte-identical to CortexTerm's: unslothai `468ad58` (b11876) with the four `server-context.cpp`
patches in `backend/vendor/PATCHES.md`. Build: `sh backend/build.sh` (Metal on macOS; CUDA, Vulkan
and CPU backends as loadable libraries on Linux).

In cortexmodelserver the frame runs this plugin as several instances, one per lane. Each instance
gets `[backend.config]` merged with the lane's config (`plugins/serve/data/recipes/model-llama.toml`),
its own `CORTEX_PLUGIN_DATA`, `CORTEX_LANE=<lane id>`, and on Linux `CUDA_VISIBLE_DEVICES` /
`GGML_VK_VISIBLE_DEVICES` restricting it to the lane's GPU.

## Variants

| Variant | Origin | Files | Settings |
|---|---|---|---|
| `gemma-4-e4b-qat` | CortexTerm / the knowledge app's office lane | unsloth/gemma-4-E4B-it-qat-GGUF@8c5a9e4 UD-Q4_K_XL + MTP head | ctx 32768 (one unified pool), 2 slots, ubatch 2048, f16, MTP 4, checkpoints 2 (min step 2048), cache-ram 1024, cache-reuse 0, no context shift |
| `gemma-4-e2b-qat` | CortexTerm | unsloth/gemma-4-E2B-it-qat-GGUF@66a399f UD-Q4_K_XL + MTP head | as E4B |
| `gemma-4-e4b-qat-report` | the knowledge app's report lane | the E4B files | as E4B, but ctx 131072, 1 slot, answers capped at 16384 tokens, greedy, thinking per request |
| `qwen3.8-27b-q4` | CortexCode (Linux) | unsloth/Qwen3.8-27B-GGUF@4ca7207 UD-Q4_K_XL + mmproj-F16, z-lab/Qwen3.8-27B-DFlash2-GGUF@2d9571f Q4_K_M | ctx 32768 per slot, 2 slots, ubatch 2048 (CortexCode runs 512), q8_0 with flash attention (f16 fallback), `draft-dflash` at most 7, projector on the GPU, llama-server's own checkpoint and cache-ram defaults |
| `qwen3.8-27b-q3` | CortexCode (Linux) | as Q4 with UD-Q3_K_XL | as Q4, projector on the CPU (`--no-mmproj-offload`) |

`variant = ""` still picks E4B above 8 GiB and E2B otherwise, as in CortexTerm.

## CortexModelServer changes

Every site is marked `CortexModelServer patch:` in the source.

- **Loading on demand** (`backend/src/main.rs`, `backend/src/engine.rs`): a lane's self-test does
  not load the model when its weights are here, and reading `model.slots` does not count as a use,
  so an idle lane unloads.
- **NVIDIA's libraries for the other engines** (`backend/build.sh`): the Linux package also carries
  `libcurand`, which model-relex's CUDA build links. It uses this
  package's copy of all of NVIDIA's libraries, so the release carries them once.
- **Per-variant settings** (`backend/src/config.rs`, `backend/src/catalog.rs` `Defaults`, `plugin.toml`
  `[backend.config]`): `ctx`, `parallel`, `ubatch`, `kv_type`, `draft_max`, `cache_ram`,
  `checkpoints`, `max_tokens`, `greedy` are optional in the config; a given value wins, otherwise the
  variant's own (its origin's exact settings). `plugin.toml` no longer sets them, so a variant's
  defaults are not overridden by the manifest. New keys: `draft_max`, `cache_ram` (MiB),
  `checkpoints`, `max_tokens` (answer cap), `greedy`, `borrow`.
- **New variants** (`catalog.rs`): the E4B report lane and Qwen3.8-27B Q4 / Q3. `FileSpec.from` names
  a file kept in another repository (the DFlash 2 drafter); each file lives in its own
  `<models>/<org>-<repo>/<revision>` folder and is locked there while installed (as CortexCode does).
- **Two command lines** (`catalog::Profile`, `backend/src/server.rs`): `GemmaMtp` is CortexTerm's,
  unchanged in substance; `QwenDflash` is CortexCode's (`--spec-type draft-dflash`, `--mmproj`, `-c`
  = ctx * parallel, no `--kv-unified`).
- **Qwen request handling** (`backend/src/local.rs`): image parts go to the projector as data URLs
  (CortexCode's code; needs the `base64` crate, `backend/Cargo.toml`); without `--kv-unified` the
  slot pool is the sum of the slots. The thinking switch is the same for both templates
  (`chat_template_kwargs.enable_thinking`, budget when on).
- **Answer cap and greedy** (`local.rs`): a variant's `max_tokens` caps every call's `max_tokens`
  (and fills it in when the call has none); `greedy` forces temperature 0, top_k 1.
- **Lanes** (`backend/src/engine.rs`): with `CORTEX_LANE` set, the variant comes from the config only
  (the shared `model.variant` setting is neither read nor written), and the instance never borrows
  a member's model (a lane without a usable GPU answers `unavailable`). `borrow = false` turns
  borrowing off outside lanes too (CortexTerm ignored the key). `model.info` carries `extra.lane`,
  `extra.gpu` and `extra.device`, the variant's family, tag, speculative type and vision.
- **Contract**: `ModelInfo.kind = "chat"` (the protocol's new field).

GPU pinning needs no change: `llama-server --list-devices` lists only what `CUDA_VISIBLE_DEVICES`
and `GGML_VK_VISIBLE_DEVICES` leave visible, and the plugin passes the chosen device as `--device`.
