# model-embed: embedding engine (`model@1`, kind `embed`)

`cc-model-embed` serves the `model` category over CPP/1 with **the knowledge app's embedder**: texts to
vectors (`model.embed`). Swap policy: drain first. The model card tags it `embed`;
`model.info.kind` is `embed`. It serves one model, bekko, the embedder every Cortex app uses
(the knowledge app's semantic index, CortexCode's and CortexTerm's memory, CortexAgent):

| `model` | repository @ revision | runtime | dims |
|---|---|---|---|
| `bekko-embedding-v1-a8m` | `hotchpotch/bekko-embedding-v1-a8m@c721113d59a1d91b447450324f51c4b3332c924a` | ModernBERT on candle 0.10.2: Metal on macOS, CUDA on Linux where NVIDIA's driver is (`prefer_gpu`), else the CPU | 384 |

It mean-pools the token states over the attention mask and L2-normalizes. At most 512 tokens are
read per text.

## Where the engine comes from (copied as is)

From the knowledge app's repository at commit `4ea0c82549a80eeadcbae463489276f424c36366`
(2026-11-06), with the removal of the unused embedders made there on 2026-11-07 (the three ONNX
presets MiniLM-L6/L12 and bge-small, and the Core ML adapter, which only those reached):

- `backend/src/embedding/` is the knowledge app's adapters crate's `src/embedding/`, byte for byte apart from the renames under "CortexModelServer changes". It holds the
  ModernBERT encoder (`modernbert.rs`, `modernbert/**`, the injection guard included), the semantic
  model's pinned files and download (`semantic_model*`), the options (`config.rs`), the work gate,
  `mod.rs`, and the RTen ONNX loader for the bindings' registered models (`onnx.rs`,
  `model_config.rs`, `download.rs`, `huggingface_hub_cache_dir.rs`), which sits behind the knowledge app's
  `onnx-embed` feature and is not compiled here. `diff -r` against the knowledge app's folder shows only those renames.
- `backend/resources/injection_guard_v2.safetensors` (7,871,916 B) is
  the adapters crate's `resources/injection_guard_v2.safetensors`, which the injection guard
  compiles in (`include_bytes!`). The guard is built but not served (`model@1` has no method for it).

Dependencies are the adapters crate's at the same versions and features (`candle-core`/`candle-nn`
`=0.10.2` with Metal + Accelerate on macOS, `tokenizers =0.23.2`, `ureq 3`, `rayon`,
`parking_lot 0.13`, its `embed` feature); the Rust pin is
the knowledge app's 1.95.0 (`parking_lot 0.13` needs it).

Written here so the copied files compile with no change but the renames below:

- `backend/embed-common/`: a crate named `embed-common` with only what the files use
  (`utils::error::{Error::Internal, Result}`, `cpu::max_cpu_threads` copied, and the
  `embed_info!`/`embed_warn!` macros on `tracing`).
- `backend/src/shared_models.rs`: what the files call `crate::shared_models` (`PinnedModelFile`
  copied; `shared_model_dir` and `fetch_shared_model_file` on `cc-shared-models`, the same layout and
  `.download.lock`).

## CortexModelServer changes

Loading on demand (`backend/src/main.rs`, a marked patch): a lane's self-test does not load the model
when its weights are here.

The copied files (`backend/src/embedding/**`, and what `embed-common/` and `src/shared_models.rs`
copy) differ from the knowledge app's only by renames of names that carry that app's own name,
which this repository keeps out:

- its common crate is `embed-common` (lib `embed_common`), so every path into it is
  `embed_common::`;
- its two log macros are `embed_info!` / `embed_warn!`;
- its environment variables are `CORTEX_EMBED_MAX_INPUT_TOKENS` and `CORTEX_SEMANTIC_MODEL_DIR`;
- the rayon thread names are `cortex-embed-{index}` / `cortex-guard-{index}`;
- in the ONNX loader (not compiled), the default Hugging Face cache is `~/.cache/huggingface/hub`
  (the origin uses a folder under its own home), and the doc examples name `crate::embedding`.

One patch beyond the renames: `modernbert/select_device.rs` also tries CUDA in a build with the
`cuda` feature (marked `CortexModelServer patch`; the origin has only a Metal path). If the GPU cannot
load or run the encoder, `from_files` falls back to the CPU as it does for Metal.

A `diff -r` against the knowledge app's folder shows those lines and nothing else.

The Linux package carries that CUDA build. candle links NVIDIA's libraries at build time, so a CUDA
executable does not start without the driver. `backend/build.sh` builds the plain executable and, on
Linux, a second one with `--features cuda` in `cuda/`. The plain one becomes the CUDA one
(`become_cuda_build` in `src/main.rs`) when `libcuda.so.1` loads and an installed model-llama package
holds NVIDIA's libraries, which it ships for llama-server (so the Linux release carries them once).
Otherwise it stays on the CPU.

## Methods

| method | answer |
|---|---|
| `model.embed` | `EmbedParams {texts, normalize = true, truncate?}` → `EmbedResult {vectors, dims, model_id, tokens}`. Vectors are always unit length (the embedder normalizes). `truncate` cuts each text after that many tokens (by the model's own tokenizer's offsets, so the model reads about that many), capped at `max_input_tokens`; `tokens` counts what is read. |
| `model.info` | `kind: "embed"`, tag `embed`; `extra` has the lane, state, runtime, dims, pooling, batch size |
| `model.slots` | one slot (busy while embedding), the queue, `loaded`, recent metrics |
| `model.metrics` | `LiveMetrics` over 5 minutes: requests, `prefill_tps` (tokens embedded per second), mean queue time; decode rates are 0 |
| other `model.*` | `unavailable` |

## Model files

In `~/.cortex-models/<org>-<repo>/<revision>/` (`cc-shared-models`, the folder the knowledge app and the other
Cortex apps use). bekko: `config.json` (1,207 B), `model.safetensors` (211,952,896 B),
`tokenizer.json` (34,363,442 B), downloaded by the knowledge app's own `provision_semantic_model_files` (pinned
size and SHA-256).

## Lanes

Every lane is its own instance: `[backend.config]` with the lane's settings merged over it,
`CORTEX_LANE` (shown in `model.info.extra.lane`), its own data folder (the plugin keeps nothing
there). Model files are shared read-only; downloads are serialized by the shared lock; the
process-wide settings the copied code reads (`CORTEX_EMBED_MAX_INPUT_TOKENS`, the progress handler)
are per process, so lanes never collide.

## Configuration (`[plugin.config]`)

| key | default | meaning |
|---|---|---|
| `model` | `bekko-embedding-v1-a8m` | the model (the only one) |
| `batch_size` | 32 | texts per encoder pass (ModernBERT orders texts by length first) |
| `threads` | 4 | CPU threads (the knowledge app's semantic default; capped at half the cores) |
| `prefer_gpu` | true | ModernBERT on Metal when the Mac has it |
| `max_input_tokens` | 512 | tokens read per text at most |
| `idle_unload_s` | 300 | drop the model after this long without a request; 0 keeps it |
| `models_dir` | `~/.cortex-models` | the shared models folder |
| `download` | true | download missing files |

## Building

`cargo build --release` in `backend/` (Rust 1.95.0, this folder's `rust-toolchain.toml`). Without a
frame: `cc-model-embed call model.embed '{"texts":["hello"]}'`.
