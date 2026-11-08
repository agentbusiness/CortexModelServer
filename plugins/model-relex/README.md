# model-relex: GLiNER-relex extractor engine (`model@1`, kind `extract`)

`cc-model-relex` serves the `model` category over CPP/1 with **the knowledge app's relex lane**: GLiNER-relex
large (`knowledgator/gliner-relex-large-v1.0@4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290`, DeBERTa-v3
encoder plus small heads) on candle 0.11. One encoder pass over a passage gives its entities (spans
typed from a list of kinds) and the relations between them (typed from a list of relation names).
No generation. Swap policy: drain first. The model card tags it `bilstm`, the knowledge app's claim tag, alone, as the knowledge app's lane does;
`model.info.kind` is `extract`.

## Where the engine comes from (copied as is)

`backend/src/relex/` is the knowledge app's `model-worker/src/relex/` byte for byte, from
the knowledge app's repository at commit `4ea0c82549a80eeadcbae463489276f424c36366`
(2026-11-06, the files clean in that tree):

| file | sha256 |
|---|---|
| `mod.rs` | `8ed8b1879deb6e8d4b1ce449b403067cb6ce6b83f60dd0af644f103c9b928a85` |
| `config.rs` | `e8f8595b524f1bf801fcd1920a849af7a862df3f03c587784a60ec6bc21c00d6` |
| `deberta.rs` | `a26008a2d5927965476dfacc1ba863ffe3cc6e6a265c5bd1233d07cd75a59574` |
| `decode.rs` | `521aee14ae1748096f1e3646b3dd4a781f2bb4425f28e6e703fcf67b4dbcb5ee` |
| `encode.rs` | `596604b299aae32d6d9fb59b066f9821a260589ab68bb5919f7281c806bf8c94` |
| `lstm.rs` | `cc6c7fe642e854cb6456ba83b54c4312710e247b681c3eedc2c360d465fa0048` |
| `mlp.rs` | `aae1674982677cdccea32d1ee570adb7827a5b2a1808613c9420725ed1c8ee3b` |
| `model.rs` | `86d7947503a090994f0e61560e5e6b4b4f08b8c0801573d323d8f6e1e1ba9ddd` |
| `windows.rs` | `e3f6d0f8d3d32824a3154c945c5797cf33035091bd4d44b40b38c3614133a5e8` |
| `words.rs` | `d1bfc2223f1cdaa60c67df3f98941de3a3fe16fdc557b8e49fec9989ef71541b` |

`diff -r` against the knowledge app's folder prints nothing.

The dependencies are the worker's, at the same versions and features: `candle-core`/`candle-nn` 0.11
(Metal + Accelerate on macOS, `cuda` feature for CUDA), `tokenizers` 0.22 (`fancy-regex`), `regex`,
`anyhow`. What the lane does, unchanged: the encoder in f16 (`LAB_RELEX_DTYPE=f32` for 32 bits; the
plugin's `encoder_dtype` sets it), the heads in f32, the BiLSTM on the CPU, texts in windows of whole
sentences of at most 768 subtokens, a warm pass at load, a device synchronize after each passage.
Device: Metal when available, else CUDA (a build with `--features cuda`), else the CPU.

`backend/src/main.rs` adds only what the knowledge app's worker gave the lane (`Args { model }`, `emit`,
`model_name`, the last two copied from the knowledge app's `model-worker/src/main.rs`).

## CortexModelServer changes

None in the copied files: `backend/src/relex/` needed none of the renames the embedder took (the
knowledge app's name appears nowhere in it), so it is still byte for byte the origin's. Only prose
(comments, this README, the descriptions) names the origin as "the knowledge app".

## How it runs

The plugin process speaks CPP/1. On the first `model.extract` it starts one child, its own
executable as `cc-model-relex worker --model <dir>`, which is the knowledge app's lane reading
`{"id", "relex": {...}}` lines on stdin and answering `{"id", "text", "stats"}` on stdout, exactly as
the knowledge app's worker pool drives it. One slot: passages run one at a time; others wait in arrival order
(`model.slots.queued`). After `idle_unload_s` without a passage the child is ended, which frees all of
the model's memory; the next passage starts it again (load plus the Metal kernel compile, 10-20 s).
A cancelled call drops its answer; the passage already in the encoder finishes (stopping it would
mean killing the model).

## Methods

| method | answer |
|---|---|
| `model.extract` | `{"relex": {text, kinds, relations?, threshold?, relation_threshold?}}` → `{"text": "<JSON {entities, relations, words}>", "stats": {...}}`, the same shape the knowledge app's mesh sends to a lent extractor and reads back (`cc_protocol::contracts::model::ExtractParams` / `ExtractResult`). Missing thresholds take the config's (0.3 / 0.5). |
| `model.info` | `kind: "extract"`, tag `bilstm`; `extra` has the lane, state, dtypes, window, thresholds |
| `model.slots` | one slot (busy while extracting), the queue, `loaded`, recent metrics |
| `model.metrics` | `LiveMetrics` over 5 minutes: requests, `prefill_tps` (subtokens read per second of extraction), mean queue time; decode rates are 0 |
| other `model.*` | `unavailable` |

## Model files

`~/.cortex-models/knowledgator-gliner-relex-large-v1.0/4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290/`
(`cc-shared-models`; the folder the knowledge app uses too): `gliner_config.json` (3,565 B),
`model.safetensors` (1,866,369,448 B), `tokenizer.json` (8,649,416 B), pinned to the sizes and
SHA-256 of the knowledge app's catalog. A missing file is downloaded with the revision folder's
`.download.lock` held, into `<file>.part`, checked and renamed. The probe reports a running
download instead of failing.

## Lanes

Every lane is its own instance of this plugin: its config is `[backend.config]` with the lane's
settings merged over it, `CORTEX_LANE` names it (shown in `model.info.extra.lane`) and its data
folder is its own. The plugin keeps nothing in its data folder; the model files are shared
read-only and their download is serialized by the shared lock, so lanes never collide. Each lane
holds its own copy of the model in memory.

## Configuration (`[plugin.config]`)

| key | default | meaning |
|---|---|---|
| `idle_unload_s` | 300 | end the worker after this long without a passage (the knowledge app: 5 minutes); 0 keeps it |
| `threshold` | 0.3 | entity threshold when a passage names none |
| `relation_threshold` | 0.5 | relation threshold when a passage names none |
| `encoder_dtype` | `f16` | `f32` doubles the encoder's memory |
| `models_dir` | `~/.cortex-models` | the shared models folder |
| `download` | true | download missing files |

## Building

`cargo build --release` in `backend/` (Rust 1.94.1, this folder's `rust-toolchain.toml`); on Linux
with an NVIDIA GPU, `cargo build --release --features cuda` (CUDA toolkit needed). Without a frame:
`cc-model-relex call model.extract '{"relex":{"text":"...","kinds":["person"]}}'`.
