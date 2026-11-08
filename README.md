<p align="center">
  <img src="plugins/workbench/ui/window/public/cortexmodelserver.svg" alt="CortexModelServer" width="96" />
</p>

<h1 align="center">CortexModelServer</h1>

<p align="center">
  <strong>A local model server for your Macs and GPU servers.</strong><br/>
  Runs many models on your own GPUs and shares them. No account, no cloud.
</p>

<p align="center">
  <a href="https://github.com/agentbusiness/CortexModelServer">github.com/agentbusiness/CortexModelServer</a> ·
  <a href="LICENSE.txt">Apache-2.0</a>
</p>

---

## Why CortexModelServer

Local models are good enough to do real work now, but each app loads its own. Two apps on one Mac
hold two copies of the same model, and the big GPU server in the corner sits idle while your laptop
runs out of memory. CortexModelServer turns every machine you own into one pool of models.

- **Many models at once.** Each GPU runs several models side by side, each with its own slots,
  context and cache, sized so they fit together.
- **Shared, not duplicated.** A model already in memory on a computer is used before another is
  loaded. Apps like [CortexCode](https://github.com/agentbusiness/CortexCode) and
  [CortexTerm](https://github.com/agentbusiness/CortexTerm) borrow from it instead of loading their own.
- **One window for the whole cluster.** Every model server you own shows up as a card with live
  metrics. Click one to change what it runs, from any other machine.
- **Everything stays yours.** No account, no API key, no telemetry. Models run on your GPUs and
  requests stay on your machines.

## Lanes

A model running on a GPU is a **lane**: one model, on one engine, with its own slots, context
length and prompt cache. A machine runs as many lanes as fit, of any engine, and the same model can
run in more than one lane (one per GPU, say).

- **Runs what your apps selected.** A computer starts with no lanes. When a Cortex app runs on it,
  the model selected in that app becomes a lane, if it fits this computer, so the app loads none of
  its own. A selection is kept while the app is closed. You can add any other lane by hand.
- **Fitted to memory.** Each lane is sized from its weights, the cache for its context and the
  engine's buffers, and gets that as a hard limit. An engine never plans for the whole GPU when
  other lanes share it.
- **Yours to change.** Add, edit or remove any lane by hand. Lanes you set stay as you set them; the
  plan keeps running the rest.
- **Measured.** Calibration times each chat lane on this hardware: one request alone (prefill,
  decode, time to first token), then reloaded at each slot count its recipe and memory allow, and
  names the fewest slots that give close to the highest total. The chat lanes of each GPU then run
  at once, because what fits in memory is not what the GPU can compute together, and the
  recommendation comes from that run. A full run also tries each lane's drafting (MTP or
  DFlash), grows a shared context with its slots, measures extraction, moves lanes
  between GPUs when that gets more done, and applies all of it, to lanes set by hand too. Calls in progress finish first, but lanes reload several times, so run it
  when quiet. While a person waits on one lane, new other work on that GPU's other lanes waits its
  turn, for at most 30 seconds.

## Engines

Each model runs on the engine and with the exact settings it was tuned with, so it answers here the
way it does in the app it comes from. These settings are kept as **recipes**
(`plugins/serve/data/recipes/`).

| Engine | Runs on | Models |
|---|---|---|
| `model-uzu` | macOS (Metal) | Qwen3.8-27B (vision, speculative decoding), Qwen3.5-4B, LFM2.5-1.2B |
| `model-llama` | macOS (Metal), Linux (CUDA, Vulkan) | Gemma 4 E4B / E2B with MTP drafts, Gemma 4 E4B 131k, Qwen3.8-27B GGUF |
| `model-llama-agent` | macOS (Metal) | Gemma 4 E4B and 26B-A4B for agents, with saved slots |
| `model-bonsai` | macOS (Metal) | Ternary Bonsai 2 27B, for smaller Macs |
| `model-relex` | macOS, Linux | GLiNER-relex large: entities and relations for knowledge graphs |

Every engine is a patched build pinned to an exact revision, and each one is its own plugin.
`cortexmodelserver recipes` lists every recipe and whether it can run on this machine.

Weights live in `~/.cortex-models`, the folder every Cortex app shares. A model another app on the
machine already downloaded is not downloaded again.

## The mesh

Your machines form a private mesh. Every lane is lent to it unless you say otherwise, so a Cortex
app on any of your machines can use a model running on another.

- To pair a machine, type a four-word code on it. Each install has its own key, and only machines in
  your signed member list can connect.
- Connections are end-to-end encrypted (QUIC, TLS 1.3). They go direct on your home or office
  network, and through a relay when you're elsewhere. The relay can't read the traffic.
- Each request goes to the best lane that is free at that moment, then the lane is released again.
  A request someone is waiting for goes ahead of background work.
- Every model server in the mesh can administer every other one. Membership is the trust, the same
  as for borrowing models.

## The window

- **Top:** the whole cluster at a glance: GPU load and memory, tokens per second, millions of tokens
  per hour, requests and time to first token, slots and queue, servers and power.
- **Centre:** every model server in the mesh as a card, with live metrics and its lanes per GPU.
- **Right:** click a card to set up that server: pick a recipe and a GPU, set slots, context and
  cache, choose whether to lend it, add lanes that fit, apply the automatic plan, or calibrate.

## The command line

Everything the window does, the command line does too, on this machine or any other in the mesh
(`--on <server>`). `--json` prints the raw answer. Run them as `npx cortexmodelserver <command>`,
or install the command with `npm i -g cortexmodelserver`.

```sh
cortexmodelserver status [--on <server>]        # GPUs, engines, lanes, slots, tok/s
cortexmodelserver cluster                       # the whole mesh, added up
cortexmodelserver peers                         # every server
cortexmodelserver recipes                       # what can run here
cortexmodelserver plan [--reset] [--on <server>]
cortexmodelserver lane add "qwen3.8-27b uzu" --gpu 0 --slots 2 --ctx 32k --cache large
cortexmodelserver lane edit <lane> --slots 4
cortexmodelserver lane remove <lane>
cortexmodelserver models [list | remove <recipe>]
cortexmodelserver calibrate [<lane>]
```

## Service mode

A GPU server has no one sitting at it. Service mode runs CortexModelServer without a window,
started by the operating system: a login service on macOS, and a systemd unit on Linux that starts
at boot and keeps running after SSH logout. It updates itself and restarts onto a new release only
once its lanes are idle.

```sh
npx cortexmodelserver service join <four-word-code>
```

installs the service, joins the mesh and applies the plan in one step. See [SERVICE.md](SERVICE.md)
for the other service commands, logs on each platform, updates and uninstalling.

## Who it is for

- People with more than one machine who would rather run each model once and share it.
- Anyone with a Linux GPU server who wants it working for the apps on their laptop.
- Teams that want local models on their own hardware, with no cloud in between.
- Users of CortexCode and CortexTerm who want those apps to use bigger models than their laptop
  can hold.

## Requirements

- macOS on Apple silicon, or Linux (x64 or arm64) with an NVIDIA (CUDA) or Vulkan GPU. A machine
  without a GPU serves no model.
- Disk space for the models (downloaded when a lane first needs them).

## Install

```sh
npx cortexmodelserver
```

The first run downloads the app for your machine from this repository's releases, checks it
against the checksum in the release catalog and starts it; later runs start the installed app at
once, and the app updates itself. It also adds CortexModelServer to your applications (on macOS
`/Applications`, or `~/Applications` when that is not writable; on Linux the desktop's app list), so
later you can open it from there without a terminal. The app runs on its own: you can close the
terminal, and running the command again while it runs only says so.

On a GPU server without a screen, install it as a service instead (see [Service mode](#service-mode)).

Updates come by themselves: plugins in the background, and a new frame waits for *Restart to
update* (a service restarts onto it by itself once its lanes are idle).

## Releasing

Everything is built on a Mac with Apple silicon; Linux builds run in Docker on it. Each plugin is
released on its own, so a change to one plugin builds only that plugin. The frame is a small core:
the frame and its Electron shell. `scripts/release/digest.mjs` says what each part is built from.

| Script | What it releases |
|---|---|
| `scripts/release.sh` | **whatever changed since the last release**, in order: frame, plugins, app, npx (`--plan` says what, `--dry-run` builds without uploading) |
| `scripts/release-plugin.sh <plugin> [version]` | one plugin (`--changed`: every plugin whose source changed) |
| `scripts/release-frame.sh [version]` | the frame and its Electron shell |
| `scripts/release-app.sh [version]` | the full app for new users (`.tar.gz`; `.dmg` with `RELEASE_DMG=yes`), from the released parts |
| `scripts/release-npm.sh [version]` | the `npx cortexmodelserver` launcher |

Without a version each takes the next one after the released one: three numbers, each 0 to 99
(`0.3.99` is followed by `0.4.0`). Every script takes `--dry-run` (build into `artifacts/`, upload
nothing). Configuration: `.env.release` (see `.env.release.example`).

## Status

CortexModelServer is in early development. Expect fast changes and rough edges.

## License

Apache License 2.0. See [LICENSE.txt](LICENSE.txt).

CortexModelServer's records live in [IronGraph](https://github.com/agentbusiness/IronGraph).
Third-party components, including the vendored llama.cpp, uzu and candle engines, keep their own
licenses.
