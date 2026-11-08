# Service mode

Service mode runs CortexModelServer **without a window**, started by the operating system: at login
on macOS, at boot on Linux (and it survives SSH logout). It is the same frame and the same plugins as
the app with its window, started with `--service` (`--runner` is the same flag): the window is not
started, the updater stays on (even when the desktop's updater was turned off; its saved preference
is kept) and restarts the frame onto a new release when the lanes are idle. Nothing else changes:
the engines serve their lanes exactly as they do in the window.

## Install and join

```sh
npx cortexmodelserver service join <four-word-code>
```

1. installs the service and starts it (the installed app, downloaded first on a machine that has
   none; the service never runs from npx's temporary cache),
2. joins the mesh with the code that Settings > Mesh shows on another computer (`mesh.pair.join`),
3. applies the plan (`serve.plan.run`): lanes made from this computer's GPUs and from what the Cortex
   apps on this computer use. A server that follows the plan (`auto`, the default) keeps following
   it; one whose lanes were set by hand gets the plan once.

The other service commands:

```sh
npx cortexmodelserver service                    # install and start it, without joining
npx cortexmodelserver service allow <computer>   # let that computer of the mesh work here
npx cortexmodelserver service status             # the service manager, the frame, the mesh, the lanes
npx cortexmodelserver service off                # stop and remove it
```

Joining puts the server in the mesh: its lent lanes are offered to every member at once, and every
member may open this one's window through the mesh and change its lanes (`--on`, below). A member
switched off in Settings > Mesh may not; `service allow <computer>` (the name the other computer
shows in Settings > Mesh) switches it on again, the switch a service has no window for.

There is one service per OS account. A CortexModelServer already running for the same
`CORTEX_HOME` with its window is not taken over: quit it first (`service join` then joins through the
running app instead of installing the service).

## What it serves

Every lane is one recipe (a model on one of the engines, with the exact settings of the Cortex app it
comes from) on one GPU, with its own slots, context and prompt cache. The engines this app ships:

| engine | runs on | models |
| --- | --- | --- |
| model-uzu | macOS | Qwen3.8-27B (CortexCode's uzu) |
| model-llama | macOS (Metal), Linux (CUDA, Vulkan, CPU) | Gemma 4 E4B / E2B, Qwen3.8-27B GGUF |
| model-llama-agent | macOS | Gemma 4 (CortexAgent's llama.cpp) |
| model-bonsai | macOS | Ternary Bonsai 2 27B |
| model-relex | macOS, Linux | GLiNER-relex (extraction) |
| model-embed | macOS, Linux | embeddings |

`cortexmodelserver recipes` lists every recipe with whether it can run here. Weights go to
`~/.cortex-models`, the folder every Cortex app shares: a model another app on this computer already
downloaded is not downloaded again.

Lanes are lent to the mesh unless set otherwise (`lane add ... --no-lend`): another computer's
Cortex app claims a lane tagged for its work, as it would claim its own model.

## Every day

Every command calls the running frame, with the same methods as the window; `--on <server>` acts on
any cortexmodelserver of the mesh (through `mesh.link`, a tunnel to that computer's frame), `--json`
prints the raw answer.

```
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

Contexts take `32k`, `128k` or a number of tokens; the prompt cache takes `off` (0), `small`
(1024 MiB), `large` (6144 MiB), a number of MiB, or `default` (the recipe's). A lane you add,
change or remove by hand stays that way while the plan keeps running the rest; `plan --reset` goes back to the plan.

`models remove` deletes the weights from `~/.cortex-models` on this computer only, after asking
(`--yes`): the folder is shared, so another Cortex app here would download them again. A model a lane
uses is not removed.

## The service on each platform

**macOS**: the LaunchAgent `~/Library/LaunchAgents/com.cortexmodelserver.service.plist`. It starts at
login and restarts on exit; it is a login service, not a boot daemon running before login. Its output
is in `~/.cortexmodelserver/logs/service.log`.

**Linux** with systemd: a normal user gets the user unit
`~/.config/systemd/user/cortexmodelserver.service` (or under `$XDG_CONFIG_HOME`), and the installer
enables the user's linger, so the service starts at boot and survives SSH logout. When the account
may not enable linger, the installer says the command an administrator runs instead of claiming
the service persists. Run as root, the installer makes the system unit
`/etc/systemd/system/cortexmodelserver.service` with root's own home. The output is in the journal:
`journalctl --user -u cortexmodelserver.service` (without `--user` for the system unit). A Linux
host without systemd runs `npx cortexmodelserver --service --foreground` under its own supervisor.

Both platforms also keep the frame's and the plugins' logs under `~/.cortexmodelserver/logs/`.

The unit stops with `KillMode=mixed`: the frame drains its lanes and stops its engines; an engine a
frame that died left behind is killed with it, so no stray engine holds GPU memory across a restart.
The GPUs the engines see follow `CUDA_VISIBLE_DEVICES` when it was set for the installer.

## Updates

Plugin releases install through the updater as in the window; a frame release is downloaded,
verified and staged, and the service restarts onto it once the lanes are idle (requests in flight
defer the restart). An incomplete or failed download never replaces the running frame.

## Uninstall

```sh
npx cortexmodelserver service off
```

stops and removes the LaunchAgent or the unit. The home (`~/.cortexmodelserver`: lanes, the plan,
the mesh membership) and the models (`~/.cortex-models`) are kept, so `service` brings the same
server back. The installer does not remove a user's linger setting, since other user services may
use it. To remove everything, also delete `~/.cortexmodelserver` (and the models no other Cortex app
uses, `cortexmodelserver models remove <recipe>`).

## Development

```sh
./cortexmodelserver service
```

brings the development build up to date (`./cortexmodelserver --build-only`) and points the service
at this checkout's built frame (`build/CortexModelServer.app/` on macOS, `build/cortexmodelserver/`
on Linux). The service carries `CORTEXMODELSERVER_DEV_ROOT` and `CORTEX_NO_HANDOVER=1`, so it never
replaces the checkout with a staged release, and starting at login or boot does not compile the
repository. After editing source, `./cortexmodelserver --build-only` refreshes it: changed plugins
are swapped into the running frame, and after a complete build whose frame, config or shell changed
the updater restarts the service onto it (the build's `build/.runner-restart` marker). Stop it with
`./cortexmodelserver service off`. The day-to-day commands work from the checkout too
(`./cortexmodelserver status`).
