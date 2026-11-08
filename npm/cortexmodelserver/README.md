# cortexmodelserver

```sh
npx cortexmodelserver
```

Starts [CortexModelServer](https://github.com/agentbusiness/CortexModelServer), the model server of
the Cortex apps: it runs models on this computer's GPUs in lanes (one model on one of the patched
engines, with its own slots, context and cache) and lends them to the mesh. The first run downloads
the app for this machine (Apple Silicon Macs; Linux x64 and arm64) from the official releases,
checks it against the checksum in the release catalog and unpacks it into
`~/.cortexmodelserver/app/`. Later runs start the installed app at once; the app updates its frame
and plugins itself. The app runs on its own: you can close the terminal (its start output goes to
`~/.cortexmodelserver/logs/start.log`); running the command again while it runs only says so.

Each start also puts a CortexModelServer starter where the system keeps apps: `CortexModelServer.app`
in `/Applications` on macOS (`~/Applications` when that is not writable), a
`cortexmodelserver.desktop` entry in `~/.local/share/applications` on Linux.

Arguments go to the app: `npx cortexmodelserver --headless`, `npx cortexmodelserver --safe`.

## Background service (macOS and Linux)

```sh
npx cortexmodelserver service join <four-word-code>
```

installs the service (a LaunchAgent on macOS, a systemd unit on Linux that starts at boot and
survives SSH logout), joins the mesh with the code from Settings > Mesh on another computer and
applies the plan. `service` alone installs it without joining; `service allow <computer>` lets a
computer of the mesh work here; `service status` says how it is; `service off` stops and removes it.
`runner` is the same command. Full details: [SERVICE.md](https://github.com/agentbusiness/CortexModelServer/blob/main/SERVICE.md).

## Every day

Every command calls the running server's methods, as the window does; `--on <server>` acts on any
cortexmodelserver of the mesh, `--json` prints the raw answer.

```
cortexmodelserver status [--on <server>]        # GPUs, engines, lanes, slots, tok/s
cortexmodelserver cluster                       # the whole mesh, added up
cortexmodelserver peers                         # every server
cortexmodelserver recipes                       # what can run here
cortexmodelserver plan [--reset]                # the plan (it runs by itself); --reset: back to it
cortexmodelserver lane add "qwen3.8-27b uzu" --gpu 0 --slots 2 --ctx 32k --cache large
cortexmodelserver lane edit <lane> --slots 4
cortexmodelserver lane remove <lane>
cortexmodelserver models [list | remove <recipe>]
cortexmodelserver calibrate [<lane>]
cortexmodelserver service off
```

(Installed with `npm i -g cortexmodelserver`, or as `npx cortexmodelserver <command>`.)
