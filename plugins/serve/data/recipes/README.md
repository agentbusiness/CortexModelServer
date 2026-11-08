# Recipes

One file per engine plugin (`<engine plugin id>.toml`). A recipe is one model on one engine with the
exact settings of the app it comes from. The serve plugin reads these files (shipped data, the app's
source, not records), plans lanes from them, and starts each lane as one frame instance of the engine
plugin with `[recipe.config]` merged with the lane's own settings (mapped through `[recipe.keys]`).

```toml
[[recipe]]
id = "qwen3.8-27b-uzu"            # unique across all files
name = "Qwen3.8-27B"              # shown in the UI
engine = "model-uzu"              # engine plugin id
origin = "CortexCode"             # whose exact settings these are
kind = "chat"                     # chat | embed | extract
family = "qwen3.8"
tags = ["coding"]                 # mesh tags the lane is lent under
platforms = ["macos"]             # macos | linux
backends = ["metal"]              # metal | cuda | vulkan | cpu
repo = "trymirai/Qwen3.8-27B-M"   # Hugging Face repo in ~/.cortex-models
revision = "<40-char revision>"
weights_bytes = 0                 # bytes resident for weights (+drafter, projector)
compute_bytes = 0                 # compute buffers and runtime overhead
kv_bytes_per_token = 0            # KV per context token at the default kv type (0 for embed/extract)
min_gpu_bytes = 0                 # smallest GPU (or Mac unified memory) it is offered on
ctx_default = 32768               # lane context default (0 for embed/extract)
ctx_max = 262144
slots_default = 2
slots_max = 8
vision = false
speculative = "mtp"               # none | mtp | dflash | dflash2

[recipe.config]                   # the engine plugin's [backend.config] for this recipe
# variant = "medium"

[recipe.keys]                     # engine config key that carries each lane setting ("" = not settable)
slots = ""
ctx = ""
cache_ram = ""                    # prompt cache in host RAM, bytes or MiB as the engine reads it
checkpoints = ""
kv_type = ""
speculative = ""
idle_unload_s = ""
memory_limit = ""
```
