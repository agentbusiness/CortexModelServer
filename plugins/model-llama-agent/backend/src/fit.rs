// CortexModelServer: from CortexAgent (crates/model/src/llama_server.rs `LlamaServerConfig`,
// `MemoryBudget`, `ContextFit`, `fit_slots_to_memory`, `fit_to_memory`, `slots_fit` and their
// constants; crates/model/src/device.rs `physical_memory_bytes` / `available_memory_bytes`;
// crates/server/src/config.rs `model_memory_budget`). The arithmetic is CortexAgent's, line for
// line. What a lane may change is marked `CortexModelServer patch:`: each lane setting lowers one of
// CortexAgent's ceilings (never raises it), so a lane config can only make the fit smaller.
//
//! The per-machine fit: slots, context, KV cache type, checkpoints and prompt cache, decided once
//! when llama-server starts from the memory measured then.

use crate::gguf_meta::{self, KvCacheType};
use std::path::PathBuf;

pub const GIB: u64 = 1024 * 1024 * 1024;
/// Most host RAM llama-server may use for displaced processed prompts (`--cache-ram`).
pub const PROMPT_CACHE_MAX_BYTES: u64 = 6 * GIB;
/// Context checkpoints kept per slot when their RAM cost is small.
pub const MAX_CONTEXT_CHECKPOINTS: u32 = 16;
/// Fewest checkpoints a fit leaves a slot.
const MIN_CONTEXT_CHECKPOINTS: u32 = 2;
/// The share of RAM left after the weights that the prompt cache, and separately the checkpoints,
/// may take.
const HOST_CACHE_SHARE_DIVISOR: u64 = 4;
/// Smallest context a fit sets (llama.cpp's own `--fit-ctx` floor).
pub const MIN_FITTED_CONTEXT_TOKENS: u32 = 4096;
/// Fitted contexts are whole multiples of this.
const CONTEXT_GRANULARITY_TOKENS: u32 = 1024;
/// Minimum spacing between checkpoints (`--checkpoint-min-step`).
pub const CHECKPOINT_MIN_STEP_TOKENS: u32 = 2048;
/// KV cache type kept when it reaches the native context; else the fit takes `q8_0`.
pub const DEFAULT_KV_CACHE_TYPE: KvCacheType = KvCacheType::F16;
pub const MIN_SLOTS: u16 = 1;
/// Fewest slots the memory fit chooses.
pub const DEFAULT_SLOTS: u16 = 2;
/// Context each slot must be able to hold when the fit chooses the slot count.
pub const MIN_CONTEXT_PER_SLOT_TOKENS: u64 = 32_768;
/// Most decode slots.
pub const MAX_SLOTS: u16 = 16;
/// Compute buffers one llama.cpp context reserves beyond its attention mask.
const CONTEXT_COMPUTE_BASE_BYTES: u64 = 768 * 1024 * 1024;
const MASK_COPIES: u64 = 2;
const MASK_ELEMENT_BYTES: u64 = 2;
/// What the multimodal projector's encoder reserves beyond its weights.
const PROJECTOR_COMPUTE_BYTES: u64 = 384 * 1024 * 1024;
/// A quarter of physical RAM stays outside the fitted budget.
const RAM_SAFETY_DIVISOR: u64 = 4;
/// CortexAgent's `CORTEXAGENT_DEVICE_RESERVED_BYTES` default: subtracted from the device capacity.
pub const DEVICE_RESERVED_BYTES: u64 = 3 * GIB;

/// The memory a start may use, measured when it starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Accelerator bytes (Metal's recommended working set less the reserve); 0 = RAM alone bounds.
    pub device_bytes: u64,
    /// Host RAM available now less a quarter of physical RAM.
    pub ram_bytes: u64,
    /// Device allocations come out of host RAM (Apple unified memory, or a CPU device).
    pub unified: bool,
}

impl MemoryBudget {
    /// CortexAgent's `MemoryBudget::measure` with the device budget of its server config
    /// (`capacity.min(limit) - reserved`). `capacity`: the device's memory as llama.cpp reports it
    /// (Metal: recommendedMaxWorkingSetSize). CortexModelServer patch: `limit` is the lane's
    /// `memory_limit` (CortexAgent's `CORTEXAGENT_DEVICE_MEMORY_LIMIT_BYTES`), 0 = none.
    pub fn measure(capacity: u64, limit: u64, unified: bool) -> Option<Self> {
        let physical = physical_memory_bytes()?;
        let available = available_memory_bytes()?;
        let limit = if limit == 0 { u64::MAX } else { limit };
        let device_bytes = capacity.min(limit).checked_sub(DEVICE_RESERVED_BYTES).filter(|b| *b > 0).unwrap_or(0);
        let ram = available.saturating_sub(physical / RAM_SAFETY_DIVISOR);
        Some(Self { device_bytes, ram_bytes: ram, unified })
    }
}

/// What the fit chose and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextFit {
    pub slots: u16,
    pub context_tokens: u32,
    pub native_context_tokens: u32,
    pub kv_cache_type: KvCacheType,
    pub context_checkpoints: u32,
    pub checkpoint_bytes: u64,
    pub prompt_cache_bytes: u64,
    pub device_bytes: u64,
    pub ram_bytes: u64,
    pub budget: MemoryBudget,
    pub within_budget: bool,
}

impl ContextFit {
    pub fn log(&self) {
        if self.within_budget {
            tracing::info!(
                slots = self.slots,
                context_tokens = self.context_tokens,
                native_context_tokens = self.native_context_tokens,
                kv_cache_type = self.kv_cache_type.as_str(),
                context_checkpoints = self.context_checkpoints,
                checkpoint_bytes = self.checkpoint_bytes,
                prompt_cache_bytes = self.prompt_cache_bytes,
                device_bytes = self.device_bytes,
                ram_bytes = self.ram_bytes,
                device_budget_bytes = self.budget.device_bytes,
                ram_budget_bytes = self.budget.ram_bytes,
                "context fitted to measured memory"
            );
        } else {
            tracing::warn!(
                slots = self.slots,
                context_tokens = self.context_tokens,
                native_context_tokens = self.native_context_tokens,
                kv_cache_type = self.kv_cache_type.as_str(),
                device_bytes = self.device_bytes,
                ram_bytes = self.ram_bytes,
                device_budget_bytes = self.budget.device_bytes,
                ram_budget_bytes = self.budget.ram_bytes,
                "context could not be fitted within the measured memory; the engine reports the outcome"
            );
        }
    }
}

fn slots_fit(fit: &ContextFit, slots: u16) -> bool {
    let wanted = MIN_CONTEXT_PER_SLOT_TOKENS.saturating_mul(u64::from(slots)).min(u64::from(fit.native_context_tokens));
    fit.within_budget && u64::from(fit.context_tokens) >= wanted
}

/// CortexAgent's `LlamaServerConfig` (the fields the fit reads and sets).
#[derive(Clone, Debug)]
pub struct LaunchConfig {
    pub model_file: PathBuf,
    pub projector: Option<PathBuf>,
    pub drafter: Option<PathBuf>,
    pub slots: u16,
    /// Context of the one shared pool (`--ctx-size` with `--kv-unified`).
    pub context_tokens: u32,
    pub gpu_layers: u16,
    pub ubatch_tokens: u32,
    pub kv_cache_type: KvCacheType,
    pub context_checkpoints: u32,
    pub prompt_cache_bytes: u64,
    pub draft_max: u16,
    // CortexModelServer patch: the lane's ceilings (CortexAgent's constants by default).
    pub max_slots: u16,
    pub max_checkpoints: u32,
    pub max_prompt_cache_bytes: u64,
}

impl LaunchConfig {
    /// CortexAgent's `for_model`: native context, all layers offloaded, ubatch 2048, f16, 16
    /// checkpoints, 6 GiB prompt cache, draft max 4. `native` is the header's context length;
    /// CortexModelServer patch: `ctx_cap` (the lane's `ctx`, 0 = none) lowers it.
    pub fn for_model(model_file: PathBuf, ctx_cap: u32) -> Self {
        let native = gguf_meta::context_length(&model_file).unwrap_or(0);
        let context_tokens = if ctx_cap > 0 && native > 0 { native.min(ctx_cap) } else { native };
        Self {
            model_file,
            projector: None,
            drafter: None,
            slots: MIN_SLOTS,
            context_tokens,
            gpu_layers: 99,
            ubatch_tokens: 2048,
            kv_cache_type: DEFAULT_KV_CACHE_TYPE,
            context_checkpoints: MAX_CONTEXT_CHECKPOINTS,
            prompt_cache_bytes: PROMPT_CACHE_MAX_BYTES,
            draft_max: 4,
            max_slots: MAX_SLOTS,
            max_checkpoints: MAX_CONTEXT_CHECKPOINTS,
            max_prompt_cache_bytes: PROMPT_CACHE_MAX_BYTES,
        }
    }

    /// CortexAgent's `fit_slots_to_memory`: from MAX_SLOTS (CortexModelServer patch: the lane's
    /// cap), halve while the pool cannot give every slot 32k, never below two.
    pub fn fit_slots_to_memory(&mut self, budget: MemoryBudget, kv_cache: Option<KvCacheType>) -> ContextFit {
        let native = self.context_tokens;
        let floor = DEFAULT_SLOTS.min(self.max_slots.max(MIN_SLOTS));
        let mut slots = self.max_slots.clamp(MIN_SLOTS, MAX_SLOTS);
        loop {
            self.context_tokens = native;
            self.slots = slots;
            let fit = self.fit_to_memory(budget, kv_cache);
            if slots_fit(&fit, slots) || slots <= floor {
                return fit;
            }
            tracing::debug!(slots, context_tokens = fit.context_tokens, "the pool is short for this many slots; trying fewer");
            slots = (slots / 2).max(floor);
        }
    }

    /// CortexAgent's `fit_to_memory`, unchanged but for the lane ceilings (prompt cache and
    /// checkpoints) in place of its constants.
    pub fn fit_to_memory(&mut self, budget: MemoryBudget, kv_cache: Option<KvCacheType>) -> ContextFit {
        let native = self.context_tokens;
        let slots = u64::from(self.slots.max(1));
        let max_checkpoints = self.max_checkpoints.min(MAX_CONTEXT_CHECKPOINTS);
        let min_checkpoints = MIN_CONTEXT_CHECKPOINTS.min(max_checkpoints);
        let prompt_cache_max = self.max_prompt_cache_bytes.min(PROMPT_CACHE_MAX_BYTES);
        let Some(geometry) = gguf_meta::kv_geometry(&self.model_file) else {
            self.kv_cache_type = kv_cache.unwrap_or(DEFAULT_KV_CACHE_TYPE);
            self.context_checkpoints = self.context_checkpoints.min(max_checkpoints);
            self.prompt_cache_bytes = self.prompt_cache_bytes.min(prompt_cache_max);
            return ContextFit {
                slots: self.slots,
                context_tokens: native,
                native_context_tokens: native,
                kv_cache_type: self.kv_cache_type,
                context_checkpoints: self.context_checkpoints,
                checkpoint_bytes: 0,
                prompt_cache_bytes: self.prompt_cache_bytes,
                device_bytes: 0,
                ram_bytes: 0,
                budget,
                within_budget: false,
            };
        };
        let file_bytes = |path: Option<&PathBuf>| path.and_then(|path| std::fs::metadata(path).ok()).map_or(0, |metadata| metadata.len());
        let ubatch = u64::from(self.ubatch_tokens);
        let contexts = 1 + u64::from(self.drafter.is_some());
        let mask_per_token = contexts * MASK_COPIES * MASK_ELEMENT_BYTES * ubatch;
        let projector_bytes = self.projector.as_ref().map_or(0, |path| file_bytes(Some(path)).saturating_add(PROJECTOR_COMPUTE_BYTES));
        let fixed_device = file_bytes(Some(&self.model_file))
            .saturating_add(projector_bytes)
            .saturating_add(file_bytes(self.drafter.as_ref()))
            .saturating_add(contexts * CONTEXT_COMPUTE_BASE_BYTES);
        let device_limit = if budget.device_bytes == 0 { u64::MAX } else { budget.device_bytes };
        let ram_after_fixed = if budget.unified { budget.ram_bytes.saturating_sub(fixed_device) } else { budget.ram_bytes };
        let host_share = ram_after_fixed / HOST_CACHE_SHARE_DIVISOR;
        let prompt_cache_bytes = prompt_cache_max.min(host_share);

        let plan = |cache: KvCacheType| {
            let native = u64::from(native);
            let per_token = geometry.bytes_per_token(cache);
            let checkpoint_bytes = geometry.checkpoint_bytes(native, ubatch, cache);
            let checkpoints = if checkpoint_bytes == 0 {
                max_checkpoints
            } else {
                u32::try_from(host_share / checkpoint_bytes.saturating_mul(slots)).unwrap_or(u32::MAX).clamp(min_checkpoints, max_checkpoints)
            };
            let host_bytes = prompt_cache_bytes.saturating_add(checkpoint_bytes.saturating_mul(u64::from(checkpoints)).saturating_mul(slots));
            let device_at = |context: u64| {
                fixed_device
                    .saturating_add(per_token.saturating_mul(context))
                    .saturating_add(geometry.checkpoint_bytes(context, ubatch, cache).saturating_mul(slots))
                    .saturating_add(mask_per_token.saturating_mul(context))
            };
            let ram_at = |context: u64| {
                let device = if budget.unified { device_at(context) } else { 0 };
                device.saturating_add(host_bytes)
            };
            let fixed_slots = fixed_device.saturating_add(checkpoint_bytes.saturating_mul(slots));
            let device_room = device_limit.saturating_sub(fixed_slots);
            let ram_fixed = if budget.unified { fixed_slots } else { 0 }.saturating_add(host_bytes);
            let ram_room = budget.ram_bytes.saturating_sub(ram_fixed);
            let room = device_room.min(ram_room);
            let affordable = room / per_token.saturating_add(mask_per_token);
            let granularity = u64::from(CONTEXT_GRANULARITY_TOKENS);
            let fitted = native.min(affordable / granularity * granularity);
            let floor = u64::from(MIN_FITTED_CONTEXT_TOKENS).min(native);
            let context = fitted.max(floor);
            let within = fitted >= floor || fitted == native;
            (context, checkpoints, checkpoint_bytes, device_at(context), ram_at(context), within)
        };

        let preferred = kv_cache.unwrap_or(DEFAULT_KV_CACHE_TYPE);
        let mut cache = preferred;
        let mut chosen = plan(preferred);
        if kv_cache.is_none() && chosen.0 < u64::from(native) {
            let wider = plan(KvCacheType::Q8_0);
            if wider.0 > chosen.0 {
                cache = KvCacheType::Q8_0;
                chosen = wider;
            }
        }
        let (context, checkpoints, checkpoint_bytes, device_bytes, ram_bytes, within_budget) = chosen;
        self.context_tokens = u32::try_from(context).unwrap_or(native);
        self.kv_cache_type = cache;
        self.context_checkpoints = checkpoints;
        self.prompt_cache_bytes = prompt_cache_bytes;
        ContextFit {
            slots: self.slots,
            context_tokens: self.context_tokens,
            native_context_tokens: native,
            kv_cache_type: cache,
            context_checkpoints: checkpoints,
            checkpoint_bytes,
            prompt_cache_bytes,
            device_bytes,
            ram_bytes,
            budget,
            within_budget,
        }
    }

    /// CortexAgent's `arguments()`, in its order. `slot_dir`: `--slot-save-path`
    /// (CortexModelServer patch: under this lane's CORTEX_PLUGIN_DATA instead of
    /// ~/.cortexagent/cache/inference/llama).
    pub fn arguments(&self, slot_dir: &std::path::Path) -> Vec<String> {
        let mut args: Vec<String> = Vec::new();
        let mut push = |value: String| args.push(value);
        push("--model".into());
        push(self.model_file.display().to_string());
        push("--parallel".into());
        push(self.slots.to_string());
        push("--kv-unified".into());
        push("--ctx-size".into());
        push(self.context_tokens.to_string());
        push("--n-gpu-layers".into());
        push(self.gpu_layers.to_string());
        push("--ubatch-size".into());
        push(self.ubatch_tokens.to_string());
        push("--cache-type-k".into());
        push(self.kv_cache_type.as_str().into());
        push("--cache-type-v".into());
        push(self.kv_cache_type.as_str().into());
        if self.kv_cache_type.requires_flash_attention() {
            push("--flash-attn".into());
            push("on".into());
        }
        push("--cache-ram".into());
        // Never `0` from rounding: llama.cpp reads 0 as "disable the cache".
        push((self.prompt_cache_bytes / (1024 * 1024)).max(1).to_string());
        push("--ctx-checkpoints".into());
        push(self.context_checkpoints.to_string());
        push("--checkpoint-min-step".into());
        push(CHECKPOINT_MIN_STEP_TOKENS.to_string());
        push("--cache-reuse".into());
        push("0".into());
        push("--no-prefill-assistant".into());
        push("--no-context-shift".into());
        push("--jinja".into());
        push("--slot-save-path".into());
        push(slot_dir.display().to_string());
        if let Some(projector) = &self.projector {
            push("--mmproj".into());
            push(projector.display().to_string());
        }
        if let Some(drafter) = &self.drafter {
            push("--spec-type".into());
            push("draft-mtp".into());
            push("--spec-draft-model".into());
            push(drafter.display().to_string());
            push("--spec-draft-n-max".into());
            push(self.draft_max.to_string());
        }
        args
    }
}

#[cfg(target_os = "macos")]
fn sysctl<T: Default + Copy>(name: &std::ffi::CStr) -> Option<T> {
    let mut value = T::default();
    let mut length = std::mem::size_of::<T>();
    // SAFETY: NUL-terminated key, writable output whose size is passed, no new value.
    let status = unsafe { libc::sysctlbyname(name.as_ptr(), (&raw mut value).cast(), &mut length, std::ptr::null_mut(), 0) };
    (status == 0 && length == std::mem::size_of::<T>()).then_some(value)
}

/// Physical host RAM (`hw.memsize`; Linux: physical pages).
pub fn physical_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        sysctl::<u64>(c"hw.memsize").filter(|b| *b > 0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        // SAFETY: sysconf takes no pointers.
        let (pages, size) = unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE)) };
        u64::try_from(pages).ok()?.checked_mul(u64::try_from(size).ok()?).filter(|b| *b > 0)
    }
}

/// Host RAM available now: macOS `kern.memorystatus_level` percent of physical RAM; Linux
/// `MemAvailable`.
pub fn available_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let level = sysctl::<i32>(c"kern.memorystatus_level")?;
        let level = u64::try_from(level).ok().filter(|level| (1..=100).contains(level))?;
        Some(physical_memory_bytes()? / 100 * level)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kib: u64 = meminfo.lines().find_map(|line| line.strip_prefix("MemAvailable:"))?.trim().strip_suffix("kB")?.trim().parse().ok()?;
        Some(kib.saturating_mul(1024))
    }
}
