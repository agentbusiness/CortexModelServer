//! `[backend.config]` of plugin.toml merged with the lane's own config (HelloAck.config; the frame
//! config's values win).
//!
//! CortexModelServer patch: CortexAgent's fitter decides slots, context, KV type, checkpoints and the
//! prompt cache per machine. Every key below defaults to "let the fit decide" (0 or ""); a lane sets
//! one to fix or cap it. ct's `ctx`, `parallel`, `ubatch` and fixed `kv_type` defaults are gone, and
//! ct's `borrow` does not exist here (a lane never borrows a member's model).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// CortexAgent catalog id (`gemma_4_e4b_it_q4km`, `gemma_4_26b_a4b_it_q4km`); "" = CortexAgent's
    /// recommendation for this machine.
    pub variant: String,
    /// Slots (`--parallel`, 1..=16); 0 = fitted (CortexAgent: from 16, halved until each gets 32k,
    /// at least 2). A value is CortexAgent's `CORTEXAGENT_LLAMA_SLOTS`.
    pub slots: u16,
    /// Ceiling on the fitted slot count when `slots` is 0 (default 16).
    pub max_slots: u16,
    /// Ceiling on the shared pool's context (`--ctx-size`); 0 = the model's native context.
    pub ctx: u32,
    /// Ceiling on the host-RAM prompt cache in MiB (`--cache-ram`); 0 = CortexAgent's 6 GiB. Never
    /// above 6 GiB.
    pub cache_ram: u32,
    /// Ceiling on context checkpoints per slot (`--ctx-checkpoints`); 0 = CortexAgent's 16. Never
    /// above 16.
    pub checkpoints: u32,
    /// "" = fitted (f16 when it reaches the native context, else q8_0 with flash attention);
    /// "f16" | "q8_0" = CortexAgent's `CORTEXAGENT_KV_CACHE_TYPE`.
    pub kv_type: String,
    /// Device memory ceiling in bytes (CortexAgent's `CORTEXAGENT_DEVICE_MEMORY_LIMIT_BYTES`); 0 =
    /// none. 3 GiB are reserved below it as in CortexAgent.
    pub memory_limit: u64,
    /// Tokens the MTP head drafts ahead (`--spec-draft-n-max`), CortexAgent's 4.
    pub draft_max: u16,
    /// Layers offloaded (`--n-gpu-layers`), CortexAgent's 99.
    pub gpu_layers: u16,
    /// Physical batch (`--ubatch-size`), CortexAgent's 2048.
    pub ubatch: u32,
    /// Load the vision projector (`--mmproj`), as CortexAgent does.
    pub vision: bool,
    /// Unload after this long without a request; 0 = never (CortexAgent keeps its model loaded).
    pub idle_unload_s: u64,
    /// The llama-server binary (default: `llama/llama-server` beside this executable).
    pub server_bin: Option<String>,
    /// Where models live (default `~/.cortex-models`, the folder every Cortex app shares).
    pub models_dir: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            variant: String::new(),
            slots: 0,
            max_slots: 0,
            ctx: 0,
            cache_ram: 0,
            checkpoints: 0,
            kv_type: String::new(),
            memory_limit: 0,
            draft_max: 4,
            gpu_layers: 99,
            ubatch: 2048,
            vision: true,
            idle_unload_s: 0,
            server_bin: None,
            models_dir: None,
        }
    }
}

impl Config {
    pub fn from_value(value: &serde_json::Value) -> Self {
        if value.is_null() {
            return Self::default();
        }
        serde_json::from_value(value.clone()).unwrap_or_else(|error| {
            tracing::warn!(%error, "plugin config not understood; using the defaults");
            Self::default()
        })
    }

    /// The KV type setting, parsed strictly as CortexAgent parses its environment variable.
    pub fn kv_cache(&self) -> Result<Option<crate::gguf_meta::KvCacheType>, String> {
        if self.kv_type.is_empty() {
            return Ok(None);
        }
        crate::gguf_meta::KvCacheType::parse(&self.kv_type).map(Some)
    }
}
