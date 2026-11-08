//! `[backend.config]` of plugin.toml (the frame config's `[plugin.config]` wins).

use crate::catalog::Spec;
use serde::Deserialize;

// CortexModelServer patch: the server settings that differ between variants (ctx, parallel,
// ubatch, kv_type, draft_max, cache_ram, checkpoints, max_tokens, greedy) are optional here: a value
// given in the config wins, otherwise the chosen variant's own (`catalog::Defaults`, the exact
// settings of the app it comes from). `borrow` is read (ct ignored the key), and a lane never
// borrows (`engine`).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// "" = by memory (`catalog::for_memory`).
    pub variant: String,
    /// Gemma: context tokens of the one KV pool every slot shares (`--kv-unified`): any request
    /// may use all of it. Qwen: context tokens per slot (the pool is ctx * parallel).
    pub ctx: Option<u32>,
    pub parallel: Option<u32>,
    /// Prompt tokens read per step (`-ub`).
    pub ubatch: Option<u32>,
    /// f16 | q8_0 (q8_0 needs flash attention; checked once, f16 otherwise).
    pub kv_type: Option<String>,
    /// Tokens the drafter proposes per step at most (`--spec-draft-n-max`).
    pub draft_max: Option<u32>,
    /// Host-RAM prompt cache in MiB (`--cache-ram`).
    pub cache_ram: Option<u32>,
    /// Saved context states per slot (`--ctx-checkpoints`).
    pub checkpoints: Option<u32>,
    /// Answer cap in tokens (a call's own `max_tokens` is capped to it; a call without one gets it).
    pub max_tokens: Option<u32>,
    /// Greedy decoding whatever sampling the call asks for (temperature 0, top_k 1).
    pub greedy: Option<bool>,
    pub idle_unload_s: u64,
    /// Use a member's model through the mesh on a machine without a usable GPU (never on a lane).
    pub borrow: bool,
    /// The llama-server binary (default: `llama/llama-server` beside this executable).
    pub server_bin: Option<String>,
    /// Where models live (default `~/.cortex-models`, the folder every Cortex app shares).
    pub models_dir: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            variant: String::new(),
            ctx: None,
            parallel: None,
            ubatch: None,
            kv_type: None,
            draft_max: None,
            cache_ram: None,
            checkpoints: None,
            max_tokens: None,
            greedy: None,
            idle_unload_s: 600,
            borrow: true,
            server_bin: None,
            models_dir: None,
        }
    }
}

// CortexModelServer patch: the settings a variant runs with (the config's values over the
// variant's defaults).
#[derive(Debug, Clone)]
pub struct Settings {
    pub ctx: u32,
    pub parallel: u32,
    pub ubatch: u32,
    pub kv_type: String,
    pub draft_max: u32,
    /// None: llama-server's own default (the flag is not passed).
    pub cache_ram: Option<u32>,
    pub checkpoints: Option<u32>,
    pub max_tokens: Option<u32>,
    pub greedy: bool,
}

impl Settings {
    /// Context tokens of the whole KV pool (`-c`).
    pub fn pool(&self, spec: &Spec) -> u32 {
        if spec.profile.kv_unified() { self.ctx } else { self.ctx * self.parallel }
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

    // CortexModelServer patch: see `Settings`.
    pub fn settings(&self, spec: &Spec) -> Settings {
        let d = &spec.defaults;
        Settings {
            ctx: self.ctx.filter(|n| *n > 0).unwrap_or(d.ctx),
            parallel: self.parallel.filter(|n| *n > 0).unwrap_or(d.parallel),
            ubatch: self.ubatch.filter(|n| *n > 0).unwrap_or(d.ubatch),
            kv_type: self.kv_type.clone().filter(|t| !t.is_empty()).unwrap_or_else(|| d.kv_type.to_owned()),
            draft_max: self.draft_max.filter(|n| *n > 0).unwrap_or(d.draft_max),
            cache_ram: self.cache_ram.or(d.cache_ram),
            checkpoints: self.checkpoints.or(d.checkpoints),
            max_tokens: self.max_tokens.filter(|n| *n > 0).or(d.max_tokens),
            greedy: self.greedy.unwrap_or(d.greedy),
        }
    }
}
