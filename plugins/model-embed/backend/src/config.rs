//! `[backend.config]` of plugin.toml, with the lane's own settings merged over it (the frame's
//! hello ack carries the result).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// The model this lane serves: `bekko-embedding-v1-a8m` (`store::BEKKO`), the only one.
    pub model: String,
    /// Texts per encoder pass. Texts are ordered by length first, so a pass pads only to its
    /// longest text.
    pub batch_size: usize,
    /// CPU threads (the knowledge app's semantic embedder uses 4; capped at half the cores).
    pub threads: usize,
    /// Run ModernBERT on the GPU (Metal) when this machine has one.
    pub prefer_gpu: bool,
    /// Tokens read per text at most (`CORTEX_EMBED_MAX_INPUT_TOKENS`; ModernBERT is also capped
    /// at its context). A request's `truncate` may cut shorter.
    pub max_input_tokens: usize,
    /// Unload the model after this long without a request; 0 keeps it loaded.
    pub idle_unload_s: u64,
    /// Where models live (default `~/.cortex-models`, the folder every Cortex app shares).
    pub models_dir: Option<String>,
    /// Download the model when it is missing (otherwise `model.embed` answers unavailable).
    pub download: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "bekko-embedding-v1-a8m".into(),
            batch_size: 32,
            threads: 4,
            prefer_gpu: true,
            max_input_tokens: 512,
            idle_unload_s: 300,
            models_dir: None,
            download: true,
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
}
