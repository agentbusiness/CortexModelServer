//! `[backend.config]` of plugin.toml, with the lane's own settings merged over it (the frame's
//! hello ack carries the result).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Unload GLiNER-relex (end its worker process, freeing the GPU memory) after this long
    /// without a request; 0 keeps it loaded. The knowledge app's lanes stop after 5 minutes.
    pub idle_unload_s: u64,
    /// Entity threshold for a passage that names none (the knowledge app's ingest uses 0.3).
    pub threshold: f32,
    /// Relation threshold for a passage that names none (the knowledge app's ingest uses 0.5).
    pub relation_threshold: f32,
    /// The encoder's weights: `f16` (the knowledge app's default, half the memory) or `f32`. The heads stay
    /// `f32` and the BiLSTM runs on the CPU either way.
    pub encoder_dtype: String,
    /// Where models live (default `~/.cortex-models`, the folder every Cortex app shares).
    pub models_dir: Option<String>,
    /// Download the model when it is missing (otherwise `model.extract` answers unavailable).
    pub download: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self { idle_unload_s: 300, threshold: 0.3, relation_threshold: 0.5, encoder_dtype: "f16".into(), models_dir: None, download: true }
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
