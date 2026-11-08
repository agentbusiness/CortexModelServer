use std::path::Path;

use embed_common::utils::error::{Error, Result};

/// The parts of a ModernBERT `config.json` the encoder is built from.
#[derive(Debug, Clone, serde::Deserialize)]
pub(super) struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    pub local_attention: usize,
    pub local_rope_theta: f64,
    pub hidden_activation: String,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    pub norm_bias: bool,
}

impl ModernBertConfig {
    /// Reads `config.json` and checks it describes an encoder this
    /// implementation runs: exact GELU, no bias terms, whole heads.
    pub(super) fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            Error::Internal(format!(
                "Failed to read encoder config {}: {e}",
                path.display()
            ))
        })?;
        let config: Self = serde_json::from_str(&text).map_err(|e| {
            Error::Internal(format!(
                "Failed to parse encoder config {}: {e}",
                path.display()
            ))
        })?;
        let unsupported = |what: &str| {
            Err(Error::Internal(format!(
                "Encoder config {} is not supported: {what}",
                path.display()
            )))
        };
        if config.hidden_activation != "gelu" {
            return unsupported("hidden_activation must be gelu");
        }
        if config.attention_bias || config.mlp_bias || config.norm_bias {
            return unsupported("bias terms are not supported");
        }
        if config.num_attention_heads == 0
            || !config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
            || !(config.hidden_size / config.num_attention_heads).is_multiple_of(2)
        {
            return unsupported("hidden_size must split into heads of even size");
        }
        if config.global_attn_every_n_layers == 0 || config.num_hidden_layers == 0 {
            return unsupported("the encoder needs layers and a global attention period");
        }
        Ok(config)
    }
}
