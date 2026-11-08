//! The model folder's `gliner_config.json`: the GLiNER-side settings this lane
//! needs and the encoder's own configuration for candle's DeBERTa-v2.

use anyhow::{Context, Result, ensure};
use super::deberta::Config as DebertaConfig;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
pub(super) struct RelexConfig {
    /// Width of the GLiNER heads (the encoder's hidden size is projected to it).
    pub hidden_size: usize,
    /// Longest span in words.
    pub max_width: usize,
    /// Token id of the entity-kind marker (`<<ENT>>`).
    pub class_token_index: u32,
    /// Token id of the relation marker (`<<REL>>`).
    pub rel_token_index: u32,
    pub ent_token: String,
    pub rel_token: String,
    pub sep_token: String,
    pub span_mode: String,
    pub subtoken_pooling: String,
    pub embed_ent_token: bool,
    pub embed_rel_token: bool,
    pub num_rnn_layers: usize,
    pub encoder_config: DebertaConfig,
}

pub(super) fn load(dir: &Path) -> Result<RelexConfig> {
    let path = dir.join("gliner_config.json");
    let config: RelexConfig = serde_json::from_slice(&std::fs::read(&path).with_context(|| path.display().to_string())?)
        .with_context(|| format!("{}", path.display()))?;
    // The port covers the architecture of gliner-relex-large-v1.0; any other
    // setting is refused rather than run wrong.
    ensure!(config.span_mode == "markerV0", "span_mode {} is not markerV0", config.span_mode);
    ensure!(config.subtoken_pooling == "first", "subtoken_pooling {} is not first", config.subtoken_pooling);
    ensure!(config.embed_ent_token && config.embed_rel_token, "the marker token itself must carry the prompt");
    ensure!(config.num_rnn_layers == 1, "{} rnn layers", config.num_rnn_layers);
    Ok(config)
}
