//! ModernBERT sentence encoder on candle.
//!
//! Runs a ModernBERT encoder from its `safetensors` weights with the
//! Hugging Face tokenizer, mean-pools the token states over the attention mask
//! and L2-normalizes the result. On macOS it runs on Metal when a GPU is
//! available and preferred, and on the CPU otherwise.

mod candle_error;
mod config;
mod embed_texts;
mod encoder;
mod forward;
mod from_files;
mod injection_guard;
mod load_encoder;
mod run_chunk;
mod select_device;

use candle_core::Device;
use embed_common::utils::error::Result;
use tokenizers::Tokenizer;

use super::EmbeddingModel;

pub use injection_guard::{INJECTION_GUARD_FLAG_LOGIT, InjectionGuard, InjectionVerdict};

/// A ModernBERT sentence-embedding model running on candle.
///
/// Load with [`from_files()`](Self::from_files).
pub struct ModernBertEmbeddingModel {
    encoder: encoder::Encoder,
    tokenizer: Tokenizer,
    device: Device,
    pool: rayon::ThreadPool,
    name: String,
    dimensions: usize,
    batch_size: usize,
}

impl std::fmt::Debug for ModernBertEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModernBertEmbeddingModel")
            .field("name", &self.name)
            .field("dimensions", &self.dimensions)
            .field("batch_size", &self.batch_size)
            .field("execution_target", &self.execution_target())
            .finish()
    }
}

impl EmbeddingModel for ModernBertEmbeddingModel {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_texts(texts)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn backend(&self) -> &'static str {
        "candle"
    }

    fn execution_target(&self) -> &'static str {
        if self.device.is_metal() { "gpu" } else { "cpu" }
    }
}
