//! In-process embedding generation for zero-API-dependency vector workflows.
//!
//! The `embed` feature carries the semantic embedder (ModernBERT on candle).
//! The `onnx-embed` feature adds ONNX models on the pure-Rust RTen runtime, so
//! you can generate embeddings directly in your application without calling
//! external APIs or shipping a separate native inference runtime.
//!
//! # Example
//!
//! ```no_run
//! use crate::embedding::{EmbeddingModel, OnnxEmbeddingModel};
//!
//! # fn main() -> embed_common::utils::error::Result<()> {
//! let model = OnnxEmbeddingModel::from_files(
//!     "my-model",
//!     "model.onnx",
//!     "tokenizer.json",
//! )?;
//! let vectors = model.embed(&["Hello world", "Graph databases"])?;
//! assert_eq!(vectors.len(), 2);
//! assert_eq!(vectors[0].len(), model.dimensions());
//! # Ok(())
//! # }
//! ```

mod config;
#[cfg(feature = "onnx-embed")]
mod download;
#[cfg(feature = "onnx-embed")]
mod huggingface_hub_cache_dir;
#[cfg(feature = "onnx-embed")]
mod model_config;
mod modernbert;
#[cfg(feature = "onnx-embed")]
mod onnx;
mod semantic_model;
mod work_gate;

pub use config::EmbeddingOptions;
#[cfg(feature = "onnx-embed")]
pub use model_config::EmbeddingModelConfig;
pub use modernbert::{
    INJECTION_GUARD_FLAG_LOGIT, InjectionGuard, InjectionVerdict, ModernBertEmbeddingModel,
};
#[cfg(feature = "onnx-embed")]
pub use onnx::OnnxEmbeddingModel;
pub use semantic_model::{
    SEMANTIC_EMBEDDING_CONFIG_FILE, SEMANTIC_EMBEDDING_FILES, SEMANTIC_EMBEDDING_MODEL_NAME,
    SEMANTIC_EMBEDDING_REPO_ID, SEMANTIC_EMBEDDING_REVISION, SEMANTIC_EMBEDDING_TOKENIZER_FILE,
    SEMANTIC_EMBEDDING_WEIGHTS_FILE, SemanticModelFiles, local_semantic_model_files,
    provision_semantic_model_files,
};
pub use work_gate::{
    LensEmbeddingScopeGuard, NonLensEmbeddingWorkGuard, begin_lens_embedding_scope,
    begin_non_lens_embedding_work, lens_embedding_pause_requested, lens_embedding_scope_is_active,
    try_begin_non_lens_embedding_work,
};

use embed_common::utils::error::Result;
use std::sync::{Arc, OnceLock, RwLock};

type EmbeddingProgressHandler = Arc<dyn Fn(String) + Send + Sync>;

static EMBEDDING_PROGRESS_HANDLER: OnceLock<RwLock<Option<EmbeddingProgressHandler>>> =
    OnceLock::new();

/// Installs the process-wide progress observer used by the embedding host.
pub fn set_embedding_progress_handler(handler: Option<EmbeddingProgressHandler>) {
    let state = EMBEDDING_PROGRESS_HANDLER.get_or_init(|| RwLock::new(None));
    *state
        .write()
        .expect("embedding progress handler lock poisoned") = handler;
}

pub(crate) fn update_embedding_progress(stage: impl Into<String>) {
    let Some(state) = EMBEDDING_PROGRESS_HANDLER.get() else {
        return;
    };
    let handler = state
        .read()
        .expect("embedding progress handler lock poisoned")
        .clone();
    if let Some(handler) = handler {
        handler(stage.into());
    }
}

/// Trait for embedding models that convert text to vectors.
///
/// Implement this trait for custom embedding backends (API-based, local
/// inference, etc.). With `onnx-embed`, `OnnxEmbeddingModel` runs ONNX models
/// locally via the pure-Rust RTen runtime.
pub trait EmbeddingModel: Send + Sync {
    /// Generates embeddings for a batch of texts.
    ///
    /// Returns one vector per input text, each with [`EmbeddingModel::dimensions()`] elements.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Returns the dimensionality of the embedding vectors.
    fn dimensions(&self) -> usize;

    /// Returns the model name.
    fn name(&self) -> &str;

    /// Returns the backend used by this model.
    fn backend(&self) -> &'static str {
        "custom"
    }

    /// Returns the execution target used by this model.
    fn execution_target(&self) -> &'static str {
        "cpu"
    }

    /// Returns whether concurrent calls to [`EmbeddingModel::embed`] are safe.
    fn allows_concurrent_embed(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A deterministic mock embedding model for testing.
    /// Generates vectors based on text length and character sum.
    struct MockEmbeddingModel {
        dims: usize,
    }

    impl MockEmbeddingModel {
        fn new(dims: usize) -> Self {
            Self { dims }
        }
    }

    impl EmbeddingModel for MockEmbeddingModel {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let seed = t.bytes().map(|b| b as f32).sum::<f32>();
                    (0..self.dims)
                        .map(|i| ((seed + i as f32) * 0.01).sin())
                        .collect()
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            self.dims
        }

        fn name(&self) -> &str {
            "mock-embedding"
        }
    }

    #[test]
    fn trait_returns_correct_dimensions() {
        let model = MockEmbeddingModel::new(384);
        let vecs = model.embed(&["hello"]).unwrap();
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].len(), 384);
        assert_eq!(model.dimensions(), 384);
    }

    #[test]
    fn trait_batch_returns_one_vec_per_input() {
        let model = MockEmbeddingModel::new(128);
        let texts = &["one", "two", "three", "four", "five"];
        let vecs = model.embed(texts).unwrap();
        assert_eq!(vecs.len(), 5);
        for v in &vecs {
            assert_eq!(v.len(), 128);
        }
    }

    #[test]
    fn trait_empty_input_returns_empty() {
        let model = MockEmbeddingModel::new(64);
        let vecs = model.embed(&[]).unwrap();
        assert!(vecs.is_empty());
    }

    #[test]
    fn trait_different_texts_produce_different_vectors() {
        let model = MockEmbeddingModel::new(32);
        let vecs = model.embed(&["hello", "world"]).unwrap();
        assert_ne!(vecs[0], vecs[1]);
    }

    #[test]
    fn trait_same_text_produces_same_vector() {
        let model = MockEmbeddingModel::new(32);
        let v1 = model.embed(&["hello"]).unwrap();
        let v2 = model.embed(&["hello"]).unwrap();
        assert_eq!(v1[0], v2[0]);
    }

    #[test]
    fn trait_is_send_sync() {
        // Verify the trait object can be shared across threads
        let model: Arc<dyn EmbeddingModel> = Arc::new(MockEmbeddingModel::new(64));
        let model_clone = Arc::clone(&model);
        let handle =
            std::thread::spawn(move || model_clone.embed(&["from another thread"]).unwrap());
        let result = handle.join().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), model.dimensions());
    }
}
