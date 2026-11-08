//! Local ONNX embedding model implementation backed by RTen.

use std::path::Path;
use std::sync::Arc;

use embed_common::embed_info;
use embed_common::utils::error::{Error, Result};
use rten::{Model, NodeId, RunOptions, ThreadPool, Value};
use rten_text::Tokenizer;

use super::EmbeddingModel;

const DEFAULT_MAX_INPUT_TOKENS: usize = 512;
const MAX_INPUT_TOKENS_ENV: &str = "CORTEX_EMBED_MAX_INPUT_TOKENS";

use super::update_embedding_progress as update_embedder_progress;

/// An embedding model backed by an ONNX graph and a Hugging Face tokenizer.
///
/// Load from local files using [`from_files()`](Self::from_files). The model
/// must accept `input_ids` and `attention_mask` as token tensors and produce a
/// pooled embedding output or token embeddings that can be mean-pooled.
pub struct OnnxEmbeddingModel {
    model: Model,
    tokenizer: Tokenizer,
    dimensions: usize,
    name: String,
    batch_size: usize,
    run_options: Option<RunOptions>,
    input_ids_node: NodeId,
    attention_mask_node: Option<NodeId>,
    token_type_ids_node: Option<NodeId>,
    output_node: NodeId,
}

// SAFETY: rten-text's Tokenizer does not advertise Send + Sync because its
// trait object fields do not carry those bounds. The knowledge app constructs the
// tokenizer once and then only uses it immutably for inference.
#[allow(unsafe_code)]
unsafe impl Send for OnnxEmbeddingModel {}
#[allow(unsafe_code)]
unsafe impl Sync for OnnxEmbeddingModel {}

impl std::fmt::Debug for OnnxEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEmbeddingModel")
            .field("name", &self.name)
            .field("dimensions", &self.dimensions)
            .field("batch_size", &self.batch_size)
            .finish()
    }
}

impl OnnxEmbeddingModel {
    fn max_input_tokens() -> usize {
        Self::max_input_tokens_from_env(std::env::var(MAX_INPUT_TOKENS_ENV).ok())
    }

    fn max_input_tokens_from_env(value: Option<String>) -> usize {
        value
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|limit| *limit > 0)
            .unwrap_or(DEFAULT_MAX_INPUT_TOKENS)
    }

    fn l2_normalize(vector: &mut [f32]) {
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for value in vector.iter_mut() {
                *value /= norm;
            }
        }
    }

    fn run_options(options: &super::EmbeddingOptions) -> Option<RunOptions> {
        let threads = options
            .intra_threads
            .max(options.inter_threads)
            .clamp(1, embed_common::cpu::max_cpu_threads());
        let pool = Arc::new(ThreadPool::with_num_threads(threads));
        Some(RunOptions::default().with_thread_pool(Some(pool)))
    }

    fn resolve_inputs(model: &Model) -> Result<(NodeId, Option<NodeId>, Option<NodeId>, NodeId)> {
        let input_ids_node = model
            .node_id("input_ids")
            .map_err(|e| Error::Internal(format!("Missing input_ids node: {e}")))?;
        let attention_mask_node = model.node_id("attention_mask").ok();
        let token_type_ids_node = model.node_id("token_type_ids").ok();
        let output_node = model
            .output_ids()
            .first()
            .copied()
            .ok_or_else(|| Error::Internal("Model has no outputs".to_string()))?;
        Ok((
            input_ids_node,
            attention_mask_node,
            token_type_ids_node,
            output_node,
        ))
    }

    fn load_backend(
        name: String,
        model_path: &Path,
        tokenizer_path: &Path,
        options: super::EmbeddingOptions,
    ) -> Result<Self> {
        update_embedder_progress(format!("loading ONNX graph from {}", model_path.display()));
        embed_info!("Loading RTen model from path={}", model_path.display());
        let model = Model::load_file(model_path)
            .map_err(|e| Error::Internal(format!("Failed to load ONNX model: {e}")))?;

        update_embedder_progress(format!(
            "loading tokenizer from {}",
            tokenizer_path.display()
        ));
        embed_info!("Loading tokenizer from path={}", tokenizer_path.display());
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::Internal(format!("Failed to load tokenizer: {e}")))?;

        let (input_ids_node, attention_mask_node, token_type_ids_node, output_node) =
            Self::resolve_inputs(&model)?;

        let mut instance = Self {
            model,
            tokenizer,
            dimensions: 0,
            name,
            batch_size: options.batch_size,
            run_options: Self::run_options(&options),
            input_ids_node,
            attention_mask_node,
            token_type_ids_node,
            output_node,
        };

        update_embedder_progress("resolving embedding dimensions");
        instance.dimensions = Self::probe_dimensions(&instance)?;
        update_embedder_progress(format!(
            "embedding model ready with {} dimensions",
            instance.dimensions
        ));

        Ok(instance)
    }

    /// Loads a pre-configured embedding model, downloading from HuggingFace Hub if needed.
    ///
    /// For preset models, the ONNX model and tokenizer are automatically downloaded
    /// on first use and cached locally. See [`EmbeddingModelConfig`](super::EmbeddingModelConfig)
    /// for available presets.
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to download, load, or initialize.
    #[cfg(feature = "embed")]
    pub fn from_config(config: super::EmbeddingModelConfig) -> Result<Self> {
        Self::from_config_with_options(config, super::EmbeddingOptions::default())
    }

    /// Loads a pre-configured embedding model with custom options.
    ///
    /// See [`EmbeddingOptions`](super::EmbeddingOptions) for batch size and thread configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the model or tokenizer fails to initialize.
    #[cfg(feature = "embed")]
    pub fn from_config_with_options(
        config: super::EmbeddingModelConfig,
        options: super::EmbeddingOptions,
    ) -> Result<Self> {
        update_embedder_progress("resolving embedding model files");
        let resolved = super::download::resolve(&config)?;
        update_embedder_progress(format!(
            "resolved model files: model={}, tokenizer={}",
            resolved.model_path.display(),
            resolved.tokenizer_path.display()
        ));

        Self::load_backend(
            resolved.name,
            &resolved.model_path,
            &resolved.tokenizer_path,
            options,
        )
    }

    /// Loads an embedding model from local ONNX model and tokenizer files.
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable model name (e.g., "my-model")
    /// * `model_path` - Path to the `.onnx` model file
    /// * `tokenizer_path` - Path to the `tokenizer.json` file
    ///
    /// # Errors
    ///
    /// Returns an error if the files cannot be loaded or the model is invalid.
    pub fn from_files(
        name: impl Into<String>,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::load_backend(
            name.into(),
            model_path.as_ref(),
            tokenizer_path.as_ref(),
            super::EmbeddingOptions::default(),
        )
    }

    /// Sets the maximum batch size for embedding generation.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    fn run_text_model(
        &self,
        batch: usize,
        seq_len: usize,
        ids: &[i32],
        mask: &[i32],
        token_type_ids: Option<&[i32]>,
    ) -> Result<Value> {
        let mut inputs = Vec::with_capacity(3);
        let input_ids = Value::from_shape([batch, seq_len], ids.to_vec())
            .map_err(|e| Error::Internal(format!("Failed to create input_ids tensor: {e}")))?;
        inputs.push((self.input_ids_node, input_ids.into()));

        if let Some(attention_mask_node) = self.attention_mask_node {
            let attention_mask =
                Value::from_shape([batch, seq_len], mask.to_vec()).map_err(|e| {
                    Error::Internal(format!("Failed to create attention_mask tensor: {e}"))
                })?;
            inputs.push((attention_mask_node, attention_mask.into()));
        }

        if let Some(token_type_ids_node) = self.token_type_ids_node {
            let token_type_ids = token_type_ids
                .map(<[i32]>::to_vec)
                .unwrap_or_else(|| vec![0; batch * seq_len]);
            let token_type_ids =
                Value::from_shape([batch, seq_len], token_type_ids).map_err(|e| {
                    Error::Internal(format!("Failed to create token_type_ids tensor: {e}"))
                })?;
            inputs.push((token_type_ids_node, token_type_ids.into()));
        }

        let [output] = self
            .model
            .run_n(inputs, [self.output_node], self.run_options.clone())
            .map_err(|e| Error::Internal(format!("Model inference failed: {e}")))?;
        Ok(output)
    }

    /// Probes model output dimensions with a dummy input.
    fn probe_dimensions(model: &Self) -> Result<usize> {
        let encoding = model
            .tokenizer
            .encode("hello", None)
            .map_err(|e| Error::Internal(format!("Tokenizer encode error: {e}")))?;

        let ids: Vec<i32> = encoding.token_ids().iter().map(|&id| id as i32).collect();
        let mask: Vec<i32> = vec![1; ids.len()];
        let token_type_ids: Vec<i32> = encoding.token_type_ids().map(|id| id as i32).collect();
        let len = ids.len();

        let output = model.run_text_model(1, len, &ids, &mask, Some(&token_type_ids))?;
        if let Ok((shape, _)) = output.clone().into_shape_vec::<f32, 3>() {
            return Ok(shape[2]);
        }
        if let Ok((shape, _)) = output.into_shape_vec::<f32, 2>() {
            return Ok(shape[1]);
        }
        Err(Error::Internal(
            "Model output has unsupported rank for embeddings".to_string(),
        ))
    }

    /// Embeds a single batch of texts.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(*text, None)
                    .map_err(|e| Error::Internal(format!("Tokenizer encode error: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;

        let token_limit = Self::max_input_tokens();
        let max_len = encodings
            .iter()
            .map(|encoding| encoding.token_ids().len().min(token_limit))
            .max()
            .unwrap_or(0);

        let batch = texts.len();
        let mut all_ids = vec![0i32; batch * max_len];
        let mut all_mask = vec![0i32; batch * max_len];
        let mut all_token_types = vec![0i32; batch * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            for (j, &id) in encoding.token_ids().iter().take(max_len).enumerate() {
                all_ids[i * max_len + j] = id as i32;
                all_mask[i * max_len + j] = 1;
            }
            for (j, token_type_id) in encoding.token_type_ids().take(max_len).enumerate() {
                all_token_types[i * max_len + j] = token_type_id as i32;
            }
        }

        let output =
            self.run_text_model(batch, max_len, &all_ids, &all_mask, Some(&all_token_types))?;

        if let Ok((shape, data)) = output.clone().into_shape_vec::<f32, 3>() {
            let seq_len = shape[1];
            let dims = shape[2];
            return Ok((0..batch)
                .map(|b| {
                    let token_base = b * seq_len * dims;
                    let mask_base = b * max_len;
                    let mut pooled = vec![0.0f32; dims];
                    let mut token_count = 0.0f32;

                    for token_idx in 0..seq_len {
                        if all_mask[mask_base + token_idx] == 0 {
                            continue;
                        }
                        let offset = token_base + token_idx * dims;
                        for dim_idx in 0..dims {
                            pooled[dim_idx] += data[offset + dim_idx];
                        }
                        token_count += 1.0;
                    }

                    if token_count > 0.0 {
                        for value in &mut pooled {
                            *value /= token_count;
                        }
                    }
                    Self::l2_normalize(&mut pooled);
                    pooled
                })
                .collect());
        }

        if let Ok((shape, data)) = output.into_shape_vec::<f32, 2>() {
            let dims = shape[1];
            return Ok((0..batch)
                .map(|b| {
                    let offset = b * dims;
                    let mut embedding = data[offset..offset + dims].to_vec();
                    Self::l2_normalize(&mut embedding);
                    embedding
                })
                .collect());
        }

        Err(Error::Internal(
            "Unexpected output shape from embedding model".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_input_tokens_defaults_to_minilm_position_limit() {
        assert_eq!(OnnxEmbeddingModel::max_input_tokens_from_env(None), 512);
    }
}

impl EmbeddingModel for OnnxEmbeddingModel {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            results.extend(self.embed_batch(chunk)?);
        }
        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn backend(&self) -> &'static str {
        "rten_onnx"
    }

    fn execution_target(&self) -> &'static str {
        "cpu"
    }
}
