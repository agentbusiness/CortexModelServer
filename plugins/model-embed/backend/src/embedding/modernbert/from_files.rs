use candle_core::Device;
use embed_common::utils::error::{Error, Result};
use embed_common::{embed_info, embed_warn};
use tokenizers::{Tokenizer, TruncationParams};

use super::super::semantic_model::SemanticModelFiles;
use super::super::{EmbeddingModel, EmbeddingOptions, update_embedding_progress};
use super::ModernBertEmbeddingModel;
use super::config::ModernBertConfig;
use super::load_encoder::load_encoder;
use super::select_device::select_device;

const DEFAULT_MAX_INPUT_TOKENS: usize = 512;
const MAX_INPUT_TOKENS_ENV: &str = "CORTEX_EMBED_MAX_INPUT_TOKENS";

impl ModernBertEmbeddingModel {
    /// Loads the encoder from its config, weights and tokenizer files.
    ///
    /// With [`EmbeddingOptions::prefer_gpu`] it runs on the GPU when this
    /// build can drive one; a GPU that cannot load or run the encoder is
    /// reported and the encoder is loaded on the CPU instead. CPU work runs
    /// on a pool of `max(intra_threads, inter_threads)` threads.
    ///
    /// # Errors
    ///
    /// Returns an error when the files cannot be read or the encoder cannot
    /// run on the CPU.
    pub fn from_files(
        name: impl Into<String>,
        files: &SemanticModelFiles,
        options: EmbeddingOptions,
    ) -> Result<Self> {
        let name = name.into();
        update_embedding_progress(format!(
            "loading encoder config from {}",
            files.config.display()
        ));
        let config = ModernBertConfig::read(&files.config)?;
        let device = select_device(options.prefer_gpu);
        if !device.is_cpu() {
            match Self::load_on(&name, files, &config, &options, device) {
                Ok(model) => return Ok(model),
                Err(error) => {
                    embed_warn!(
                        "Semantic encoder '{}' could not run on the GPU: {}; loading it on the CPU",
                        name,
                        error
                    );
                }
            }
        }
        Self::load_on(&name, files, &config, &options, Device::Cpu)
    }

    fn load_on(
        name: &str,
        files: &SemanticModelFiles,
        config: &ModernBertConfig,
        options: &EmbeddingOptions,
        device: Device,
    ) -> Result<Self> {
        let target = if device.is_cpu() { "cpu" } else { "gpu" };
        update_embedding_progress(format!(
            "loading tokenizer from {}",
            files.tokenizer.display()
        ));
        let mut tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| Error::Internal(format!("Failed to load tokenizer: {e}")))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_input_tokens().min(config.max_position_embeddings),
                ..Default::default()
            }))
            .map_err(|e| Error::Internal(format!("Failed to configure tokenizer: {e}")))?;
        tokenizer.with_padding(None);

        let threads = options
            .intra_threads
            .max(options.inter_threads)
            .clamp(1, embed_common::cpu::max_cpu_threads());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("cortex-embed-{index}"))
            .build()
            .map_err(|e| Error::Internal(format!("Failed to start embedding threads: {e}")))?;

        update_embedding_progress(format!(
            "loading encoder weights from {} on {target}",
            files.weights.display()
        ));
        let encoder = pool.install(|| load_encoder(&files.weights, config, &device))?;
        let model = Self {
            encoder,
            tokenizer,
            device,
            pool,
            name: name.to_string(),
            dimensions: config.hidden_size,
            batch_size: options.batch_size.max(1),
        };

        update_embedding_progress(format!("checking encoder output on {target}"));
        let probe = model.embed(&["hello"])?;
        let usable = probe.len() == 1
            && probe[0].len() == model.dimensions
            && probe[0].iter().all(|value| value.is_finite())
            && probe[0].iter().any(|value| *value != 0.0);
        if !usable {
            return Err(Error::Internal(format!(
                "Semantic encoder produced no usable vector on {target}"
            )));
        }
        embed_info!(
            "Semantic encoder '{}' ready on {} with {} dimensions",
            model.name,
            target,
            model.dimensions
        );
        update_embedding_progress(format!(
            "embedding model ready with {} dimensions",
            model.dimensions
        ));
        Ok(model)
    }
}

fn max_input_tokens() -> usize {
    std::env::var(MAX_INPUT_TOKENS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|limit| *limit > 0)
        .unwrap_or(DEFAULT_MAX_INPUT_TOKENS)
}
