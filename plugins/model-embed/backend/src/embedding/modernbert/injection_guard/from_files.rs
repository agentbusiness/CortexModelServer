use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear};
use embed_common::embed_warn;
use embed_common::utils::error::{Error, Result};
use tokenizers::{Tokenizer, TruncationParams};

use super::super::super::EmbeddingOptions;
use super::super::super::semantic_model::SemanticModelFiles;
use super::super::candle_error::candle_error;
use super::super::config::ModernBertConfig;
use super::super::load_encoder::load_encoder;
use super::super::select_device::select_device;
use super::InjectionGuard;

/// The trained top layer, final norm and head.
const TRAINED_WEIGHTS: &[u8] =
    include_bytes!("../../../../resources/injection_guard_v2.safetensors");

/// Tokens read per window, as in training.
const MAX_INPUT_TOKENS: usize = 512;

impl InjectionGuard {
    /// Loads the semantic encoder's files with the trained part on top.
    ///
    /// Runs on the GPU when [`EmbeddingOptions::prefer_gpu`] asks for it and
    /// one can load it, otherwise on the CPU.
    ///
    /// # Errors
    ///
    /// Returns an error when the files cannot be read or the detector cannot
    /// run on the CPU.
    pub fn from_files(files: &SemanticModelFiles, options: EmbeddingOptions) -> Result<Self> {
        let config = ModernBertConfig::read(&files.config)?;
        let device = select_device(options.prefer_gpu);
        if !device.is_cpu() {
            match load_on(files, &config, &options, device) {
                Ok(guard) => return Ok(guard),
                Err(error) => {
                    embed_warn!(
                        "Injection guard could not run on the GPU: {}; loading it on the CPU",
                        error
                    );
                }
            }
        }
        load_on(files, &config, &options, Device::Cpu)
    }
}

fn load_on(
    files: &SemanticModelFiles,
    config: &ModernBertConfig,
    options: &EmbeddingOptions,
    device: Device,
) -> Result<InjectionGuard> {
    let mut tokenizer = Tokenizer::from_file(&files.tokenizer)
        .map_err(|e| Error::Internal(format!("Failed to load tokenizer: {e}")))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: MAX_INPUT_TOKENS.min(config.max_position_embeddings),
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
        .thread_name(|index| format!("cortex-guard-{index}"))
        .build()
        .map_err(|e| Error::Internal(format!("Failed to start guard threads: {e}")))?;

    let mut encoder = pool.install(|| load_encoder(&files.weights, config, &device))?;
    let trained = candle_core::safetensors::load_buffer(TRAINED_WEIGHTS, &device)
        .map_err(|e| candle_error("Failed to read the injection guard weights", e))?;
    let take = |name: &str| trained_tensor(&trained, name);
    let norm = |name: &str| -> Result<LayerNorm> {
        let bias = Tensor::zeros(config.hidden_size, DType::F32, &device)
            .map_err(|e| candle_error("Failed to allocate layer norm bias", e))?;
        Ok(LayerNorm::new(take(name)?, bias, config.norm_eps))
    };

    let top = encoder
        .layers
        .last_mut()
        .ok_or_else(|| Error::Internal("Encoder has no layers".to_string()))?;
    if top.local {
        return Err(Error::Internal(
            "The injection guard expects the encoder's top layer to use global attention"
                .to_string(),
        ));
    }
    top.attn_norm = Some(norm("top.attn_norm")?);
    top.wqkv = Linear::new(take("top.wqkv")?, None);
    top.attn_out = Linear::new(take("top.attn_out")?, None);
    top.mlp_norm = norm("top.mlp_norm")?;
    top.mlp_in = Linear::new(take("top.mlp_in")?, None);
    top.mlp_out = Linear::new(take("top.mlp_out")?, None);
    encoder.final_norm = norm("top.final_norm")?;

    Ok(InjectionGuard {
        encoder,
        query: take("head.query")?,
        hidden_projection: Linear::new(take("head.l1.weight")?, Some(take("head.l1.bias")?)),
        output_projection: Linear::new(take("head.l2.weight")?, Some(take("head.l2.bias")?)),
        tokenizer,
        device,
        pool,
    })
}

fn trained_tensor(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .get(name)
        .cloned()
        .ok_or_else(|| Error::Internal(format!("Injection guard weights are missing '{name}'")))
}
