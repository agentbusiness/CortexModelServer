use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear};
use embed_common::utils::error::{Error, Result};

use super::candle_error::candle_error;
use super::config::ModernBertConfig;
use super::encoder::{Encoder, EncoderLayer, Rope};

/// Builds the encoder from its `safetensors` weights on `device`.
///
/// Weights are read on the CPU, widened to `f32` and then moved, except the
/// token embedding table, which keeps its stored dtype.
pub(super) fn load_encoder(
    weights: &Path,
    config: &ModernBertConfig,
    device: &Device,
) -> Result<Encoder> {
    let tensors = candle_core::safetensors::load(weights, &Device::Cpu)
        .map_err(|e| candle_error(&format!("Failed to read {}", weights.display()), e))?;
    let hidden = config.hidden_size;
    let take = |name: &str, shape: &[usize], widen: bool| -> Result<Tensor> {
        weight(&tensors, name, shape, widen, device)
    };
    let norm = |name: &str| -> Result<LayerNorm> {
        let bias = Tensor::zeros(hidden, DType::F32, device)
            .map_err(|e| candle_error("Failed to allocate layer norm bias", e))?;
        Ok(LayerNorm::new(
            take(name, &[hidden], true)?,
            bias,
            config.norm_eps,
        ))
    };
    let linear = |name: &str, out_features: usize, in_features: usize| -> Result<Linear> {
        Ok(Linear::new(
            take(name, &[out_features, in_features], true)?,
            None,
        ))
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for index in 0..config.num_hidden_layers {
        let prefix = format!("layers.{index}");
        let attn_norm_name = format!("{prefix}.attn_norm.weight");
        layers.push(EncoderLayer {
            attn_norm: if tensors.contains_key(&attn_norm_name) {
                Some(norm(&attn_norm_name)?)
            } else {
                None
            },
            wqkv: linear(&format!("{prefix}.attn.Wqkv.weight"), hidden * 3, hidden)?,
            attn_out: linear(&format!("{prefix}.attn.Wo.weight"), hidden, hidden)?,
            mlp_norm: norm(&format!("{prefix}.mlp_norm.weight"))?,
            mlp_in: linear(
                &format!("{prefix}.mlp.Wi.weight"),
                config.intermediate_size * 2,
                hidden,
            )?,
            mlp_out: linear(
                &format!("{prefix}.mlp.Wo.weight"),
                hidden,
                config.intermediate_size,
            )?,
            local: index % config.global_attn_every_n_layers != 0,
        });
    }

    let head_dim = hidden / config.num_attention_heads;
    Ok(Encoder {
        tok_embeddings: take(
            "embeddings.tok_embeddings.weight",
            &[config.vocab_size, hidden],
            false,
        )?,
        embedding_norm: norm("embeddings.norm.weight")?,
        layers,
        final_norm: norm("final_norm.weight")?,
        global_rope: rope(
            config.global_rope_theta,
            head_dim,
            config.max_position_embeddings,
            device,
        )?,
        local_rope: rope(
            config.local_rope_theta,
            head_dim,
            config.max_position_embeddings,
            device,
        )?,
        half_window: config.local_attention / 2,
        num_heads: config.num_attention_heads,
        head_dim,
        hidden_size: hidden,
    })
}

fn weight(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    shape: &[usize],
    widen: bool,
    device: &Device,
) -> Result<Tensor> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| Error::Internal(format!("Encoder weights are missing '{name}'")))?;
    if tensor.dims() != shape {
        return Err(Error::Internal(format!(
            "Encoder weight '{name}' has shape {:?}, expected {shape:?}",
            tensor.dims()
        )));
    }
    let tensor = if widen {
        tensor
            .to_dtype(DType::F32)
            .map_err(|e| candle_error(&format!("Failed to widen '{name}'"), e))?
    } else {
        tensor.clone()
    };
    tensor
        .to_device(device)
        .map_err(|e| candle_error(&format!("Failed to place '{name}' on the device"), e))
}

/// Rotary tables as the reference model computes them: `f32` inverse
/// frequencies over even dimensions, one row per position.
fn rope(theta: f64, head_dim: usize, positions: usize, device: &Device) -> Result<Rope> {
    let inverse: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|dim| 1.0 / (theta as f32).powf(dim as f32 / head_dim as f32))
        .collect();
    let mut cos = Vec::with_capacity(positions * inverse.len());
    let mut sin = Vec::with_capacity(positions * inverse.len());
    for position in 0..positions {
        for frequency in &inverse {
            let angle = position as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let table = |values: Vec<f32>| {
        Tensor::from_vec(values, (positions, head_dim / 2), device)
            .map_err(|e| candle_error("Failed to build rotary table", e))
    };
    Ok(Rope {
        cos: table(cos)?,
        sin: table(sin)?,
    })
}
