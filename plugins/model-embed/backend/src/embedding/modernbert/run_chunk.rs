use candle_core::Tensor;
use embed_common::utils::error::{Error, Result};
use tokenizers::Encoding;

use super::ModernBertEmbeddingModel;
use super::candle_error::candle_error;

impl ModernBertEmbeddingModel {
    /// Embeds one batch of tokenized texts: right-pads them to the longest,
    /// runs the encoder, mean-pools each text's real tokens and L2-normalizes.
    pub(super) fn run_chunk(&self, encodings: &[&Encoding]) -> Result<Vec<Vec<f32>>> {
        let batch = encodings.len();
        let len = encodings
            .iter()
            .map(|encoding| encoding.len())
            .max()
            .unwrap_or(0)
            .max(1);
        // Padding positions hold token 0; the mask keeps them out of both
        // attention and pooling, so the id they carry is never read.
        let mut ids = vec![0u32; batch * len];
        let mut mask = vec![0f32; batch * len];
        for (row, encoding) in encodings.iter().enumerate() {
            for (column, (&id, &real)) in encoding
                .get_ids()
                .iter()
                .zip(encoding.get_attention_mask())
                .enumerate()
            {
                ids[row * len + column] = id;
                mask[row * len + column] = if real > 0 { 1.0 } else { 0.0 };
            }
        }

        let failed = |error| candle_error("Embedding inference failed", error);
        let ids = Tensor::from_vec(ids, (batch, len), &self.device).map_err(failed)?;
        let hidden = self
            .encoder
            .forward(&ids, &mask, batch, len)
            .map_err(failed)?;
        let weights = Tensor::from_vec(mask, (batch, len, 1), &self.device).map_err(failed)?;
        let pooled = hidden
            .broadcast_mul(&weights)
            .and_then(|summed| summed.sum(1))
            .and_then(|summed| summed.broadcast_div(&weights.sum(1)?))
            .and_then(|pooled| pooled.to_vec2::<f32>())
            .map_err(failed)?;
        if pooled.len() != batch {
            return Err(Error::Internal(format!(
                "Embedding inference returned {} vectors for {batch} texts",
                pooled.len()
            )));
        }
        Ok(pooled
            .into_iter()
            .map(|mut vector| {
                l2_normalize(&mut vector);
                vector
            })
            .collect())
    }
}

fn l2_normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}
