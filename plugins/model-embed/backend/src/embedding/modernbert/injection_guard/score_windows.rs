use candle_core::{D, Tensor};
use candle_nn::Module;
use embed_common::utils::error::{Error, Result};
use tokenizers::Encoding;

use super::super::candle_error::candle_error;
use super::InjectionGuard;

/// Windows run together in one batch.
const BATCH_SIZE: usize = 32;

impl InjectionGuard {
    /// One injection logit per window, in input order.
    ///
    /// Windows are tokenized once, ordered by length and run in batches, so
    /// each batch pads only to its own longest window.
    pub(super) fn score_windows(&self, windows: &[&str]) -> Result<Vec<f32>> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        self.pool.install(|| {
            let encodings = self
                .tokenizer
                .encode_batch(windows.to_vec(), true)
                .map_err(|e| Error::Internal(format!("Tokenizer encode error: {e}")))?;
            let mut order: Vec<usize> = (0..encodings.len()).collect();
            order.sort_by_key(|&index| encodings[index].len());
            let mut logits = vec![0f32; windows.len()];
            for chunk in order.chunks(BATCH_SIZE) {
                let batch: Vec<&Encoding> = chunk.iter().map(|&index| &encodings[index]).collect();
                for (&index, logit) in chunk.iter().zip(self.run_batch(&batch)?) {
                    logits[index] = logit;
                }
            }
            Ok(logits)
        })
    }

    /// Runs the encoder on one right-padded batch, then the head: attention
    /// pooling with the learnt query next to the mean of the real tokens.
    fn run_batch(&self, encodings: &[&Encoding]) -> Result<Vec<f32>> {
        let batch = encodings.len();
        let len = encodings
            .iter()
            .map(|encoding| encoding.len())
            .max()
            .unwrap_or(0)
            .max(1);
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

        let failed = |error| candle_error("Injection guard inference failed", error);
        let run = || -> candle_core::Result<Vec<f32>> {
            let ids = Tensor::from_vec(ids, (batch, len), &self.device)?;
            let hidden = self.encoder.forward(&ids, &mask, batch, len)?;
            let weights = Tensor::from_vec(mask.clone(), (batch, len, 1), &self.device)?;
            let padding: Vec<f32> = mask
                .iter()
                .map(|&real| if real > 0.0 { 0.0 } else { -1e9 })
                .collect();
            let padding = Tensor::from_vec(padding, (batch, len), &self.device)?;
            let scores = hidden
                .broadcast_matmul(&self.query.unsqueeze(1)?)?
                .squeeze(2)?
                .broadcast_add(&padding)?;
            let attention = candle_nn::ops::softmax(&scores, D::Minus1)?.unsqueeze(2)?;
            let attended = hidden.broadcast_mul(&attention)?.sum(1)?;
            let mean = hidden
                .broadcast_mul(&weights)?
                .sum(1)?
                .broadcast_div(&weights.sum(1)?)?;
            let features = Tensor::cat(&[attended, mean], 1)?;
            self.output_projection
                .forward(&self.hidden_projection.forward(&features)?.gelu()?)?
                .squeeze(1)?
                .to_vec1::<f32>()
        };
        let logits = run().map_err(failed)?;
        if logits.len() != batch {
            return Err(Error::Internal(format!(
                "Injection guard returned {} scores for {batch} windows",
                logits.len()
            )));
        }
        Ok(logits)
    }
}
