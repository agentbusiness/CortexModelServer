use candle_core::{D, DType, Tensor};
use candle_nn::Module;

use super::encoder::{Encoder, EncoderLayer, Rope};

impl Encoder {
    /// Final token states, `(batch, len, hidden)`, for a right-padded batch.
    ///
    /// `ids` is `(batch, len)`; `key_mask` holds, row by row, 1.0 for a real
    /// token and 0.0 for padding. Padding is never attended to, and local
    /// layers see only tokens within the sliding window.
    pub(super) fn forward(
        &self,
        ids: &Tensor,
        key_mask: &[f32],
        batch: usize,
        len: usize,
    ) -> candle_core::Result<Tensor> {
        let device = ids.device();
        let padding: Vec<f32> = key_mask
            .iter()
            .map(|&real| if real > 0.0 { 0.0 } else { f32::MIN })
            .collect();
        let global_mask = Tensor::from_vec(padding, (batch, 1, 1, len), device)?;
        let mut window = Vec::with_capacity(len * len);
        for query in 0..len {
            for key in 0..len {
                window.push(if query.abs_diff(key) > self.half_window {
                    f32::NEG_INFINITY
                } else {
                    0.0
                });
            }
        }
        let local_mask =
            global_mask.broadcast_add(&Tensor::from_vec(window, (1, 1, len, len), device)?)?;

        let mut xs = self
            .tok_embeddings
            .index_select(&ids.flatten_all()?, 0)?
            .to_dtype(DType::F32)?
            .reshape((batch, len, self.hidden_size))?;
        xs = self.embedding_norm.forward(&xs)?;
        for layer in &self.layers {
            let (mask, rope) = if layer.local {
                (&local_mask, &self.local_rope)
            } else {
                (&global_mask, &self.global_rope)
            };
            xs = self.layer_forward(layer, &xs, mask, rope, batch, len)?;
        }
        self.final_norm.forward(&xs)
    }

    fn layer_forward(
        &self,
        layer: &EncoderLayer,
        xs: &Tensor,
        mask: &Tensor,
        rope: &Rope,
        batch: usize,
        len: usize,
    ) -> candle_core::Result<Tensor> {
        let normed = match &layer.attn_norm {
            Some(norm) => norm.forward(xs)?,
            None => xs.clone(),
        };
        let xs = (xs + self.attention(layer, &normed, mask, rope, batch, len)?)?;
        let projected = layer.mlp_in.forward(&layer.mlp_norm.forward(&xs)?)?;
        let halves = projected.chunk(2, D::Minus1)?;
        let gated = (halves[0].gelu_erf()? * &halves[1])?;
        xs + layer.mlp_out.forward(&gated)?
    }

    fn attention(
        &self,
        layer: &EncoderLayer,
        xs: &Tensor,
        mask: &Tensor,
        rope: &Rope,
        batch: usize,
        len: usize,
    ) -> candle_core::Result<Tensor> {
        let qkv = layer
            .wqkv
            .forward(xs)?
            .reshape((batch, len, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let rotate =
            |part: Tensor| candle_nn::rotary_emb::rope(&part.contiguous()?, &rope.cos, &rope.sin);
        let query = rotate(qkv.get(0)?)?;
        let key = rotate(qkv.get(1)?)?;
        let value = qkv.get(2)?.contiguous()?;
        let query = (query * (self.head_dim as f64).powf(-0.5))?;
        let scores = query
            .matmul(&key.transpose(D::Minus2, D::Minus1)?)?
            .broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended =
            weights
                .matmul(&value)?
                .transpose(1, 2)?
                .reshape((batch, len, self.hidden_size))?;
        layer.attn_out.forward(&attended)
    }
}
