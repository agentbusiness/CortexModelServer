use candle_core::Tensor;
use candle_nn::{LayerNorm, Linear};

/// A loaded ModernBERT encoder: weights and the tables its attention uses.
pub(super) struct Encoder {
    /// Token embedding table, kept in the dtype it was stored in; rows are
    /// widened to `f32` after lookup, so the lookup itself is exact.
    pub tok_embeddings: Tensor,
    pub embedding_norm: LayerNorm,
    pub layers: Vec<EncoderLayer>,
    pub final_norm: LayerNorm,
    pub global_rope: Rope,
    pub local_rope: Rope,
    /// Farthest position a local-attention token attends to on either side.
    pub half_window: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
}

/// One pre-norm transformer layer with GeGLU feed-forward.
pub(super) struct EncoderLayer {
    /// Absent on the first layer, whose input is already normalized.
    pub attn_norm: Option<LayerNorm>,
    pub wqkv: Linear,
    pub attn_out: Linear,
    pub mlp_norm: LayerNorm,
    pub mlp_in: Linear,
    pub mlp_out: Linear,
    /// Sliding-window attention rather than attention over the whole input.
    pub local: bool,
}

/// Rotary position tables for one base frequency, one row per position.
pub(super) struct Rope {
    pub cos: Tensor,
    pub sin: Tensor,
}
