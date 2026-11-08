//! Prompt-injection detector on the semantic embedder.
//!
//! The semantic encoder's lower layers run unchanged; its top layer and final
//! norm are replaced by weights trained to recognise instructions aimed at an
//! AI model, and an attention-pooling head turns the token states into one
//! logit. The trained part ships inside the binary
//! (`resources/injection_guard_v2.safetensors`, 7.9 MB); how it was trained and
//! measured is recorded in `doc/operations/llm-guard.md`.
//!
//! Text is cut into short windows so an instruction hidden in a long passage
//! is not averaged away; a text scores as its most suspicious window.

mod from_files;
mod scan;
mod scan_passages;
mod score_windows;
mod text_windows;

use text_windows::text_windows;

use candle_core::{Device, Tensor};
use candle_nn::Linear;
use tokenizers::Tokenizer;

use super::encoder::Encoder;

/// Logit at which a window counts as an injection: on held-out ordinary text
/// it is exceeded by 0.3% of windows.
pub const INJECTION_GUARD_FLAG_LOGIT: f32 = 2.943;

/// The detector: the semantic encoder with the trained top layer, and its head.
pub struct InjectionGuard {
    encoder: Encoder,
    query: Tensor,
    hidden_projection: Linear,
    output_projection: Linear,
    tokenizer: Tokenizer,
    device: Device,
    pool: rayon::ThreadPool,
}

/// What a scan found in one text.
#[derive(Debug, Clone, PartialEq)]
pub struct InjectionVerdict {
    /// Whether the most suspicious window reached [`INJECTION_GUARD_FLAG_LOGIT`].
    pub flagged: bool,
    /// The detector's probability for the most suspicious window, 0.0 to 1.0.
    pub certainty: f32,
    /// The most suspicious window's text; empty when the text had none.
    pub passage: String,
}

impl std::fmt::Debug for InjectionGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InjectionGuard")
            .field(
                "execution_target",
                &if self.device.is_metal() { "gpu" } else { "cpu" },
            )
            .finish()
    }
}
