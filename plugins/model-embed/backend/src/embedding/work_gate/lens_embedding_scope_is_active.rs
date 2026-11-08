use super::*;
use std::sync::atomic::Ordering;

/// Returns whether at least one process-wide Lens embedding scope is active.
pub fn lens_embedding_scope_is_active() -> bool {
    embedding_work_gate::embedding_work_gate()
        .lens_pause_requested
        .load(Ordering::Acquire)
}
