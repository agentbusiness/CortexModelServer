use super::*;
use std::sync::atomic::Ordering;

/// Starts a process-wide Lens embedding scope and waits for already-running
/// non-Lens embedding operations to drain.
pub fn begin_lens_embedding_scope() -> LensEmbeddingScopeGuard {
    let gate = embedding_work_gate::embedding_work_gate();
    let mut state = gate.state.lock();
    state.lens_depth = state.lens_depth.saturating_add(1);
    gate.lens_pause_requested.store(true, Ordering::SeqCst);
    while state.active_non_lens_work > 0 {
        gate.state_changed.wait(&mut state);
    }
    drop(state);
    LensEmbeddingScopeGuard { gate, active: true }
}
