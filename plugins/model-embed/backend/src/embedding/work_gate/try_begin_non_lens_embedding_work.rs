use super::*;

/// Claims a process-wide non-Lens embedding operation without waiting, or
/// returns `None` when a Lens scope is active.
pub fn try_begin_non_lens_embedding_work() -> Option<NonLensEmbeddingWorkGuard> {
    let gate = embedding_work_gate::embedding_work_gate();
    let mut state = gate.state.lock();
    if state.lens_depth > 0 {
        return None;
    }
    state.active_non_lens_work = state.active_non_lens_work.saturating_add(1);
    drop(state);
    Some(NonLensEmbeddingWorkGuard { gate, active: true })
}
