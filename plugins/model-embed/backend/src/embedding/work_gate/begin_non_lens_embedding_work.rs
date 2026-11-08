use super::*;

/// Waits until no Lens scope is active, then identifies one process-wide
/// non-Lens embedding operation until the returned guard drops.
pub fn begin_non_lens_embedding_work() -> NonLensEmbeddingWorkGuard {
    let gate = embedding_work_gate::embedding_work_gate();
    let mut state = gate.state.lock();
    while state.lens_depth > 0 {
        gate.state_changed.wait(&mut state);
    }
    state.active_non_lens_work = state.active_non_lens_work.saturating_add(1);
    drop(state);
    NonLensEmbeddingWorkGuard { gate, active: true }
}
