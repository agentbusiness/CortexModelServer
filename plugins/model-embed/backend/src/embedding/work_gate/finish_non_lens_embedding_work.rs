use super::*;

pub(super) fn finish_non_lens_embedding_work(guard: &mut NonLensEmbeddingWorkGuard) {
    if !guard.active {
        return;
    }
    guard.active = false;
    let mut state = guard.gate.state.lock();
    debug_assert!(state.active_non_lens_work > 0);
    state.active_non_lens_work = state.active_non_lens_work.saturating_sub(1);
    if state.active_non_lens_work == 0 {
        guard.gate.state_changed.notify_all();
    }
}
