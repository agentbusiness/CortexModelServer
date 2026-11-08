use super::*;
use std::sync::atomic::Ordering;

pub(super) fn finish_lens_embedding_scope(guard: &mut LensEmbeddingScopeGuard) -> bool {
    if !guard.active {
        return false;
    }
    guard.active = false;
    let mut state = guard.gate.state.lock();
    debug_assert!(state.lens_depth > 0);
    state.lens_depth = state.lens_depth.saturating_sub(1);
    let became_inactive = state.lens_depth == 0;
    if became_inactive {
        guard
            .gate
            .lens_pause_requested
            .store(false, Ordering::SeqCst);
        guard.gate.state_changed.notify_all();
    }
    became_inactive
}
