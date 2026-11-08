mod begin_lens_embedding_scope;
mod begin_non_lens_embedding_work;
mod drop_lens_embedding_scope_guard;
mod drop_non_lens_embedding_work_guard;
mod embedding_work_gate;
mod finish_lens_embedding_scope;
mod finish_non_lens_embedding_work;
mod lens_embedding_pause_requested;
mod lens_embedding_scope_guard_finish;
mod lens_embedding_scope_is_active;
mod try_begin_non_lens_embedding_work;

pub use begin_lens_embedding_scope::begin_lens_embedding_scope;
pub use begin_non_lens_embedding_work::begin_non_lens_embedding_work;
pub use lens_embedding_pause_requested::lens_embedding_pause_requested;
pub use lens_embedding_scope_is_active::lens_embedding_scope_is_active;
pub use try_begin_non_lens_embedding_work::try_begin_non_lens_embedding_work;

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;

use parking_lot::{Condvar, Mutex};

#[derive(Default)]
struct EmbeddingWorkGateState {
    lens_depth: usize,
    active_non_lens_work: usize,
}

struct EmbeddingWorkGate {
    state: Mutex<EmbeddingWorkGateState>,
    state_changed: Condvar,
    lens_pause_requested: Arc<AtomicBool>,
}

impl Default for EmbeddingWorkGate {
    fn default() -> Self {
        Self {
            state: Mutex::new(EmbeddingWorkGateState::default()),
            state_changed: Condvar::new(),
            lens_pause_requested: Arc::new(AtomicBool::new(false)),
        }
    }
}

static EMBEDDING_WORK_GATE: OnceLock<EmbeddingWorkGate> = OnceLock::new();

/// RAII guard that keeps process-wide non-Lens embedding work paused.
pub struct LensEmbeddingScopeGuard {
    gate: &'static EmbeddingWorkGate,
    active: bool,
}

/// RAII guard that identifies one active non-Lens embedding operation.
pub struct NonLensEmbeddingWorkGuard {
    gate: &'static EmbeddingWorkGate,
    active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn lens_scope_is_nested_and_waits_for_non_lens_work_process_wide() {
        let non_lens = begin_non_lens_embedding_work();
        let (lens_attempt_tx, lens_attempt_rx) = mpsc::channel();
        let (lens_started_tx, lens_started_rx) = mpsc::channel();
        let (release_lens_tx, release_lens_rx) = mpsc::channel();
        let lens_thread = thread::spawn(move || {
            lens_attempt_tx.send(()).expect("lens attempt signal");
            let lens = begin_lens_embedding_scope();
            lens_started_tx.send(()).expect("lens start signal");
            release_lens_rx.recv().expect("lens release signal");
            drop(lens);
        });

        lens_attempt_rx.recv().expect("lens attempt");
        let pause_deadline = Instant::now() + Duration::from_secs(1);
        while !lens_embedding_pause_requested().load(std::sync::atomic::Ordering::Acquire)
            && Instant::now() < pause_deadline
        {
            thread::yield_now();
        }
        assert!(
            lens_started_rx
                .recv_timeout(Duration::from_millis(50))
                .is_err()
        );
        assert!(lens_embedding_pause_requested().load(std::sync::atomic::Ordering::Acquire));
        drop(non_lens);
        lens_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("Lens scope should start after prior non-Lens work drains");
        assert!(try_begin_non_lens_embedding_work().is_none());

        let nested = begin_lens_embedding_scope();
        release_lens_tx.send(()).expect("release outer Lens scope");
        lens_thread.join().expect("Lens thread");
        assert!(lens_embedding_scope_is_active());
        assert!(try_begin_non_lens_embedding_work().is_none());
        drop(nested);

        assert!(!lens_embedding_scope_is_active());
        assert!(try_begin_non_lens_embedding_work().is_some());
    }
}
