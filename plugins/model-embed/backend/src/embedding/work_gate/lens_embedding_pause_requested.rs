use super::*;

/// Returns the shared pause signal checked by interruptible non-Lens embedding
/// loops when a process-wide Lens scope starts.
pub fn lens_embedding_pause_requested() -> Arc<AtomicBool> {
    Arc::clone(&embedding_work_gate::embedding_work_gate().lens_pause_requested)
}
