use super::*;

impl Drop for NonLensEmbeddingWorkGuard {
    fn drop(&mut self) {
        finish_non_lens_embedding_work::finish_non_lens_embedding_work(self);
    }
}
