use super::*;

impl Drop for LensEmbeddingScopeGuard {
    fn drop(&mut self) {
        finish_lens_embedding_scope::finish_lens_embedding_scope(self);
    }
}
