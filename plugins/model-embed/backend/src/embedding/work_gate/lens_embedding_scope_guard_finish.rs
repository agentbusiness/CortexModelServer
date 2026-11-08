use super::*;

impl LensEmbeddingScopeGuard {
    /// Finishes this scope and returns whether it was the final active Lens
    /// embedding scope in the process.
    pub fn finish(mut self) -> bool {
        finish_lens_embedding_scope::finish_lens_embedding_scope(&mut self)
    }
}
