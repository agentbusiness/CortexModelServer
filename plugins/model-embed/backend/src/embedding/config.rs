//! Embedding options shared by every embedding backend.

/// Options for embedding model loading and inference.
///
/// Use the builder methods to customize batch size and thread configuration.
/// Defaults are tuned for single-user workloads on CPU.
///
/// # Examples
///
/// ```no_run
/// use crate::embedding::EmbeddingOptions;
///
/// let options = EmbeddingOptions::new()
///     .with_batch_size(64)
///     .with_intra_threads(4);
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddingOptions {
    /// Maximum batch size for embedding generation.
    pub batch_size: usize,
    /// Number of ONNX intra-op threads (parallelism within a single operation).
    pub intra_threads: usize,
    /// Number of ONNX inter-op threads (parallelism across operations).
    pub inter_threads: usize,
    /// Whether a model that can run on a GPU should try it before the CPU.
    pub prefer_gpu: bool,
}

impl Default for EmbeddingOptions {
    fn default() -> Self {
        Self {
            batch_size: 32,
            intra_threads: 1,
            inter_threads: 1,
            prefer_gpu: true,
        }
    }
}

impl EmbeddingOptions {
    /// Creates default embedding options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum batch size for embedding generation.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the number of ONNX intra-op threads.
    ///
    /// Controls parallelism within individual ONNX operations (e.g., matrix
    /// multiplication). Higher values help with larger models or batch sizes.
    #[must_use]
    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = threads;
        self
    }

    /// Sets the number of ONNX inter-op threads.
    ///
    /// Controls parallelism across independent ONNX operations. Usually
    /// less impactful than intra-op threads for embedding models.
    #[must_use]
    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = threads;
        self
    }

    /// Sets whether a GPU-capable model tries the GPU before the CPU.
    #[must_use]
    pub fn with_prefer_gpu(mut self, prefer_gpu: bool) -> Self {
        self.prefer_gpu = prefer_gpu;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_default_values() {
        let opts = EmbeddingOptions::default();
        assert_eq!(opts.batch_size, 32);
        assert_eq!(opts.intra_threads, 1);
        assert_eq!(opts.inter_threads, 1);
        assert!(opts.prefer_gpu);
    }

    #[test]
    fn options_builder_chaining() {
        let opts = EmbeddingOptions::new()
            .with_batch_size(64)
            .with_intra_threads(4)
            .with_inter_threads(2);
        assert_eq!(opts.batch_size, 64);
        assert_eq!(opts.intra_threads, 4);
        assert_eq!(opts.inter_threads, 2);
    }
}
