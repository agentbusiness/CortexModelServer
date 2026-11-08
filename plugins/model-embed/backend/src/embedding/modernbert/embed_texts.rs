use embed_common::utils::error::{Error, Result};

use super::ModernBertEmbeddingModel;

impl ModernBertEmbeddingModel {
    /// One normalized vector per text, in input order.
    ///
    /// Texts are tokenized once, ordered by length and run in batches of
    /// `batch_size`, so each batch pads only to its own longest text.
    pub(super) fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.pool.install(|| {
            let encodings = self
                .tokenizer
                .encode_batch(texts.to_vec(), true)
                .map_err(|e| Error::Internal(format!("Tokenizer encode error: {e}")))?;
            let mut order: Vec<usize> = (0..encodings.len()).collect();
            order.sort_by_key(|&index| encodings[index].len());
            let mut vectors = vec![Vec::new(); texts.len()];
            for chunk in order.chunks(self.batch_size) {
                let batch = chunk
                    .iter()
                    .map(|&index| &encodings[index])
                    .collect::<Vec<_>>();
                for (&index, vector) in chunk.iter().zip(self.run_chunk(&batch)?) {
                    vectors[index] = vector;
                }
            }
            Ok(vectors)
        })
    }
}
