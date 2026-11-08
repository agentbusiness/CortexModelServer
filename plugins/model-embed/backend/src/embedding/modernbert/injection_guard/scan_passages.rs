use embed_common::utils::error::Result;

use super::{INJECTION_GUARD_FLAG_LOGIT, InjectionGuard, InjectionVerdict, text_windows};

impl InjectionGuard {
    /// Scans several passages, each cut into its own windows so no window
    /// joins two passages: the most suspicious window of any decides.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenizing or inference fails.
    pub fn scan_passages(&self, passages: &[String]) -> Result<InjectionVerdict> {
        let windows: Vec<String> = passages
            .iter()
            .flat_map(|passage| text_windows(passage))
            .collect();
        let borrowed: Vec<&str> = windows.iter().map(String::as_str).collect();
        let logits = self.score_windows(&borrowed)?;
        let Some((index, &logit)) = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
        else {
            return Ok(InjectionVerdict {
                flagged: false,
                certainty: 0.0,
                passage: String::new(),
            });
        };
        Ok(InjectionVerdict {
            flagged: logit >= INJECTION_GUARD_FLAG_LOGIT,
            certainty: 1.0 / (1.0 + (-logit).exp()),
            passage: windows[index].clone(),
        })
    }
}
