use embed_common::utils::error::Result;

use super::{InjectionGuard, InjectionVerdict};

impl InjectionGuard {
    /// Scans one text: its most suspicious window decides the verdict.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenizing or inference fails.
    pub fn scan(&self, text: &str) -> Result<InjectionVerdict> {
        self.scan_passages(&[text.to_string()])
    }
}
