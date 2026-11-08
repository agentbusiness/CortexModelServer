//! Small helpers.

use std::path::PathBuf;

/// The user's home folder.
pub fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

/// The shared models folder every Cortex app uses (`docs/SHARED-MODELS.md`).
pub fn models_root() -> PathBuf {
    home().join(".cortex-models")
}

pub fn now_ms() -> u64 {
    cc_protocol::contracts::now_ms()
}

/// `macos` | `linux`.
pub fn platform() -> &'static str {
    if cfg!(target_os = "macos") { "macos" } else { "linux" }
}

/// Bytes as people read them: `27.1 GB`.
pub fn gb(bytes: u64) -> String {
    let value = bytes as f64 / 1e9;
    if value >= 10.0 { format!("{value:.0} GB") } else { format!("{value:.1} GB") }
}

/// Context tokens as people read them: `32k`.
pub fn ctx_label(tokens: u32) -> String {
    if tokens >= 1024 && tokens % 1024 == 0 { format!("{}k", tokens / 1024) } else { tokens.to_string() }
}
