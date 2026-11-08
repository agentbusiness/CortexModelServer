use embed_common::utils::error::Error;

/// Wraps a candle failure with what was being done when it happened.
pub(super) fn candle_error(context: &str, error: candle_core::Error) -> Error {
    Error::Internal(format!("{context}: {error}"))
}
