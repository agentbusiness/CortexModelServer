//! The items of the knowledge app's common crate the copied embedders use (see Cargo.toml).

#[doc(hidden)]
pub use tracing as __tracing;

pub mod utils {
    pub mod error {
        use std::fmt;

        /// The knowledge app's error, cut down to the variant the embedders build.
        #[derive(Debug, Clone, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum Error {
            /// An internal failure, with what was being done.
            Internal(String),
        }

        impl fmt::Display for Error {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Error::Internal(message) => write!(f, "internal error: {message}"),
                }
            }
        }

        impl std::error::Error for Error {}

        pub type Result<T> = std::result::Result<T, Error>;
    }
}

pub mod cpu {
    // Copied from the knowledge app's common crate, src/cpu/max_cpu_threads.rs.
    /// The most threads any CPU-bound pool or work split may use: half of the
    /// machine's available cores, never fewer than one.
    ///
    /// A machine whose core count cannot be read is treated as having one core,
    /// so the answer is then one thread.
    #[must_use]
    pub fn max_cpu_threads() -> usize {
        (std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get) / 2).max(1)
    }
}

/// Emits a warning-level event (the knowledge app's macro, on `tracing`).
#[macro_export]
macro_rules! embed_warn {
    ($($arg:tt)*) => {{
        $crate::__tracing::warn!($($arg)*);
    }};
}

/// Emits an info-level event (the knowledge app's macro, on `tracing`).
#[macro_export]
macro_rules! embed_info {
    ($($arg:tt)*) => {{
        $crate::__tracing::info!($($arg)*);
    }};
}
