//! The CLI's expected-failure type: printed without a backtrace, exits with `code`.

/// An expected failure with a user-facing message and a process exit code.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct OflError {
    /// The message printed after `ofl: `.
    pub message: String,
    /// The process exit code (1 for failures, 2 for misuse).
    pub code: u8,
}

impl OflError {
    /// A failure exiting with code 1.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), code: 1 }
    }

    /// A misuse failure exiting with code 2.
    #[must_use]
    pub fn misuse(message: impl Into<String>) -> Self {
        Self { message: message.into(), code: 2 }
    }
}

/// Result alias for command code; the error type stays overridable.
pub type Result<T, E = OflError> = std::result::Result<T, E>;
