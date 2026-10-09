//! Errors produced by pipeline stages.

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

/// Anything a stage can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Resource keys travel inside a committed index, so in the
    /// clone-someone's-repository case they are attacker-controlled. A key
    /// containing `..` would let a consumer resolving a span read outside the
    /// repository and feed the result into model context.
    #[error("resource key {0:?} is not root-relative")]
    UnsafeResourceKey(String),

    #[error("no {stage} implementation named {implementation:?}")]
    UnknownImplementation {
        stage: &'static str,
        implementation: String,
    },

    /// A stored payload did not decode. Reachable from committed bytes in a
    /// cloned repository, so it must be an error rather than an assumption.
    #[error("malformed stored payload: {0}")]
    MalformedPayload(&'static str),

    #[error("could not decode stored payload: {0}")]
    Decode(#[from] serde_json::Error),

    #[error(transparent)]
    Format(#[from] map_format::Error),
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}
