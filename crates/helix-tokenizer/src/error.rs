use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("vocab file not found: {path}")]
    VocabNotFound { path: String },

    #[error("invalid vocab JSON: {0}")]
    InvalidVocab(String),

    #[error("invalid BPE merge rule: {0}")]
    InvalidMergeRule(String),

    #[error("token id {0} not in vocab")]
    UnknownTokenId(u32),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type TokenizerResult<T> = Result<T, TokenizerError>;
