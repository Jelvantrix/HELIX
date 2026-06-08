use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("dimension out of bounds: axis {axis} on tensor with {ndim} dims")]
    DimOutOfBounds { axis: usize, ndim: usize },

    #[error("dtype mismatch: cannot {op} {lhs} and {rhs}")]
    DtypeMismatch { op: &'static str, lhs: &'static str, rhs: &'static str },

    #[error("non-contiguous tensor: operation requires contiguous memory layout")]
    NonContiguous,

    #[error("arena exhausted: requested {requested} bytes, only {available} remain")]
    ArenaExhausted { requested: usize, available: usize },

    #[error("invalid index: index {index} out of range for dim of size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    #[error("broadcast error: shapes {a:?} and {b:?} are not broadcastable")]
    BroadcastError { a: Vec<usize>, b: Vec<usize> },
}

pub type CoreResult<T> = Result<T, CoreError>;
