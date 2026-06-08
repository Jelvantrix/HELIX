//! # helix-core
//!
//! The tensor engine powering HELIX.
//! Provides: raw buffer management, dtype system, shape/stride algebra,
//! memory arena, and all primitive tensor operations (matmul, attention,
//! activation functions, normalization).
//!
//! All `unsafe` code is contained within this crate, documented, and tested.

pub mod buffer;
pub mod dtype;
pub mod shape;
pub mod tensor;
pub mod arena;
pub mod error;
pub mod ops;

// Convenient re-exports
pub use tensor::{Tensor, Device};
pub use dtype::{DType, Scalar};
pub use shape::{Shape, Strides, MAX_DIMS};
pub use arena::Arena;
pub use error::{CoreError, CoreResult};
pub use ops::{
    matmul, softmax, gelu, gelu_approx, layer_norm,
    scaled_dot_product_attention,
};
