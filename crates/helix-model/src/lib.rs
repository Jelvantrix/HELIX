//! # helix-model
//!
//! GPT-2 transformer architecture.
//! All weight tensors are plain Tensor instances — loading is handled by helix-loader.
//! The model is purely functional: forward() takes input, returns output.

pub mod config;
pub mod embedding;
pub mod layer_norm;
pub mod attention;
pub mod mlp;
pub mod block;
pub mod gpt2;

pub use config::ModelConfig;
pub use gpt2::GPT2;
