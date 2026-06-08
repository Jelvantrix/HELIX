pub mod matmul;
pub mod activation;
pub mod norm;
pub mod attention;

// Re-export the top-level functions for ergonomic imports
pub use matmul::matmul;
pub use activation::{gelu, gelu_approx, softmax, sigmoid};
pub use norm::layer_norm;
pub use attention::scaled_dot_product_attention;
