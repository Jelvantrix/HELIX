use helix_core::{Tensor, DType};
use crate::config::ModelConfig;
use crate::attention::linear;

/// GPT-2 MLP block: two linear layers with GELU in between.
pub struct MLP {
    pub c_fc_weight:   Tensor,   // [n_embd, n_inner]
    pub c_fc_bias:     Tensor,   // [n_inner]
    pub c_proj_weight: Tensor,   // [n_inner, n_embd]
    pub c_proj_bias:   Tensor,   // [n_embd]
}

impl MLP {
    pub fn new(cfg: &ModelConfig) -> Self {
        let n = cfg.n_embd;
        let inner = cfg.n_inner();
        Self {
            c_fc_weight:   Tensor::zeros(vec![n, inner], DType::F32),
            c_fc_bias:     Tensor::zeros(vec![inner], DType::F32),
            c_proj_weight: Tensor::zeros(vec![inner, n], DType::F32),
            c_proj_bias:   Tensor::zeros(vec![n], DType::F32),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = linear(x, &self.c_fc_weight, &self.c_fc_bias);
        let h = helix_core::ops::gelu_approx(&h);
        linear(&h, &self.c_proj_weight, &self.c_proj_bias)
    }
}
