use helix_core::Tensor;

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias:   Tensor,
    pub eps:    f32,
}

impl LayerNorm {
    pub fn new(n_embd: usize, eps: f32) -> Self {
        // Initialize weight to ones, bias to zeros
        let w_data = vec![1.0f32; n_embd];
        let b_data = vec![0.0f32; n_embd];
        Self {
            weight: Tensor::from_vec_f32(w_data, vec![n_embd]).unwrap(),
            bias:   Tensor::from_vec_f32(b_data, vec![n_embd]).unwrap(),
            eps,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        helix_core::ops::layer_norm(x, &self.weight, &self.bias, self.eps)
    }
}
