use helix_core::Tensor;
use crate::{
    config::ModelConfig,
    layer_norm::LayerNorm,
    attention::MultiHeadAttention,
    mlp::MLP,
};

/// One GPT-2 transformer block: LN → Attention → residual → LN → MLP → residual
pub struct Block {
    pub ln_1:  LayerNorm,
    pub attn:  MultiHeadAttention,
    pub ln_2:  LayerNorm,
    pub mlp:   MLP,
}

impl Block {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            ln_1:  LayerNorm::new(cfg.n_embd, cfg.layer_norm_eps),
            attn:  MultiHeadAttention::new(cfg),
            ln_2:  LayerNorm::new(cfg.n_embd, cfg.layer_norm_eps),
            mlp:   MLP::new(cfg),
        }
    }

    /// Forward pass through one transformer block.
    /// x: [batch, seq, n_embd]
    /// Returns: (output, new_key_cache, new_value_cache)
    pub fn forward(
        &self,
        x: &Tensor,
        past_k: Option<&Tensor>,
        past_v: Option<&Tensor>,
    ) -> (Tensor, Tensor, Tensor) {
        // Pre-LN attention with residual
        let normed = self.ln_1.forward(x);
        let (attn_out, new_k, new_v) = self.attn.forward(&normed, past_k, past_v);
        let x = tensor_add(x, &attn_out);

        // Pre-LN MLP with residual
        let normed2 = self.ln_2.forward(&x);
        let mlp_out = self.mlp.forward(&normed2);
        let x = tensor_add(&x, &mlp_out);

        (x, new_k, new_v)
    }
}

fn tensor_add(a: &Tensor, b: &Tensor) -> Tensor {
    let numel = a.numel();
    let ad = a.as_slice::<f32>();
    let bd = b.as_slice::<f32>();
    let buf = helix_core::buffer::Buffer::zeros(numel * 4);
    let out = unsafe { buf.as_mut_slice::<f32>(0, numel) };
    for i in 0..numel { out[i] = ad[i] + bd[i]; }
    Tensor {
        data: buf, shape: a.shape().clone(), strides: a.strides().clone(),
        dtype: helix_core::DType::F32, offset: 0, device: helix_core::Device::Cpu,
    }
}
