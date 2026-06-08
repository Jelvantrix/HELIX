use helix_core::{Tensor, DType, Shape, ops::matmul};
use crate::config::ModelConfig;

/// Multi-head self-attention (GPT-2 style).
/// Uses fused QKV projection.
pub struct MultiHeadAttention {
    /// Fused QKV projection: [n_embd, 3 * n_embd]
    pub c_attn_weight: Tensor,
    pub c_attn_bias:   Tensor,
    /// Output projection: [n_embd, n_embd]
    pub c_proj_weight: Tensor,
    pub c_proj_bias:   Tensor,
    pub n_head: usize,
    pub n_embd: usize,
    pub head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(cfg: &ModelConfig) -> Self {
        let n = cfg.n_embd;
        let h = cfg.n_head;
        Self {
            c_attn_weight: Tensor::zeros(vec![n, 3 * n], DType::F32),
            c_attn_bias:   Tensor::zeros(vec![3 * n], DType::F32),
            c_proj_weight: Tensor::zeros(vec![n, n], DType::F32),
            c_proj_bias:   Tensor::zeros(vec![n], DType::F32),
            n_head: h,
            n_embd: n,
            head_dim: n / h,
        }
    }

    /// Forward pass with KV-cache support.
    /// x: [batch, seq, n_embd]
    /// past_k, past_v: optional cached keys/values [batch, heads, past_seq, head_dim]
    /// Returns: (output: [batch, seq, n_embd], new_k: Tensor, new_v: Tensor)
    pub fn forward(
        &self,
        x: &Tensor,
        past_k: Option<&Tensor>,
        past_v: Option<&Tensor>,
    ) -> (Tensor, Tensor, Tensor) {
        let shape = x.shape().dims();
        let batch = shape[0];
        let seq   = shape[1];
        let n = self.n_embd;
        let h = self.n_head;
        let d = self.head_dim;

        // QKV projection: [batch, seq, 3*n]
        let qkv = linear(x, &self.c_attn_weight, &self.c_attn_bias);

        // Split QKV
        let (q, k, v) = split_qkv(&qkv, n);

        // Reshape to [batch, heads, seq, head_dim]
        let q = reshape_for_attn(&q, batch, seq, h, d);
        let k = reshape_for_attn(&k, batch, seq, h, d);
        let v = reshape_for_attn(&v, batch, seq, h, d);

        // Concatenate with KV cache if provided
        let (k_full, v_full) = match (past_k, past_v) {
            (Some(pk), Some(pv)) => (concat_seq(pk, &k), concat_seq(pv, &v)),
            _ => (k.clone(), v.clone()),
        };

        // Causal mask
        let total_seq = k_full.shape().dim(2);
        let mask = helix_core::ops::attention::causal_mask(total_seq);

        // Attention
        let attn_out = helix_core::ops::scaled_dot_product_attention(&q, &k_full, &v_full, Some(&mask));

        // Reshape back: [batch, seq, n_embd]
        let attn_out = reshape_from_attn(&attn_out, batch, seq, n);

        // Output projection
        let out = linear(&attn_out, &self.c_proj_weight, &self.c_proj_bias);

        (out, k_full, v_full)
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

pub fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Tensor {
    let result = matmul(x, weight).unwrap();
    add_bias(&result, bias)
}

fn add_bias(x: &Tensor, bias: &Tensor) -> Tensor {
    let shape  = x.shape().clone();
    let numel  = x.numel();
    let hidden = bias.numel();
    let x_data = x.as_slice::<f32>();
    let b_data = bias.as_slice::<f32>();
    let mut data = vec![0.0f32; numel];
    for i in 0..numel {
        data[i] = x_data[i] + b_data[i % hidden];
    }
    Tensor::from_vec_f32(data, shape.dims().to_vec()).unwrap()
}

fn split_qkv(qkv: &Tensor, n_embd: usize) -> (Tensor, Tensor, Tensor) {
    // qkv shape: [batch, seq, 3*n_embd]
    let shape = qkv.shape().dims();
    let batch = shape[0];
    let seq = shape[1];
    let data = qkv.as_slice::<f32>();

    let chunk = batch * seq * n_embd;
    let q_buf = helix_core::buffer::Buffer::zeros(chunk * 4);
    let k_buf = helix_core::buffer::Buffer::zeros(chunk * 4);
    let v_buf = helix_core::buffer::Buffer::zeros(chunk * 4);
    let q_out = unsafe { q_buf.as_mut_slice::<f32>(0, chunk) };
    let k_out = unsafe { k_buf.as_mut_slice::<f32>(0, chunk) };
    let v_out = unsafe { v_buf.as_mut_slice::<f32>(0, chunk) };

    for bs in 0..batch * seq {
        let src = &data[bs * 3 * n_embd..];
        q_out[bs * n_embd..(bs + 1) * n_embd].copy_from_slice(&src[..n_embd]);
        k_out[bs * n_embd..(bs + 1) * n_embd].copy_from_slice(&src[n_embd..2 * n_embd]);
        v_out[bs * n_embd..(bs + 1) * n_embd].copy_from_slice(&src[2 * n_embd..3 * n_embd]);
    }

    let s = Shape::new(&[batch, seq, n_embd]);
    let mk = |buf: helix_core::buffer::Buffer| {
        let shape_dims = s.dims().to_vec();
        let data = unsafe { buf.as_slice::<f32>(0, s.numel()) };
        let data_vec = data.to_vec();
        Tensor::from_vec_f32(data_vec, shape_dims).unwrap()
    };
    (mk(q_buf), mk(k_buf), mk(v_buf))
}

fn reshape_for_attn(t: &Tensor, batch: usize, seq: usize, heads: usize, head_dim: usize) -> Tensor {
    t.view(vec![batch, seq, heads, head_dim]).unwrap()
     .transpose().unwrap().contiguous()
    // After transpose: [batch, heads, seq, head_dim]
}

fn reshape_from_attn(t: &Tensor, batch: usize, seq: usize, n_embd: usize) -> Tensor {
    // [batch, heads, seq, head_dim] → [batch, seq, n_embd]
    let transposed = t.transpose().unwrap().contiguous();
    transposed.view(vec![batch, seq, n_embd]).unwrap()
}

fn concat_seq(past: &Tensor, new: &Tensor) -> Tensor {
    // Concatenate along seq dimension (dim 2): [b, h, past_s, d] + [b, h, new_s, d]
    let ps = past.shape().dims();
    let ns = new.shape().dims();
    let batch = ps[0]; let heads = ps[1];
    let past_seq = ps[2]; let new_seq = ns[2]; let d = ps[3];
    let total_seq = past_seq + new_seq;
    let numel = batch * heads * total_seq * d;
    let mut data = vec![0.0f32; numel];
    let pd = past.as_slice::<f32>();
    let nd = new.as_slice::<f32>();
    for b in 0..batch {
        for h in 0..heads {
            let out_bh = &mut data[(b * heads + h) * total_seq * d..];
            let past_bh = &pd[(b * heads + h) * past_seq * d..];
            let new_bh  = &nd[(b * heads + h) * new_seq * d..];
            out_bh[..past_seq * d].copy_from_slice(&past_bh[..past_seq * d]);
            out_bh[past_seq * d..(past_seq + new_seq) * d].copy_from_slice(&new_bh[..new_seq * d]);
        }
    }
    let shape = Shape::new(&[batch, heads, total_seq, d]);
    Tensor::from_vec_f32(data, shape.dims().to_vec()).unwrap()
}
