use crate::{
    ops::{matmul::matmul, activation::softmax},
    tensor::Tensor,
    dtype::DType,
    buffer::Buffer,
    shape::Shape,
    tensor::Device,
};

/// Scaled dot-product attention.
/// Q: [batch, heads, seq_q, head_dim]
/// K: [batch, heads, seq_k, head_dim]
/// V: [batch, heads, seq_k, head_dim]
/// mask: optional [1, 1, seq_q, seq_k] additive mask (use -inf for masked positions)
/// Returns: [batch, heads, seq_q, head_dim]
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Tensor {
    assert_eq!(q.dtype(), DType::F32);

    let shape = q.shape().dims();
    let _batch = shape[0];
    let _heads = shape[1];
    let _seq_q = shape[2];
    let head_dim = shape[3];

    let scale = 1.0 / (head_dim as f32).sqrt();

    // scores = Q @ K^T / sqrt(head_dim)  shape: [batch, heads, seq_q, seq_k]
    let k_t = k.transpose().unwrap().contiguous();
    let mut scores = matmul(q, &k_t).unwrap();

    // Scale
    scale_inplace(&mut scores, scale);

    // Add causal mask
    if let Some(m) = mask {
        add_inplace(&mut scores, m);
    }

    // Softmax over last dim (seq_k)
    let weights = softmax(&scores);

    // out = weights @ V  shape: [batch, heads, seq_q, head_dim]
    matmul(&weights, v).unwrap()
}

/// Build a causal mask for a given sequence length.
/// Positions where j > i get -1e9 (effectively -inf after softmax).
pub fn causal_mask(seq_len: usize) -> Tensor {
    let numel = seq_len * seq_len;
    let buf = Buffer::zeros(numel * 4);
    let data = unsafe { buf.as_mut_slice::<f32>(0, numel) };
    for i in 0..seq_len {
        for j in 0..seq_len {
            data[i * seq_len + j] = if j > i { -1e9 } else { 0.0 };
        }
    }
    let shape = Shape::new(&[1, 1, seq_len, seq_len]);
    Tensor { data: buf, shape, strides: shape.strides(), dtype: DType::F32, offset: 0, device: Device::Cpu }
}

fn scale_inplace(t: &mut Tensor, scale: f32) {
    let data = t.as_mut_slice::<f32>();
    for v in data.iter_mut() { *v *= scale; }
}

fn add_inplace(t: &mut Tensor, other: &Tensor) {
    let a = t.as_mut_slice::<f32>();
    let b = other.as_slice::<f32>();
    // Assumes same shape or broadcastable last dims
    let len = a.len().min(b.len());
    for i in 0..len { a[i] += b[i % b.len()]; }
}
