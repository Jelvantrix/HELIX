use helix_core::{Tensor, DType, Shape};

/// Embedding lookup table: [vocab_size, n_embd]
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(vocab_size: usize, n_embd: usize) -> Self {
        Self { weight: Tensor::zeros(vec![vocab_size, n_embd], DType::F32) }
    }

    /// Look up embeddings for a batch of token IDs.
    /// ids: [seq_len] → output: [seq_len, n_embd]
    pub fn forward(&self, ids: &[u32]) -> Tensor {
        let seq_len = ids.len();
        let n_embd  = self.weight.shape().dim(1);
        let w = self.weight.as_slice::<f32>();
        let mut data = vec![0.0f32; seq_len * n_embd];
        for (i, &id) in ids.iter().enumerate() {
            let src = &w[(id as usize) * n_embd..(id as usize + 1) * n_embd];
            let dst = &mut data[i * n_embd..(i + 1) * n_embd];
            dst.copy_from_slice(src);
        }
        Tensor::from_vec_f32(data, vec![seq_len, n_embd]).unwrap()
    }
}
