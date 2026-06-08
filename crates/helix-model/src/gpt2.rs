use helix_core::{Tensor, DType};
use crate::{
    config::ModelConfig,
    embedding::Embedding,
    layer_norm::LayerNorm,
    block::Block,
};

/// Complete GPT-2 model.
pub struct GPT2 {
    pub config: ModelConfig,
    pub wte:    Embedding,    // token embedding  [vocab_size, n_embd]
    pub wpe:    Embedding,    // position embedding [n_positions, n_embd]
    pub blocks: Vec<Block>,
    pub ln_f:   LayerNorm,
    // lm_head shares weights with wte (weight tying)
}

impl GPT2 {
    pub fn new(config: ModelConfig) -> Self {
        let blocks = (0..config.n_layer).map(|_| Block::new(&config)).collect();
        let ln_f = LayerNorm::new(config.n_embd, config.layer_norm_eps);
        let wte = Embedding::new(config.vocab_size, config.n_embd);
        let wpe = Embedding::new(config.n_positions, config.n_embd);
        Self { config, wte, wpe, blocks, ln_f }
    }

    /// Full forward pass.
    /// token_ids: [seq_len]
    /// past_kv: one (K,V) tensor pair per layer from previous calls (for KV-cache)
    /// Returns (logits: [seq_len, vocab_size], new_kv: Vec<(Tensor, Tensor)>)
    pub fn forward(
        &self,
        token_ids: &[u32],
        position_offset: usize,
        past_kv: Option<&Vec<(Tensor, Tensor)>>,
    ) -> (Tensor, Vec<(Tensor, Tensor)>) {
        let seq = token_ids.len();

        // Token + position embeddings
        let tok_emb = self.wte.forward(token_ids);
        let pos_ids: Vec<u32> = (position_offset..position_offset + seq)
            .map(|p| p as u32)
            .collect();
        let pos_emb = self.wpe.forward(&pos_ids);

        // Add embeddings: x = tok_emb + pos_emb
        let mut x = tensor_add(&tok_emb, &pos_emb);

        // Add batch dimension: [seq, n_embd] → [1, seq, n_embd]
        x = x.view(vec![1, seq, self.config.n_embd]).unwrap();

        // Forward through transformer blocks
        let mut new_kv = Vec::with_capacity(self.config.n_layer);
        for (i, block) in self.blocks.iter().enumerate() {
            let (pk, pv) = match past_kv {
                Some(kv) => (Some(&kv[i].0), Some(&kv[i].1)),
                None => (None, None),
            };
            let (out, k, v) = block.forward(&x, pk, pv);
            x = out;
            new_kv.push((k, v));
        }

        // Final layer norm
        x = self.ln_f.forward(&x);

        // LM head (tied to wte weights): [1, seq, n_embd] × [n_embd, vocab_size]
        // We use wte.weight transposed as the output projection
        let x_2d = x.view(vec![seq, self.config.n_embd]).unwrap();
        let logits = helix_core::ops::matmul(&x_2d, &self.wte.weight.transpose().unwrap().contiguous()).unwrap();
        // logits: [seq, vocab_size]

        (logits, new_kv)
    }

    pub fn num_params(&self) -> u64 { self.config.num_params() }
}

fn tensor_add(a: &Tensor, b: &Tensor) -> Tensor {
    let numel = a.numel();
    let ad = a.as_slice::<f32>();
    let bd = b.as_slice::<f32>();
    let buf = helix_core::buffer::Buffer::zeros(numel * 4);
    let out = unsafe { buf.as_mut_slice::<f32>(0, numel) };
    for i in 0..numel { out[i] = ad[i] + bd[i]; }
    Tensor { data: buf, shape: a.shape().clone(), strides: a.strides().clone(), dtype: DType::F32, offset: 0, device: helix_core::Device::Cpu }
}
