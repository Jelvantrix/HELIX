use serde::{Deserialize, Serialize};

/// GPT-2 model configuration. Controls all architectural dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size:     usize,
    pub n_positions:    usize,  // max sequence length (context window)
    pub n_embd:         usize,  // embedding dimension
    pub n_layer:        usize,  // number of transformer blocks
    pub n_head:         usize,  // number of attention heads
    pub n_inner:        Option<usize>,  // MLP hidden dim (default: 4 * n_embd)
    pub layer_norm_eps: f32,
    pub model_type:     String,
}

impl ModelConfig {
    /// GPT-2 Small (117M parameters)
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size:     50257,
            n_positions:    1024,
            n_embd:         768,
            n_layer:        12,
            n_head:         12,
            n_inner:        None,
            layer_norm_eps: 1e-5,
            model_type:     "gpt2".into(),
        }
    }

    /// GPT-2 Medium (345M parameters)
    pub fn gpt2_medium() -> Self {
        Self {
            vocab_size:     50257,
            n_positions:    1024,
            n_embd:         1024,
            n_layer:        24,
            n_head:         16,
            n_inner:        None,
            layer_norm_eps: 1e-5,
            model_type:     "gpt2-medium".into(),
        }
    }

    /// GPT-2 Large (762M parameters)
    pub fn gpt2_large() -> Self {
        Self {
            vocab_size:     50257,
            n_positions:    1024,
            n_embd:         1280,
            n_layer:        36,
            n_head:         20,
            n_inner:        None,
            layer_norm_eps: 1e-5,
            model_type:     "gpt2-large".into(),
        }
    }

    /// GPT-2 XL (1.5B parameters)
    pub fn gpt2_xl() -> Self {
        Self {
            vocab_size:     50257,
            n_positions:    1024,
            n_embd:         1600,
            n_layer:        48,
            n_head:         25,
            n_inner:        None,
            layer_norm_eps: 1e-5,
            model_type:     "gpt2-xl".into(),
        }
    }

    pub fn n_inner(&self) -> usize {
        self.n_inner.unwrap_or(4 * self.n_embd)
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Approximate parameter count
    pub fn num_params(&self) -> u64 {
        let embd  = (self.vocab_size + self.n_positions) * self.n_embd;
        let block = (4 * self.n_embd * self.n_embd       // QKV + proj
                   + 8 * self.n_embd * self.n_inner()) * self.n_layer;
        let head  = self.vocab_size * self.n_embd;
        (embd + block + head) as u64
    }
}
