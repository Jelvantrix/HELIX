//! # helix-tokenizer
//!
//! Byte-pair encoding tokenizer for GPT-2.
//! Loads directly from OpenAI's vocab.bpe and encoder.json.
//! No dependency on HuggingFace tokenizers crate.

pub mod error;
pub mod vocab;
pub mod pretokenize;
pub mod bpe;
pub mod special;

use std::path::Path;
use error::{TokenizerError, TokenizerResult};
use vocab::Vocab;
use pretokenize::PreTokenizer;
use bpe::{BpeRules, bpe_encode_word};
use special::SpecialTokens;

/// The main tokenizer. Owns vocab, BPE rules, and pre-tokenizer.
pub struct Tokenizer {
    vocab:          Vocab,
    bpe:            BpeRules,
    pretokenizer:   PreTokenizer,
    pub special:    SpecialTokens,
    /// Cache: word string → Vec<u32> to avoid re-running BPE on seen words
    cache:          std::collections::HashMap<String, Vec<u32>>,
}

impl Tokenizer {
    /// Load tokenizer from a directory containing encoder.json and vocab.bpe
    pub fn from_dir(dir: impl AsRef<Path>) -> TokenizerResult<Self> {
        let dir = dir.as_ref();
        let vocab = Vocab::from_file(dir.join("encoder.json"))?;
        let bpe   = BpeRules::from_file(dir.join("vocab.bpe"))?;
        Ok(Self {
            vocab,
            bpe,
            pretokenizer: PreTokenizer::new(),
            special: SpecialTokens::default(),
            cache: std::collections::HashMap::new(),
        })
    }

    /// Encode a string into token IDs.
    pub fn encode(&mut self, text: &str) -> TokenizerResult<Vec<u32>> {
        let words = self.pretokenizer.split(text);
        let mut ids = Vec::new();

        for word in words {
            if let Some(cached) = self.cache.get(word) {
                ids.extend_from_slice(cached);
                continue;
            }

            // Convert word to byte-level unicode chars
            let chars: Vec<String> = word
                .bytes()
                .map(|b| {
                    // Use GPT-2 byte encoder: every byte maps to a unicode char
                    char::from_u32(b as u32).unwrap_or('\u{FFFD}').to_string()
                })
                .collect();

            let bpe_tokens = bpe_encode_word(chars, &self.bpe);
            let token_ids: Vec<u32> = bpe_tokens
                .iter()
                .map(|t| {
                    self.vocab.encode_token(t).ok_or_else(|| {
                        TokenizerError::InvalidVocab(format!("unknown token: {t}"))
                    })
                })
                .collect::<TokenizerResult<Vec<_>>>()?;

            self.cache.insert(word.to_string(), token_ids.clone());
            ids.extend(token_ids);
        }

        Ok(ids)
    }

    /// Decode a sequence of token IDs back to a string.
    pub fn decode(&self, ids: &[u32]) -> TokenizerResult<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let token = self.vocab.decode_token(id).ok_or(TokenizerError::UnknownTokenId(id))?;
            for ch in token.chars() {
                if let Some(&byte) = self.vocab.byte_decoder.get(&ch) {
                    bytes.push(byte);
                }
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn vocab_size(&self) -> usize { self.vocab.vocab_size() }
    pub fn eos_token(&self) -> u32 { self.special.eos }
}
