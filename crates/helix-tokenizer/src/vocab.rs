use std::collections::HashMap;
use std::path::Path;
use serde_json::Value;
use crate::error::{TokenizerError, TokenizerResult};

/// Bidirectional vocabulary: token string ↔ token id
pub struct Vocab {
    /// token string → id
    pub encoder: HashMap<String, u32>,
    /// id → token string
    pub decoder: HashMap<u32, String>,
    /// byte-level fallback: byte → unicode char (GPT-2 byte-to-unicode mapping)
    pub byte_decoder: HashMap<char, u8>,
}

impl Vocab {
    /// Load from GPT-2's encoder.json
    pub fn from_file(path: impl AsRef<Path>) -> TokenizerResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|_| TokenizerError::VocabNotFound {
                path: path.as_ref().display().to_string(),
            })?;
        let json: Value = serde_json::from_str(&content)
            .map_err(|e| TokenizerError::InvalidVocab(e.to_string()))?;

        let obj = json.as_object().ok_or_else(|| {
            TokenizerError::InvalidVocab("root is not an object".into())
        })?;

        let mut encoder = HashMap::with_capacity(obj.len());
        let mut decoder = HashMap::with_capacity(obj.len());

        for (token, id_val) in obj {
            let id = id_val.as_u64()
                .ok_or_else(|| TokenizerError::InvalidVocab(format!("non-integer id for {token}")))?
                as u32;
            encoder.insert(token.clone(), id);
            decoder.insert(id, token.clone());
        }

        Ok(Self {
            byte_decoder: build_byte_decoder(),
            encoder,
            decoder,
        })
    }

    pub fn encode_token(&self, s: &str) -> Option<u32> {
        self.encoder.get(s).copied()
    }

    pub fn decode_token(&self, id: u32) -> Option<&str> {
        self.decoder.get(&id).map(|s| s.as_str())
    }

    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }
}

/// GPT-2's byte-to-unicode mapping.
/// Maps every possible byte (0–255) to a printable unicode char so that
/// BPE can operate on a "clean" unicode string.
fn build_byte_decoder() -> HashMap<char, u8> {
    let printable: Vec<u8> = (b'!'..=b'~')
        .chain(b'\xA1'..=b'\xAC')
        .chain(b'\xAE'..=b'\xFF')
        .collect();

    let mut decoder = HashMap::new();
    let mut n = 0u8;

    for b in 0u8..=255u8 {
        let ch = if printable.contains(&b) {
            b as char
        } else {
            // Map to high Unicode private use area
            let c = char::from_u32(256 + n as u32).unwrap();
            n += 1;
            c
        };
        decoder.insert(ch, b);
    }
    decoder
}
