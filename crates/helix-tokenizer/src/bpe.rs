use std::collections::HashMap;
use std::path::Path;
use crate::error::{TokenizerError, TokenizerResult};

/// A BPE merge rule: (left, right) → merged token
#[derive(Debug, Clone)]
pub struct MergeRule {
    pub left:  String,
    pub right: String,
    pub rank:  usize,   // lower = higher priority
}

/// BPE merge table loaded from vocab.bpe
pub struct BpeRules {
    /// (left, right) → rank
    merges: HashMap<(String, String), usize>,
}

impl BpeRules {
    pub fn from_file(path: impl AsRef<Path>) -> TokenizerResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|_| TokenizerError::VocabNotFound {
                path: path.as_ref().display().to_string(),
            })?;

        let mut merges = HashMap::new();

        for (rank, line) in content.lines().enumerate().skip(1) {  // skip header line
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() != 2 {
                return Err(TokenizerError::InvalidMergeRule(line.to_string()));
            }
            merges.insert((parts[0].to_string(), parts[1].to_string()), rank);
        }

        Ok(Self { merges })
    }

    pub fn rank(&self, left: &str, right: &str) -> Option<usize> {
        self.merges.get(&(left.to_string(), right.to_string())).copied()
    }
}

/// Core BPE encoder.
/// Takes a word (Vec of unicode chars as strings) and applies merges
/// until no more can be applied.
pub fn bpe_encode_word(word: Vec<String>, rules: &BpeRules) -> Vec<String> {
    if word.len() <= 1 { return word; }

    let mut parts = word;

    loop {
        // Find the lowest-rank (highest priority) merge pair
        let best = parts
            .windows(2)
            .enumerate()
            .filter_map(|(i, w)| {
                rules.rank(&w[0], &w[1]).map(|rank| (i, rank))
            })
            .min_by_key(|&(_, rank)| rank);

        match best {
            None => break,  // No more merges possible
            Some((idx, _)) => {
                // Apply the merge at idx
                let merged = format!("{}{}", parts[idx], parts[idx + 1]);
                parts.remove(idx + 1);
                parts[idx] = merged;
            }
        }
    }

    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_single_char() {
        let rules = BpeRules { merges: HashMap::new() };
        let result = bpe_encode_word(vec!["h".into(), "i".into()], &rules);
        assert_eq!(result, vec!["h", "i"]);
    }
}
