use fancy_regex::Regex;

/// GPT-2's exact pre-tokenization regex.
/// Splits text into initial tokens before BPE is applied.
/// Handles contractions, spaces, letters, digits, punctuation separately.
pub struct PreTokenizer {
    pattern: Regex,
}

impl PreTokenizer {
    pub fn new() -> Self {
        // This is GPT-2's exact pattern from the original tiktoken/transformers implementation
        let pat = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
        Self {
            pattern: Regex::new(pat).expect("invalid pre-tokenizer regex"),
        }
    }

    /// Split text into pre-tokens.
    pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        self.pattern
            .find_iter(text)
            .filter_map(|m| m.ok())
            .map(|m| &text[m.start()..m.end()])
            .collect()
    }
}

impl Default for PreTokenizer {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_basic() {
        let pt = PreTokenizer::new();
        let tokens = pt.split("Hello world! It's a test.");
        assert!(tokens.contains(&"Hello"));
        assert!(tokens.contains(&"'s"));
    }
}
