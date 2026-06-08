/// Special tokens used by GPT-2 and compatible models.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub eos: u32,
    pub bos: Option<u32>,
    pub pad: Option<u32>,
    pub unk: Option<u32>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        // GPT-2 defaults
        Self {
            eos: 50256,   // <|endoftext|>
            bos: None,
            pad: None,
            unk: None,
        }
    }
}
