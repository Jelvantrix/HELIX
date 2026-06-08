use half::{bf16, f16};

/// Supported element types for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    U8,
    Bool,
}

impl DType {
    /// Size of one element in bytes.
    #[inline]
    pub const fn size_of(self) -> usize {
        match self {
            DType::F32  => 4,
            DType::F16  => 2,
            DType::BF16 => 2,
            DType::I32  => 4,
            DType::U8   => 1,
            DType::Bool => 1,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            DType::F32  => "f32",
            DType::F16  => "f16",
            DType::BF16 => "bf16",
            DType::I32  => "i32",
            DType::U8   => "u8",
            DType::Bool => "bool",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ── Scalar trait ─────────────────────────────────────────────────────────────

/// Marker trait for types that can be stored in a Tensor.
pub trait Scalar: Copy + Send + Sync + 'static {
    fn dtype() -> DType;
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}

impl Scalar for f32 {
    fn dtype() -> DType { DType::F32 }
    fn to_f32(self) -> f32 { self }
    fn from_f32(v: f32) -> Self { v }
}

impl Scalar for f16 {
    fn dtype() -> DType { DType::F16 }
    fn to_f32(self) -> f32 { self.to_f32() }
    fn from_f32(v: f32) -> Self { f16::from_f32(v) }
}

impl Scalar for bf16 {
    fn dtype() -> DType { DType::BF16 }
    fn to_f32(self) -> f32 { self.to_f32() }
    fn from_f32(v: f32) -> Self { bf16::from_f32(v) }
}

impl Scalar for i32 {
    fn dtype() -> DType { DType::I32 }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f32(v: f32) -> Self { v as i32 }
}

impl Scalar for u8 {
    fn dtype() -> DType { DType::U8 }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f32(v: f32) -> Self { v as u8 }
}
