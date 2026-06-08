use std::sync::Arc;
use crate::{
    buffer::Buffer,
    dtype::{DType, Scalar},
    error::{CoreError, CoreResult},
    shape::{Shape, Strides},
};

/// Device a tensor lives on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    // Future: Cuda(u8), Metal
}

/// The core tensor type. Immutable view into a shared buffer.
/// Cloning a Tensor is O(1) — it just increments the Arc refcount.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) data:    Arc<Buffer>,
    pub(crate) shape:   Shape,
    pub(crate) strides: Strides,
    pub(crate) dtype:   DType,
    pub(crate) offset:  usize,   // byte offset into buffer
    pub(crate) device:  Device,
}

impl Tensor {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create a zero tensor of given shape and dtype.
    pub fn zeros(shape: impl Into<Vec<usize>>, dtype: DType) -> Self {
        let shape = Shape::new(&shape.into());
        let nbytes = shape.numel() * dtype.size_of();
        let data = Buffer::zeros(nbytes);
        let strides = shape.strides();
        Self { data, shape, strides, dtype, offset: 0, device: Device::Cpu }
    }

    /// Create tensor from a Vec<f32>.
    pub fn from_vec_f32(v: Vec<f32>, shape: impl Into<Vec<usize>>) -> CoreResult<Self> {
        let shape = Shape::new(&shape.into());
        if v.len() != shape.numel() {
            return Err(CoreError::ShapeMismatch {
                expected: shape.dims().to_vec(),
                got: vec![v.len()],
            });
        }
        let nbytes = v.len() * 4;
        let buf = Buffer::zeros(nbytes);
        unsafe {
            let dst = buf.as_mut_slice::<f32>(0, v.len());
            dst.copy_from_slice(&v);
        }
        let strides = shape.strides();
        Ok(Self { data: buf, shape, strides, dtype: DType::F32, offset: 0, device: Device::Cpu })
    }

    /// Zero-copy wrap of a memory-mapped region.
    ///
    /// # Safety
    /// `ptr` must be valid for `len` bytes for as long as this tensor exists.
    pub unsafe fn from_raw_ptr(ptr: *mut u8, len: usize, shape: Shape, dtype: DType) -> Self {
        let data = Buffer::from_raw(ptr, len);
        let strides = shape.strides();
        Self { data, shape, strides, dtype, offset: 0, device: Device::Cpu }
    }

    // ── Metadata ─────────────────────────────────────────────────────────────

    #[inline] pub fn shape(&self) -> &Shape { &self.shape }
    #[inline] pub fn strides(&self) -> &Strides { &self.strides }
    #[inline] pub fn dtype(&self) -> DType { self.dtype }
    #[inline] pub fn ndim(&self) -> usize { self.shape.ndim() }
    #[inline] pub fn numel(&self) -> usize { self.shape.numel() }
    #[inline] pub fn device(&self) -> Device { self.device }

    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }

    // ── Data Access ───────────────────────────────────────────────────────────

    /// Get typed slice. Panics if dtype doesn't match T.
    pub fn as_slice<T: Scalar>(&self) -> &[T] {
        assert_eq!(self.dtype, T::dtype(), "dtype mismatch");
        // SAFETY: dtype check above, buffer valid for offset + numel * elem_size
        unsafe { self.data.as_slice::<T>(self.offset, self.numel()) }
    }

    /// Get mutable typed slice.
    pub fn as_mut_slice<T: Scalar>(&self) -> &mut [T] {
        assert_eq!(self.dtype, T::dtype(), "dtype mismatch");
        unsafe { self.data.as_mut_slice::<T>(self.offset, self.numel()) }
    }

    // ── Shape Manipulation ───────────────────────────────────────────────────

    /// Reshape without copying. Returns error if non-contiguous.
    pub fn view(&self, new_shape: impl Into<Vec<usize>>) -> CoreResult<Self> {
        if !self.is_contiguous() {
            return Err(CoreError::NonContiguous);
        }
        let new_shape = Shape::new(&new_shape.into());
        if new_shape.numel() != self.numel() {
            return Err(CoreError::ShapeMismatch {
                expected: vec![self.numel()],
                got: vec![new_shape.numel()],
            });
        }
        let new_strides = new_shape.strides();
        Ok(Self {
            data:    self.data.clone(),
            shape:   new_shape,
            strides: new_strides,
            dtype:   self.dtype,
            offset:  self.offset,
            device:  self.device,
        })
    }

    /// Transpose last two dimensions. Returns a non-contiguous view.
    pub fn transpose(&self) -> CoreResult<Self> {
        if self.ndim() < 2 {
            return Err(CoreError::DimOutOfBounds { axis: 1, ndim: self.ndim() });
        }
        let n = self.ndim();
        let mut new_dims = self.shape.dims().to_vec();
        new_dims.swap(n - 2, n - 1);
        let mut new_strides = self.strides.as_slice().to_vec();
        new_strides.swap(n - 2, n - 1);
        Ok(Self {
            data:    self.data.clone(),
            shape:   Shape::new(&new_dims),
            strides: Strides::new(&new_strides),
            dtype:   self.dtype,
            offset:  self.offset,
            device:  self.device,
        })
    }

    /// Make contiguous copy if non-contiguous. No-op if already contiguous.
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() { return self.clone(); }
        let nbytes = self.numel() * self.dtype.size_of();
        let buf = Buffer::zeros(nbytes);
        // Naive copy via f32 intermediate (only for F32 tensors for now)
        assert_eq!(self.dtype, DType::F32, "contiguous() only supports F32 currently");
        let src = unsafe { self.data.as_slice::<f32>(self.offset, self.numel()) };
        let dst = unsafe { buf.as_mut_slice::<f32>(0, self.numel()) };
        dst.copy_from_slice(src);
        let strides = self.shape.strides();
        Self { data: buf, shape: self.shape, strides, dtype: self.dtype, offset: 0, device: self.device }
    }

    /// Slice along the first dimension. Zero-copy view.
    pub fn slice(&self, start: usize, end: usize) -> CoreResult<Self> {
        let dim0 = self.shape.dim(0);
        if end > dim0 || start >= end {
            return Err(CoreError::IndexOutOfBounds { index: end, size: dim0 });
        }
        let row_stride = self.strides.as_slice()[0];
        let byte_offset = self.offset + start * row_stride * self.dtype.size_of();
        let mut new_dims = self.shape.dims().to_vec();
        new_dims[0] = end - start;
        Ok(Self {
            data:    self.data.clone(),
            shape:   Shape::new(&new_dims),
            strides: self.strides,
            dtype:   self.dtype,
            offset:  byte_offset,
            device:  self.device,
        })
    }

    // ── Debug ─────────────────────────────────────────────────────────────────

    pub fn debug_print(&self, name: &str) {
        println!(
            "Tensor[{}] shape={} dtype={} contiguous={}",
            name, self.shape, self.dtype, self.is_contiguous()
        );
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={}, dtype={})", self.shape, self.dtype)
    }
}
