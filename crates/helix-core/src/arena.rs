use std::cell::UnsafeCell;
use std::alloc::{alloc, dealloc, Layout};
use crate::error::{CoreError, CoreResult};
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::shape::Shape;

/// A bump allocator for short-lived tensors during a single forward pass.
/// Allocate freely; reset in O(1) at end of pass. Zero fragmentation.
pub struct Arena {
    ptr:      *mut u8,
    layout:   Layout,
    cursor:   UnsafeCell<usize>,
    capacity: usize,
}

// SAFETY: Arena is not Send/Sync by default due to raw ptr. Mark it explicitly.
// Caller must ensure single-threaded access during a forward pass.
unsafe impl Send for Arena {}

impl Arena {
    pub fn new(capacity_bytes: usize) -> Self {
        let layout = Layout::from_size_align(capacity_bytes, 64).unwrap();
        let ptr = unsafe {
            let p = alloc(layout);
            assert!(!p.is_null(), "arena allocation failed");
            p
        };
        Self {
            ptr,
            layout,
            cursor: UnsafeCell::new(0),
            capacity: capacity_bytes,
        }
    }

    /// Allocate `nbytes` from the arena, returning a raw pointer.
    /// Aligned to 64 bytes.
    pub fn alloc(&self, nbytes: usize) -> CoreResult<*mut u8> {
        let cursor = unsafe { &mut *self.cursor.get() };
        // Align cursor to 64 bytes
        let aligned = (*cursor + 63) & !63;
        let next = aligned + nbytes;
        if next > self.capacity {
            return Err(CoreError::ArenaExhausted {
                requested: nbytes,
                available: self.capacity - *cursor,
            });
        }
        *cursor = next;
        Ok(unsafe { self.ptr.add(aligned) })
    }

    /// Allocate a zeroed tensor backed by arena memory.
    pub fn alloc_tensor(&self, shape: Shape, dtype: DType) -> CoreResult<Tensor> {
        let nbytes = shape.numel() * dtype.size_of();
        let ptr = self.alloc(nbytes)?;
        // Zero the memory
        unsafe { std::ptr::write_bytes(ptr, 0, nbytes); }
        // SAFETY: ptr lives as long as Arena; shape/nbytes are consistent.
        Ok(unsafe { Tensor::from_raw_ptr(ptr, nbytes, shape, dtype) })
    }

    /// Reset the arena. All previously allocated memory is considered freed.
    /// Any tensors pointing into this arena must be dropped before calling reset.
    pub fn reset(&self) {
        unsafe { *self.cursor.get() = 0; }
    }

    pub fn used_bytes(&self) -> usize {
        unsafe { *self.cursor.get() }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn utilization(&self) -> f32 {
        self.used_bytes() as f32 / self.capacity as f32
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout); }
    }
}
