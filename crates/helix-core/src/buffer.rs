use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::Arc;

/// Raw heap-allocated byte buffer.
/// Wrapped in Arc so tensors can share the same buffer with zero-copy slicing.
#[derive(Debug)]
pub struct Buffer {
    ptr:    NonNull<u8>,
    len:    usize,
    layout: Layout,
    owned:  bool,
}

// SAFETY: we manage the memory manually and guarantee no aliased mutable refs.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Allocate a zeroed buffer of `len` bytes, aligned to 64 bytes (cache line).
    pub fn zeros(len: usize) -> Arc<Self> {
        let layout = Layout::from_size_align(len.max(1), 64).unwrap();
        // SAFETY: layout is valid, we check non-null.
        let ptr = unsafe {
            let p = alloc(layout);
            assert!(!p.is_null(), "allocation failed");
            std::ptr::write_bytes(p, 0, len);
            NonNull::new_unchecked(p)
        };
        Arc::new(Self { ptr, len, layout, owned: true })
    }

    /// Wrap a raw pointer (e.g., from a memory-mapped file). Not owned.
    ///
    /// # Safety
    /// Caller must ensure `ptr` is valid for `len` bytes for the lifetime of this buffer.
    pub unsafe fn from_raw(ptr: *mut u8, len: usize) -> Arc<Self> {
        Arc::new(Self {
            ptr:    NonNull::new_unchecked(ptr),
            len,
            layout: Layout::from_size_align(len.max(1), 1).unwrap(),
            owned:  false,
        })
    }

    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn as_ptr(&self) -> *const u8 { self.ptr.as_ptr() }
    #[inline] pub fn as_mut_ptr(&self) -> *mut u8 { self.ptr.as_ptr() }

    /// Get typed slice view.
    ///
    /// # Safety
    /// Caller must ensure T is correct for the buffer's dtype, offset + count fits.
    pub unsafe fn as_slice<T>(&self, byte_offset: usize, count: usize) -> &[T] {
        let ptr = self.ptr.as_ptr().add(byte_offset) as *const T;
        std::slice::from_raw_parts(ptr, count)
    }

    /// Get mutable typed slice view.
    ///
    /// # Safety
    /// Same as as_slice, plus unique access must be guaranteed by caller.
    pub unsafe fn as_mut_slice<T>(&self, byte_offset: usize, count: usize) -> &mut [T] {
        let ptr = self.ptr.as_ptr().add(byte_offset) as *mut T;
        std::slice::from_raw_parts_mut(ptr, count)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.owned {
            // SAFETY: ptr was allocated with this exact layout.
            unsafe { dealloc(self.ptr.as_ptr(), self.layout); }
        }
    }
}
