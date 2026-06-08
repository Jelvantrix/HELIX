/// Maximum number of dimensions supported.
pub const MAX_DIMS: usize = 6;

/// Stack-allocated shape — no heap allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [usize; MAX_DIMS],
    ndim: usize,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        assert!(dims.len() <= MAX_DIMS, "too many dimensions");
        let mut arr = [0usize; MAX_DIMS];
        arr[..dims.len()].copy_from_slice(dims);
        Self { dims: arr, ndim: dims.len() }
    }

    #[inline] pub fn ndim(&self) -> usize { self.ndim }
    #[inline] pub fn dims(&self) -> &[usize] { &self.dims[..self.ndim] }

    pub fn numel(&self) -> usize {
        self.dims().iter().product()
    }

    pub fn dim(&self, axis: usize) -> usize {
        assert!(axis < self.ndim, "axis {axis} out of bounds");
        self.dims[axis]
    }

    /// C-contiguous (row-major) strides for this shape.
    pub fn strides(&self) -> Strides {
        let mut strides = [0usize; MAX_DIMS];
        if self.ndim == 0 { return Strides::new(&[]); }
        strides[self.ndim - 1] = 1;
        for i in (0..self.ndim - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        Strides { data: strides, ndim: self.ndim }
    }

    /// Try to broadcast two shapes together (NumPy rules).
    pub fn broadcast(a: &Shape, b: &Shape) -> Option<Shape> {
        let ndim = a.ndim.max(b.ndim);
        let mut out = [0usize; MAX_DIMS];
        for i in 0..ndim {
            let ai = if i < ndim - a.ndim { 1 } else { a.dims[i - (ndim - a.ndim)] };
            let bi = if i < ndim - b.ndim { 1 } else { b.dims[i - (ndim - b.ndim)] };
            out[i] = match (ai, bi) {
                (x, y) if x == y => x,
                (1, y) => y,
                (x, 1) => x,
                _ => return None,
            };
        }
        Some(Shape { dims: out, ndim })
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims().iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

// ── Strides ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strides {
    data: [usize; MAX_DIMS],
    ndim: usize,
}

impl Strides {
    pub fn new(s: &[usize]) -> Self {
        let mut data = [0usize; MAX_DIMS];
        data[..s.len()].copy_from_slice(s);
        Self { data, ndim: s.len() }
    }

    #[inline] pub fn as_slice(&self) -> &[usize] { &self.data[..self.ndim] }

    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        let expected = shape.strides();
        self.data[..self.ndim] == expected.data[..expected.ndim]
    }
}
