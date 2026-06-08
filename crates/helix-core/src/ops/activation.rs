use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::buffer::Buffer;
use crate::tensor::Device;

/// GELU activation (exact, using erf).
/// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
pub fn gelu(t: &Tensor) -> Tensor {
    assert_eq!(t.dtype(), DType::F32);
    let data = t.as_slice::<f32>();
    let buf = Buffer::zeros(data.len() * 4);
    let out = unsafe { buf.as_mut_slice::<f32>(0, data.len()) };
    for (o, &x) in out.iter_mut().zip(data.iter()) {
        *o = 0.5 * x * (1.0 + libm_erf((x / std::f32::consts::SQRT_2) as f64) as f32);
    }
    Tensor { data: buf, shape: t.shape, strides: t.strides, dtype: DType::F32, offset: 0, device: Device::Cpu }
}

/// Fast approximate GELU using tanh polynomial.
/// gelu_approx(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub fn gelu_approx(t: &Tensor) -> Tensor {
    assert_eq!(t.dtype(), DType::F32);
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    const COEFF: f32 = 0.044715;
    let data = t.as_slice::<f32>();
    let buf = Buffer::zeros(data.len() * 4);
    let out = unsafe { buf.as_mut_slice::<f32>(0, data.len()) };
    for (o, &x) in out.iter_mut().zip(data.iter()) {
        let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        *o = 0.5 * x * (1.0 + inner.tanh());
    }
    Tensor { data: buf, shape: t.shape, strides: t.strides, dtype: DType::F32, offset: 0, device: Device::Cpu }
}

/// Numerically stable softmax along the last dimension.
pub fn softmax(t: &Tensor) -> Tensor {
    assert_eq!(t.dtype(), DType::F32);
    let shape = t.shape;
    let ndim = t.ndim();
    let last_dim = shape.dim(ndim - 1);
    let rows = t.numel() / last_dim;
    let data = t.as_slice::<f32>();
    let buf = Buffer::zeros(data.len() * 4);
    let out = unsafe { buf.as_mut_slice::<f32>(0, data.len()) };

    for row in 0..rows {
        let start = row * last_dim;
        let end = start + last_dim;
        let row_data = &data[start..end];
        let row_out = &mut out[start..end];

        // Subtract max for numerical stability
        let max_val = row_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (o, &x) in row_out.iter_mut().zip(row_data.iter()) {
            *o = (x - max_val).exp();
            sum += *o;
        }
        let inv_sum = 1.0 / sum;
        for o in row_out.iter_mut() { *o *= inv_sum; }
    }

    Tensor { data: buf, shape, strides: shape.strides(), dtype: DType::F32, offset: 0, device: Device::Cpu }
}

/// Element-wise sigmoid: σ(x) = 1 / (1 + e^-x)
pub fn sigmoid(t: &Tensor) -> Tensor {
    assert_eq!(t.dtype(), DType::F32);
    let data = t.as_slice::<f32>();
    let buf = Buffer::zeros(data.len() * 4);
    let out = unsafe { buf.as_mut_slice::<f32>(0, data.len()) };
    for (o, &x) in out.iter_mut().zip(data.iter()) {
        *o = 1.0 / (1.0 + (-x).exp());
    }
    Tensor { data: buf, shape: t.shape, strides: t.strides, dtype: DType::F32, offset: 0, device: Device::Cpu }
}

/// Minimal erf via libm approximation (avoids linking libm on all platforms).
fn libm_erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 approximation
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429))));
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * (1.0 - poly * (-x * x).exp())
}
