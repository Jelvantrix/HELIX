use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::buffer::Buffer;

/// Layer normalization.
/// y = (x - mean) / sqrt(var + eps) * weight + bias
/// Online (single-pass) mean and variance via Welford's algorithm.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    assert_eq!(x.dtype(), DType::F32);
    assert_eq!(weight.dtype(), DType::F32);
    assert_eq!(bias.dtype(), DType::F32);

    let shape = x.shape;
    let ndim  = x.ndim();
    let hidden = shape.dim(ndim - 1);
    let rows   = x.numel() / hidden;

    let x_data = x.as_slice::<f32>();
    let w_data = weight.as_slice::<f32>();
    let b_data = bias.as_slice::<f32>();
    let buf = Buffer::zeros(x.numel() * 4);
    let out = unsafe { buf.as_mut_slice::<f32>(0, x.numel()) };

    for row in 0..rows {
        let start = row * hidden;

        // Welford's online mean & variance
        let mut mean = 0.0f32;
        let mut m2   = 0.0f32;
        for (n, &v) in x_data[start..start + hidden].iter().enumerate() {
            let delta = v - mean;
            mean += delta / (n + 1) as f32;
            let delta2 = v - mean;
            m2 += delta * delta2;
        }
        let variance = m2 / hidden as f32;
        let inv_std  = 1.0 / (variance + eps).sqrt();

        for i in 0..hidden {
            let normalized = (x_data[start + i] - mean) * inv_std;
            out[start + i] = normalized * w_data[i] + b_data[i];
        }
    }

    Tensor {
        data:    buf,
        shape,
        strides: shape.strides(),
        dtype:   DType::F32,
        offset:  0,
        device:  crate::tensor::Device::Cpu,
    }
}

/// Root mean square normalization (used in Llama, Mistral).
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    assert_eq!(x.dtype(), DType::F32);
    let shape  = x.shape;
    let ndim   = x.ndim();
    let hidden = shape.dim(ndim - 1);
    let rows   = x.numel() / hidden;
    let x_data = x.as_slice::<f32>();
    let w_data = weight.as_slice::<f32>();
    let buf = Buffer::zeros(x.numel() * 4);
    let out = unsafe { buf.as_mut_slice::<f32>(0, x.numel()) };
    for row in 0..rows {
        let start = row * hidden;
        let rms: f32 = (x_data[start..start + hidden].iter().map(|&v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..hidden {
            out[start + i] = x_data[start + i] * inv_rms * w_data[i];
        }
    }
    Tensor { data: buf, shape, strides: shape.strides(), dtype: DType::F32, offset: 0, device: crate::tensor::Device::Cpu }
}
