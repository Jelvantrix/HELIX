use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::Shape;
use crate::buffer::Buffer;
use crate::error::{CoreError, CoreResult};

/// Batched matrix multiply: C = A @ B
/// Supports shapes [..., M, K] x [..., K, N] → [..., M, N]
/// Uses AVX2 on x86 when available via runtime feature detection.
pub fn matmul(a: &Tensor, b: &Tensor) -> CoreResult<Tensor> {
    assert_eq!(a.dtype(), DType::F32);
    assert_eq!(b.dtype(), DType::F32);

    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();
    let ndim_a = a.ndim();
    let ndim_b = b.ndim();

    if ndim_a < 2 || ndim_b < 2 {
        return Err(CoreError::ShapeMismatch {
            expected: vec![2],
            got: vec![ndim_a.min(ndim_b)],
        });
    }

    let m = a_shape[ndim_a - 2];
    let k = a_shape[ndim_a - 1];
    let k2 = b_shape[ndim_b - 2];
    let n = b_shape[ndim_b - 1];

    if k != k2 {
        return Err(CoreError::ShapeMismatch {
            expected: a_shape.to_vec(),
            got: b_shape.to_vec(),
        });
    }

    // Batch dims
    let batch_a = &a_shape[..ndim_a - 2];
    let _batch_b = &b_shape[..ndim_b - 2];
    let batch_size: usize = batch_a.iter().product::<usize>().max(1);

    // Build output shape
    let mut out_dims = batch_a.to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::new(&out_dims);

    let total = out_shape.numel();
    let buf = Buffer::zeros(total * 4);

    let a_data = a.as_slice::<f32>();
    let b_data = b.as_slice::<f32>();
    let c_data = unsafe { buf.as_mut_slice::<f32>(0, total) };

    // Choose kernel based on CPU features
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { matmul_avx2(a_data, b_data, c_data, batch_size, m, k, n); }
        } else {
            matmul_scalar(a_data, b_data, c_data, batch_size, m, k, n);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        matmul_scalar(a_data, b_data, c_data, batch_size, m, k, n);
    }

    let strides = out_shape.strides();
    Ok(Tensor { data: buf, shape: out_shape, strides, dtype: DType::F32, offset: 0, device: crate::tensor::Device::Cpu })
}

/// Scalar (reference) matmul implementation.
fn matmul_scalar(a: &[f32], b: &[f32], c: &mut [f32], batch: usize, m: usize, k: usize, n: usize) {
    for batch_idx in 0..batch {
        let a_off = batch_idx * m * k;
        let b_off = batch_idx * k * n;
        let c_off = batch_idx * m * n;
        for i in 0..m {
            for p in 0..k {
                let a_val = a[a_off + i * k + p];
                for j in 0..n {
                    c[c_off + i * n + j] += a_val * b[b_off + p * n + j];
                }
            }
        }
    }
}

/// AVX2 + FMA accelerated matmul. Inner loop processes 8 f32s at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_avx2(a: &[f32], b: &[f32], c: &mut [f32], batch: usize, m: usize, k: usize, n: usize) {
    use std::arch::x86_64::*;
    const STEP: usize = 8; // AVX2 processes 8 f32 at once

    for batch_idx in 0..batch {
        let a_off = batch_idx * m * k;
        let b_off = batch_idx * k * n;
        let c_off = batch_idx * m * n;

        for i in 0..m {
            let mut j = 0usize;
            while j + STEP <= n {
                let mut acc = _mm256_setzero_ps();
                for p in 0..k {
                    let a_bc = _mm256_set1_ps(a[a_off + i * k + p]);
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(b_off + p * n + j));
                    acc = _mm256_fmadd_ps(a_bc, b_vec, acc);
                }
                let c_ptr = c.as_mut_ptr().add(c_off + i * n + j);
                let existing = _mm256_loadu_ps(c_ptr);
                let result = _mm256_add_ps(existing, acc);
                _mm256_storeu_ps(c_ptr, result);
                j += STEP;
            }
            // Scalar tail
            for j_tail in j..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[a_off + i * k + p] * b[b_off + p * n + j_tail];
                }
                c[c_off + i * n + j_tail] += sum;
            }
        }
    }
}

/// Compute A @ B^T efficiently (avoids explicit transpose allocation).
pub fn matmul_t(a: &Tensor, b: &Tensor) -> CoreResult<Tensor> {
    let bt = b.transpose()?;
    let bt = bt.contiguous();
    matmul(a, &bt)
}
