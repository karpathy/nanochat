//! Candle implementation of nanoAgent's GPT inference path.
//!
//! This first port deliberately mirrors the checkpoint's parameter names and
//! has a simple causal-attention fallback. A KV-cached kernel is added in the
//! inference-engine layer; this forward method is also the reference path.

use std::{collections::HashSet, path::Path};

use anyhow::{Result, ensure};
use candle_core::{CpuStorage, CustomOp1, CustomOp2, DType, Device, Layout, Shape, Tensor};
use candle_nn::{Embedding, Init, Linear, Module, VarBuilder, VarMap, ops};

use crate::model_config::GptConfig;

const ROPE_BASE: f64 = 100_000.0;
const LOGIT_SOFTCAP: f64 = 15.0;
const RMS_EPS_F32: f64 = f32::EPSILON as f64;

/// Mirror the pinned WSL PyTorch CPU FP32 reduction, whose near-zero backout
/// scalar gradients can otherwise change sign across runtimes.
struct BackoutMul;

impl CustomOp2 for BackoutMul {
    fn name(&self) -> &'static str {
        "backout_mul"
    }

    fn cpu_fwd(
        &self,
        lhs: &CpuStorage,
        lhs_layout: &Layout,
        rhs: &CpuStorage,
        rhs_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(lhs), CpuStorage::F32(rhs)) = (lhs, rhs) else {
            candle_core::bail!("backout_mul expects FP32 CPU tensors")
        };
        let dims = lhs_layout.dims();
        if dims.len() != 3 || rhs_layout.dims().len() != 3 {
            candle_core::bail!("backout_mul expects rank-3 tensors")
        }
        let mut output = Vec::with_capacity(dims.iter().product());
        for b in 0..dims[0] {
            for t in 0..dims[1] {
                for c in 0..dims[2] {
                    let lhs_idx = lhs_layout.start_offset()
                        + b * lhs_layout.stride()[0]
                        + t * lhs_layout.stride()[1]
                        + c * lhs_layout.stride()[2];
                    let rhs_idx = rhs_layout.start_offset()
                        + usize::from(rhs_layout.dims()[0] != 1) * b * rhs_layout.stride()[0]
                        + usize::from(rhs_layout.dims()[1] != 1) * t * rhs_layout.stride()[1]
                        + usize::from(rhs_layout.dims()[2] != 1) * c * rhs_layout.stride()[2];
                    output.push(lhs[lhs_idx] * rhs[rhs_idx]);
                }
            }
        }
        Ok((
            CpuStorage::F32(output),
            Shape::from((dims[0], dims[1], dims[2])),
        ))
    }

    fn bwd(
        &self,
        lhs: &Tensor,
        rhs: &Tensor,
        _output: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        let lhs_grad = rhs.broadcast_mul(grad)?;
        let products = lhs.mul(grad)?.flatten_all()?.to_vec1::<f32>()?;
        let scalar_grad = pytorch_cpu_f32_sum(&products);
        let rhs_grad = Tensor::from_vec(vec![scalar_grad], rhs.dims().to_vec(), rhs.device())?;
        Ok((Some(lhs_grad), Some(rhs_grad)))
    }
}

fn has_pytorch_cpu_f32_reduction_layout() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// CPU FP32 cross entropy with a fused PyTorch-compatible softmax backward.
struct CpuCrossEntropy {
    targets: Vec<u32>,
}

impl CustomOp1 for CpuCrossEntropy {
    fn name(&self) -> &'static str {
        "cpu_cross_entropy"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(values) = storage else {
            candle_core::bail!("cpu_cross_entropy expects f32 logits")
        };
        let [rows, classes] = layout.dims() else {
            candle_core::bail!("cpu_cross_entropy expects rank-2 logits")
        };
        if *rows != self.targets.len() || !layout.is_contiguous() {
            candle_core::bail!("cpu_cross_entropy target or layout mismatch")
        }
        let start = layout.start_offset();
        let mut loss = 0.0f32;
        for (row, &target) in self.targets.iter().enumerate() {
            if target as usize >= *classes {
                candle_core::bail!("cpu_cross_entropy target outside class range")
            }
            let logits = &values[start + row * classes..start + (row + 1) * classes];
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp = logits.iter().map(|value| (*value - max).exp()).sum::<f32>();
            loss += (sum_exp.ln() + max) - logits[target as usize];
        }
        Ok((CpuStorage::F32(vec![loss / *rows as f32]), Shape::from(())))
    }

    fn bwd(
        &self,
        input: &Tensor,
        _output: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let log_probs = candle_nn::ops::log_softmax(input, 1)?.to_vec2::<f32>()?;
        let mut probabilities = log_probs
            .into_iter()
            .map(|row| row.into_iter().map(pytorch_cpu_exp_f32).collect())
            .collect::<Vec<Vec<f32>>>();
        if probabilities.len() != self.targets.len() {
            candle_core::bail!("cpu_cross_entropy target count mismatch")
        }
        let scale = grad.to_scalar::<f32>()? / probabilities.len() as f32;
        let mut gradient = Vec::with_capacity(input.elem_count());
        for (row, &target) in probabilities.iter_mut().zip(&self.targets) {
            if target as usize >= row.len() {
                candle_core::bail!("cpu_cross_entropy target outside class range")
            }
            let summed_grad = -scale;
            let target_grad = -scale;
            for (class, probability) in row.iter_mut().enumerate() {
                let output_grad = if class == target as usize {
                    target_grad
                } else {
                    0.0
                };
                gradient.push(output_grad - *probability * summed_grad);
            }
        }
        Ok(Some(Tensor::from_vec(
            gradient,
            input.shape().clone(),
            input.device(),
        )?))
    }
}

/// SLEEF's scalar `expf` polynomial used to mirror PyTorch's AVX2 CPU exp.
/// Algorithm and constants are adapted from SLEEF, dual-licensed MIT/Apache-2.0.
fn pytorch_cpu_exp_f32(value: f32) -> f32 {
    const R_LN2: f32 = 1.442_695_040_888_963_4;
    const LN2_UPPER: f32 = 0.693_145_751_953_125;
    const LN2_LOWER: f32 = 1.428_606_765_330_187e-6;

    if value < -104.0 {
        return 0.0;
    }
    if value > 104.0 {
        return f32::INFINITY;
    }
    let qf = (value * R_LN2 + if value >= 0.0 { 0.5 } else { -0.5 }).trunc();
    let q = qf as i32;
    let s = qf.mul_add(-LN2_UPPER, value);
    let s = qf.mul_add(-LN2_LOWER, s);
    let u = 0.000_198_527_617_612_853_65f32
        .mul_add(s, 0.001_393_043_552_525_341_5)
        .mul_add(s, 0.008_333_360_776_305_199)
        .mul_add(s, 0.041_666_485_369_205_475)
        .mul_add(s, 0.166_666_671_633_720_4)
        .mul_add(s, 0.5);
    let u = s * s * u + s + 1.0;
    let q0 = q >> 1;
    let q1 = q - q0;
    let scale0 = f32::from_bits(((q0 + 0x7f) as u32) << 23);
    let scale1 = f32::from_bits(((q1 + 0x7f) as u32) << 23);
    u * scale0 * scale1
}

#[cfg(test)]
struct TorchTanh;

#[cfg(test)]
impl CustomOp1 for TorchTanh {
    fn name(&self) -> &'static str {
        "torch_tanh_fixture"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(values) = storage else {
            candle_core::bail!("fixture tanh expects f32")
        };
        if !layout.is_contiguous() {
            candle_core::bail!("fixture tanh expects contiguous input")
        }
        let count = layout.dims().iter().product::<usize>();
        let start = layout.start_offset();
        let output = values[start..start + count]
            .iter()
            .map(|value| value.tanh())
            .collect();
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }

    fn bwd(
        &self,
        input: &Tensor,
        output: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let output = output.flatten_all()?.to_vec1::<f32>()?;
        let grad = grad.flatten_all()?.to_vec1::<f32>()?;
        let derivative = output
            .into_iter()
            .map(|value| 1.0f32 - value * value)
            .zip(grad)
            .map(|(derivative, gradient)| derivative * gradient)
            .collect::<Vec<_>>();
        Ok(Some(Tensor::from_vec(
            derivative,
            input.shape().clone(),
            input.device(),
        )?))
    }
}

pub(crate) fn training_cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    if logits.device().is_cpu() && logits.dtype() == DType::F32 {
        let targets = targets.to_vec1::<u32>()?;
        Ok(logits.apply_op1(CpuCrossEntropy { targets })?)
    } else {
        #[cfg(feature = "cuda-base")]
        if logits.device().is_cuda() && logits.dtype() == DType::F32 {
            return Ok(logits
                .contiguous()?
                .apply_op2(&targets.contiguous()?, CudaCrossEntropy { softcap: false })?);
        }
        Ok(candle_nn::loss::cross_entropy(logits, targets)?)
    }
}

pub(crate) fn training_softcap_cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda-base")]
    if logits.device().is_cuda() && logits.dtype() == DType::F32 {
        return Ok(logits
            .contiguous()?
            .apply_op2(&targets.contiguous()?, CudaCrossEntropy { softcap: true })?);
    }
    let scaled = if logits.device().is_cpu() && logits.dtype() == DType::F32 {
        logits.apply_op1(CpuDivideScalar(LOGIT_SOFTCAP as f32))?
    } else {
        (logits / LOGIT_SOFTCAP)?
    };
    let softcapped = scaled.tanh()?.affine(LOGIT_SOFTCAP, 0.0)?;
    training_cross_entropy(&softcapped, targets)
}

#[cfg(feature = "cuda-base")]
struct CudaReluSquared;

#[cfg(feature = "cuda-base")]
impl CustomOp1 for CudaReluSquared {
    fn name(&self) -> &'static str {
        "cuda_fused_relu_squared"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused ReLU-squared requires CUDA")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda::CudaStorageSlice;

        ensure_cuda_contiguous_layout(layout, "fused ReLU-squared")?;
        let device = &input.device;
        let n = layout.shape().elem_count() as u64;
        let ptx = cuda_relu_squared_ptx()?;
        let output = match &input.slice {
            CudaStorageSlice::F16(source) => candle_core::CudaStorage::wrap_cuda_slice(
                launch_cuda_unary(source, device, "relu_squared_f16", ptx, n)?,
                device.clone(),
            ),
            CudaStorageSlice::BF16(source) => candle_core::CudaStorage::wrap_cuda_slice(
                launch_cuda_unary(source, device, "relu_squared_bf16", ptx, n)?,
                device.clone(),
            ),
            CudaStorageSlice::F32(source) => candle_core::CudaStorage::wrap_cuda_slice(
                launch_cuda_unary(source, device, "relu_squared_f32", ptx, n)?,
                device.clone(),
            ),
            _ => candle_core::bail!("fused ReLU-squared expects F16, BF16, or F32"),
        };
        Ok((output, layout.shape().clone()))
    }

    fn bwd(
        &self,
        input: &Tensor,
        _: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        Ok(Some(input.apply_op2(grad, CudaReluSquaredGrad)?))
    }
}

#[cfg(feature = "cuda-base")]
struct CudaReluSquaredGrad;

#[cfg(feature = "cuda-base")]
impl CustomOp2 for CudaReluSquaredGrad {
    fn name(&self) -> &'static str {
        "cuda_fused_relu_squared_backward"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused ReLU-squared backward requires CUDA")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        input_layout: &Layout,
        grad: &candle_core::CudaStorage,
        grad_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda::CudaStorageSlice;

        ensure_cuda_contiguous_layout(input_layout, "fused ReLU-squared backward")?;
        ensure_cuda_contiguous_layout(grad_layout, "fused ReLU-squared gradient")?;
        if input_layout.shape() != grad_layout.shape() || input.dtype() != grad.dtype() {
            candle_core::bail!("fused ReLU-squared backward input/gradient mismatch")
        }
        let device = &input.device;
        let n = input_layout.shape().elem_count() as u64;
        let ptx = cuda_relu_squared_ptx()?;
        let output = match (&input.slice, &grad.slice) {
            (CudaStorageSlice::F16(source), CudaStorageSlice::F16(upstream)) => {
                candle_core::CudaStorage::wrap_cuda_slice(
                    launch_cuda_binary(
                        source,
                        upstream,
                        device,
                        "relu_squared_backward_f16",
                        ptx,
                        n,
                    )?,
                    device.clone(),
                )
            }
            (CudaStorageSlice::BF16(source), CudaStorageSlice::BF16(upstream)) => {
                candle_core::CudaStorage::wrap_cuda_slice(
                    launch_cuda_binary(
                        source,
                        upstream,
                        device,
                        "relu_squared_backward_bf16",
                        ptx,
                        n,
                    )?,
                    device.clone(),
                )
            }
            (CudaStorageSlice::F32(source), CudaStorageSlice::F32(upstream)) => {
                candle_core::CudaStorage::wrap_cuda_slice(
                    launch_cuda_binary(
                        source,
                        upstream,
                        device,
                        "relu_squared_backward_f32",
                        ptx,
                        n,
                    )?,
                    device.clone(),
                )
            }
            _ => candle_core::bail!("fused ReLU-squared backward expects matching float dtypes"),
        };
        Ok((output, input_layout.shape().clone()))
    }
}

#[cfg(feature = "cuda-base")]
fn ensure_cuda_contiguous_layout(layout: &Layout, operation: &str) -> candle_core::Result<()> {
    if !layout.is_contiguous() || layout.start_offset() != 0 {
        candle_core::bail!("{operation} expects contiguous tensors")
    }
    Ok(())
}

#[cfg(feature = "cuda-base")]
fn launch_cuda_unary<T: candle_core::cuda::cudarc::driver::DeviceRepr>(
    source: &candle_core::cuda::cudarc::driver::CudaSlice<T>,
    device: &candle_core::CudaDevice,
    function_name: &'static str,
    ptx: &str,
    n: u64,
) -> candle_core::Result<candle_core::cuda::cudarc::driver::CudaSlice<T>> {
    use candle_core::cuda::cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};

    let stream = device.cuda_stream();
    let (source, source_record) = source.device_ptr(&stream);
    let mut output = unsafe { device.alloc::<T>(n as usize)? };
    let function =
        device.get_or_load_custom_func(function_name, "rust_agent_fused_relu_squared", ptx)?;
    let config = LaunchConfig::for_num_elems(n as u32);
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&mut output);
    launch.arg(&n);
    unsafe { launch.launch(config) }.map_err(|error| {
        candle_core::Error::Msg(format!("launch fused ReLU-squared kernel: {error}"))
    })?;
    drop(source_record);
    Ok(output)
}

#[cfg(feature = "cuda-base")]
fn launch_cuda_binary<T: candle_core::cuda::cudarc::driver::DeviceRepr>(
    input: &candle_core::cuda::cudarc::driver::CudaSlice<T>,
    grad: &candle_core::cuda::cudarc::driver::CudaSlice<T>,
    device: &candle_core::CudaDevice,
    function_name: &'static str,
    ptx: &str,
    n: u64,
) -> candle_core::Result<candle_core::cuda::cudarc::driver::CudaSlice<T>> {
    use candle_core::cuda::cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};

    let stream = device.cuda_stream();
    let (input, input_record) = input.device_ptr(&stream);
    let (grad, grad_record) = grad.device_ptr(&stream);
    let mut output = unsafe { device.alloc::<T>(n as usize)? };
    let function =
        device.get_or_load_custom_func(function_name, "rust_agent_fused_relu_squared", ptx)?;
    let config = LaunchConfig::for_num_elems(n as u32);
    let mut launch = function.builder();
    launch.arg(&input);
    launch.arg(&grad);
    launch.arg(&mut output);
    launch.arg(&n);
    unsafe { launch.launch(config) }.map_err(|error| {
        candle_core::Error::Msg(format!(
            "launch fused ReLU-squared backward kernel: {error}"
        ))
    })?;
    drop(input_record);
    drop(grad_record);
    Ok(output)
}

#[cfg(feature = "cuda-base")]
fn cuda_relu_squared_ptx() -> candle_core::Result<&'static str> {
    use std::sync::OnceLock;

    const KERNEL: &str = r#"
    __device__ float half_to_float(unsigned short h) {
      unsigned int sign = ((unsigned int)h & 0x8000u) << 16;
      unsigned int exp = ((unsigned int)h >> 10) & 0x1fu;
      unsigned int mant = (unsigned int)h & 0x3ffu;
      unsigned int bits;
      if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
          int unbiased = -14;
          while ((mant & 0x400u) == 0) { mant <<= 1; unbiased -= 1; }
          mant &= 0x3ffu;
          bits = sign | ((unsigned int)(unbiased + 127) << 23) | (mant << 13);
        }
      } else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
      else bits = sign | ((exp + 112u) << 23) | (mant << 13);
      return __uint_as_float(bits);
    }
    __device__ unsigned short float_to_half_rn(float value) {
      unsigned int bits = __float_as_uint(value);
      unsigned int sign = (bits >> 16) & 0x8000u;
      unsigned int exp = (bits >> 23) & 0xffu;
      unsigned int mant = bits & 0x7fffffu;
      if (exp == 0xffu) return (unsigned short)(sign | 0x7c00u | (mant ? 0x0200u : 0));
      int half_exp = (int)exp - 127 + 15;
      if (half_exp >= 31) return (unsigned short)(sign | 0x7c00u);
      if (half_exp <= 0) {
        if (half_exp < -10) return (unsigned short)sign;
        mant |= 0x800000u;
        unsigned int shift = (unsigned int)(14 - half_exp);
        unsigned int result = mant >> shift;
        unsigned int remainder = mant & ((1u << shift) - 1u);
        unsigned int halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (result & 1u))) result += 1u;
        return (unsigned short)(sign | result);
      }
      unsigned int result_mant = mant >> 13;
      unsigned int remainder = mant & 0x1fffu;
      if (remainder > 0x1000u || (remainder == 0x1000u && (result_mant & 1u))) {
        result_mant += 1u;
        if (result_mant == 0x400u) { result_mant = 0; half_exp += 1; }
      }
      if (half_exp >= 31) return (unsigned short)(sign | 0x7c00u);
      return (unsigned short)(sign | ((unsigned int)half_exp << 10) | result_mant);
    }
    __device__ float bf16_to_float(unsigned short value) {
      return __uint_as_float((unsigned int)value << 16);
    }
    __device__ unsigned short float_to_bf16_rn(float value) {
      unsigned int bits = __float_as_uint(value);
      bits += 0x7fffu + ((bits >> 16) & 1u);
      return (unsigned short)(bits >> 16);
    }
    extern "C" __global__ void relu_squared_f32(const float* x, float* y, unsigned long long n) {
      unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) { float v = x[i]; y[i] = v > 0.0f ? v * v : 0.0f; }
    }
    extern "C" __global__ void relu_squared_f16(const unsigned short* x, unsigned short* y, unsigned long long n) {
      unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) { float v = half_to_float(x[i]); y[i] = float_to_half_rn(v > 0.0f ? v * v : 0.0f); }
    }
    extern "C" __global__ void relu_squared_bf16(const unsigned short* x, unsigned short* y, unsigned long long n) {
      unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) { float v = bf16_to_float(x[i]); y[i] = float_to_bf16_rn(v > 0.0f ? v * v : 0.0f); }
    }
    extern "C" __global__ void relu_squared_backward_f32(const float* x, const float* g, float* y, unsigned long long n) {
      unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) { float v = x[i]; y[i] = v > 0.0f ? (2.0f * v) * g[i] : 0.0f; }
    }
    extern "C" __global__ void relu_squared_backward_f16(const unsigned short* x, const unsigned short* g, unsigned short* y, unsigned long long n) {
      unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) { float v = half_to_float(x[i]); float d = half_to_float(g[i]); y[i] = float_to_half_rn(v > 0.0f ? (2.0f * v) * d : 0.0f); }
    }
    extern "C" __global__ void relu_squared_backward_bf16(const unsigned short* x, const unsigned short* g, unsigned short* y, unsigned long long n) {
      unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) { float v = bf16_to_float(x[i]); float d = bf16_to_float(g[i]); y[i] = float_to_bf16_rn(v > 0.0f ? (2.0f * v) * d : 0.0f); }
    }"#;
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    let ptx = PTX.get_or_init(|| {
        candle_core::cuda::cudarc::nvrtc::compile_ptx(KERNEL)
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
    });
    ptx.as_ref().map(String::as_str).map_err(|error| {
        candle_core::Error::Msg(format!("compile fused ReLU-squared kernels: {error}"))
    })
}

#[cfg(feature = "cuda-base")]
struct CudaCrossEntropy {
    softcap: bool,
}

#[cfg(feature = "cuda-base")]
impl candle_core::CustomOp2 for CudaCrossEntropy {
    fn name(&self) -> &'static str {
        "cuda_fused_cross_entropy"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA fused cross entropy requires CUDA")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        input_layout: &Layout,
        target: &candle_core::CudaStorage,
        target_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda::{
            CudaStorageSlice,
            cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
        };

        let [rows, classes] = input_layout.dims() else {
            candle_core::bail!("CUDA fused cross entropy expects rank-2 logits")
        };
        if target_layout.dims() != [*rows]
            || !input_layout.is_contiguous()
            || !target_layout.is_contiguous()
            || input_layout.start_offset() != 0
            || target_layout.start_offset() != 0
        {
            candle_core::bail!("CUDA fused cross entropy expects contiguous logits and targets")
        }
        let CudaStorageSlice::F32(input_slice) = &input.slice else {
            candle_core::bail!("CUDA fused cross entropy expects F32 logits")
        };
        let CudaStorageSlice::U32(target_slice) = &target.slice else {
            candle_core::bail!("CUDA fused cross entropy expects U32 targets")
        };
        let device = &input.device;
        let stream = device.cuda_stream();
        let (input_ptr, _input_record) = input_slice.device_ptr(&stream);
        let (target_ptr, _target_record) = target_slice.device_ptr(&stream);
        let mut output = device.alloc_zeros::<f32>(1)?;
        let (output_ptr, output_record) = output.device_ptr_mut(&stream);
        let softcap = u32::from(self.softcap);
        let ptx = cuda_cross_entropy_ptx(FORWARD_KERNEL, true)?;
        let function = device.get_or_load_custom_func(
            "fused_cross_entropy_forward",
            "rust_agent_fused_cross_entropy_forward",
            ptx,
        )?;
        let rows = *rows as u32;
        let classes = *classes as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = function.builder();
        launch.arg(&input_ptr);
        launch.arg(&target_ptr);
        launch.arg(&output_ptr);
        launch.arg(&rows);
        launch.arg(&classes);
        launch.arg(&softcap);
        unsafe { launch.launch(cfg) }.map_err(|error| {
            candle_core::Error::Msg(format!("launch fused cross-entropy forward: {error}"))
        })?;
        drop(output_record);
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from(()),
        ))
    }

    fn bwd(
        &self,
        input: &Tensor,
        target: &Tensor,
        _: &Tensor,
        output_grad: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        use candle_core::{
            Storage,
            cuda::{
                CudaStorageSlice,
                cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
            },
        };

        let (rows, classes) = input.dims2()?;
        let (input_storage, input_layout) = input.storage_and_layout();
        let (target_storage, target_layout) = target.storage_and_layout();
        let (grad_storage, grad_layout) = output_grad.storage_and_layout();
        let (
            Storage::Cuda(input_storage),
            Storage::Cuda(target_storage),
            Storage::Cuda(grad_storage),
        ) = (&*input_storage, &*target_storage, &*grad_storage)
        else {
            candle_core::bail!("CUDA fused cross entropy backward expects CUDA tensors")
        };
        let (
            CudaStorageSlice::F32(input_slice),
            CudaStorageSlice::U32(target_slice),
            CudaStorageSlice::F32(grad_slice),
        ) = (
            &input_storage.slice,
            &target_storage.slice,
            &grad_storage.slice,
        )
        else {
            candle_core::bail!("CUDA fused cross entropy backward dtype mismatch")
        };
        if !input_layout.is_contiguous()
            || !target_layout.is_contiguous()
            || !grad_layout.is_contiguous()
            || input_layout.start_offset() != 0
            || target_layout.start_offset() != 0
            || grad_layout.start_offset() != 0
        {
            candle_core::bail!("CUDA fused cross entropy backward expects contiguous tensors")
        }
        let device = &input_storage.device;
        let stream = device.cuda_stream();
        let (input_ptr, _input_record) = input_slice.device_ptr(&stream);
        let (target_ptr, _target_record) = target_slice.device_ptr(&stream);
        let (grad_ptr, _grad_record) = grad_slice.device_ptr(&stream);
        // SAFETY: the backward kernel writes every element of this output.
        let mut gradient = unsafe { device.alloc::<f32>(rows * classes)? };
        let (gradient_ptr, gradient_record) = gradient.device_ptr_mut(&stream);
        let softcap = u32::from(self.softcap);
        let ptx = cuda_cross_entropy_ptx(BACKWARD_KERNEL, false)?;
        let function = device.get_or_load_custom_func(
            "fused_cross_entropy_backward",
            "rust_agent_fused_cross_entropy_backward",
            ptx,
        )?;
        let rows = rows as u32;
        let classes = classes as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = function.builder();
        launch.arg(&input_ptr);
        launch.arg(&target_ptr);
        launch.arg(&grad_ptr);
        launch.arg(&gradient_ptr);
        launch.arg(&rows);
        launch.arg(&classes);
        launch.arg(&softcap);
        unsafe { launch.launch(cfg) }.map_err(|error| {
            candle_core::Error::Msg(format!("launch fused cross-entropy backward: {error}"))
        })?;
        drop(gradient_record);
        let gradient = Tensor::from_storage(
            Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                gradient,
                device.clone(),
            )),
            input.shape().clone(),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok((Some(gradient), None))
    }
}

#[cfg(feature = "cuda-base")]
fn cuda_cross_entropy_ptx(kernel: &str, forward: bool) -> candle_core::Result<&'static str> {
    use std::sync::OnceLock;

    static FORWARD_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    static BACKWARD_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    let slot = if forward { &FORWARD_PTX } else { &BACKWARD_PTX };
    let ptx = slot.get_or_init(|| {
        candle_core::cuda::cudarc::nvrtc::compile_ptx(kernel)
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
    });
    let ptx = ptx.as_ref().map_err(|error| {
        candle_core::Error::Msg(format!("compile fused cross-entropy kernel: {error}"))
    })?;
    Ok(ptx.as_str())
}

#[cfg(feature = "cuda-base")]
const FORWARD_KERNEL: &str = r#"
extern "C" __global__ void fused_cross_entropy_forward(
    const float* logits, const unsigned int* targets, float* loss,
    unsigned int rows, unsigned int classes, unsigned int softcap) {
  unsigned int row = blockIdx.x;
  unsigned int tid = threadIdx.x;
  if (row >= rows) return;
  __shared__ float shared[256];
  float maximum = -3.402823466e+38F;
  const unsigned long long offset = (unsigned long long)row * classes;
  for (unsigned int col = tid; col < classes; col += blockDim.x) {
    float value = logits[offset + col];
    if (softcap) value = 15.0f * tanhf(value / 15.0f);
    maximum = fmaxf(maximum, value);
  }
  shared[tid] = maximum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    __syncthreads();
  }
  maximum = shared[0];
  float denominator = 0.0f;
  for (unsigned int col = tid; col < classes; col += blockDim.x) {
    float value = logits[offset + col];
    if (softcap) value = 15.0f * tanhf(value / 15.0f);
    denominator += expf(value - maximum);
  }
  shared[tid] = denominator;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  if (tid == 0) {
    float target = logits[offset + targets[row]];
    if (softcap) target = 15.0f * tanhf(target / 15.0f);
    atomicAdd(loss, (maximum + logf(shared[0]) - target) / rows);
  }
}"#;

#[cfg(feature = "cuda-base")]
const BACKWARD_KERNEL: &str = r#"
extern "C" __global__ void fused_cross_entropy_backward(
    const float* logits, const unsigned int* targets, const float* loss_grad,
    float* gradient, unsigned int rows, unsigned int classes, unsigned int softcap) {
  unsigned int row = blockIdx.x;
  unsigned int tid = threadIdx.x;
  if (row >= rows) return;
  __shared__ float shared[256];
  float maximum = -3.402823466e+38F;
  const unsigned long long offset = (unsigned long long)row * classes;
  for (unsigned int col = tid; col < classes; col += blockDim.x) {
    float value = logits[offset + col];
    if (softcap) value = 15.0f * tanhf(value / 15.0f);
    maximum = fmaxf(maximum, value);
  }
  shared[tid] = maximum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    __syncthreads();
  }
  maximum = shared[0];
  float denominator = 0.0f;
  for (unsigned int col = tid; col < classes; col += blockDim.x) {
    float value = logits[offset + col];
    if (softcap) value = 15.0f * tanhf(value / 15.0f);
    denominator += expf(value - maximum);
  }
  shared[tid] = denominator;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  denominator = shared[0];
  const float scale = loss_grad[0] / rows;
  for (unsigned int col = tid; col < classes; col += blockDim.x) {
    float raw = logits[offset + col];
    float value = softcap ? 15.0f * tanhf(raw / 15.0f) : raw;
    float probability = expf(value - maximum) / denominator;
    float derivative = softcap ? 1.0f - tanhf(raw / 15.0f) * tanhf(raw / 15.0f) : 1.0f;
    gradient[offset + col] = (probability - (col == targets[row] ? 1.0f : 0.0f)) * derivative * scale;
  }
}"#;

fn pytorch_cpu_f32_sum(values: &[f32]) -> f32 {
    const VECTOR_WIDTH: usize = 8;
    const ILP: usize = 4;
    const LEVELS: usize = 4;

    let vector_count = values.len() / VECTOR_WIDTH;
    let ilp_count = vector_count / ILP;
    let ceil_log2 = if ilp_count <= 1 {
        0
    } else {
        usize::BITS as usize - (ilp_count - 1).leading_zeros() as usize
    };
    let level_power = 4.max(ceil_log2 / LEVELS);
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;
    let mut accumulators = [[[0f32; VECTOR_WIDTH]; ILP]; LEVELS];
    let mut i = 0;

    while i + level_step <= ilp_count {
        for _ in 0..level_step {
            for k in 0..ILP {
                let start = (i * ILP + k) * VECTOR_WIDTH;
                for lane in 0..VECTOR_WIDTH {
                    accumulators[0][k][lane] += values[start + lane];
                }
            }
            i += 1;
        }
        for level in 1..LEVELS {
            for k in 0..ILP {
                for lane in 0..VECTOR_WIDTH {
                    accumulators[level][k][lane] += accumulators[level - 1][k][lane];
                    accumulators[level - 1][k][lane] = 0.0;
                }
            }
            let mask = level_mask << (level * level_power);
            if i & mask != 0 {
                break;
            }
        }
    }

    for i in i..ilp_count {
        for k in 0..ILP {
            let start = (i * ILP + k) * VECTOR_WIDTH;
            for lane in 0..VECTOR_WIDTH {
                accumulators[0][k][lane] += values[start + lane];
            }
        }
    }
    for level in 1..LEVELS {
        for k in 0..ILP {
            for lane in 0..VECTOR_WIDTH {
                accumulators[0][k][lane] += accumulators[level][k][lane];
            }
        }
    }

    let mut partial = accumulators[0][0];
    let mut vector_index = ilp_count * ILP;
    while vector_index < vector_count {
        let start = vector_index * VECTOR_WIDTH;
        for lane in 0..VECTOR_WIDTH {
            partial[lane] += values[start + lane];
        }
        vector_index += 1;
    }
    for k in 1..ILP {
        for lane in 0..VECTOR_WIDTH {
            partial[lane] += accumulators[0][k][lane];
        }
    }

    let mut sum = 0f32;
    for value in &values[vector_count * VECTOR_WIDTH..] {
        sum += value;
    }
    for value in partial {
        sum += value;
    }
    sum
}

struct CpuMeanLast;

struct CpuRmsNorm {
    width: usize,
}

impl CustomOp1 for CpuRmsNorm {
    fn name(&self) -> &'static str {
        "cpu_rms_norm"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(values) = storage else {
            candle_core::bail!("cpu_rms_norm expects f32")
        };
        if !layout.is_contiguous() {
            candle_core::bail!("cpu_rms_norm expects contiguous input")
        }
        let start = layout.start_offset();
        let count = layout.dims().iter().product::<usize>();
        let input = &values[start..start + count];
        let mut output = Vec::with_capacity(count);
        for row in input.chunks_exact(self.width) {
            let squares = row.iter().map(|value| value * value).collect::<Vec<_>>();
            let mean_square = pytorch_cpu_f32_sum(&squares) / self.width as f32;
            let rstd = 1.0 / (mean_square + RMS_EPS_F32 as f32).sqrt();
            output.extend(row.iter().map(|value| value * rstd));
        }
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }

    fn bwd(
        &self,
        input: &Tensor,
        _output: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let shape = input.shape().clone();
        let device = input.device().clone();
        let input = input.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let grad = grad.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let mut gradient = Vec::with_capacity(input.len());
        for (x, dy) in input
            .chunks_exact(self.width)
            .zip(grad.chunks_exact(self.width))
        {
            let products = x.iter().zip(dy).map(|(x, dy)| x * dy).collect::<Vec<_>>();
            let dot = pytorch_cpu_f32_sum(&products);
            let squares = x.iter().map(|value| value * value).collect::<Vec<_>>();
            let mean_square = pytorch_cpu_f32_sum(&squares) / self.width as f32;
            let rstd = 1.0 / (mean_square + RMS_EPS_F32 as f32).sqrt();
            let scale = rstd * rstd * rstd / self.width as f32;
            gradient.extend(
                x.iter()
                    .zip(dy)
                    .map(|(x, dy)| dy * rstd - x * (dot * scale)),
            );
        }
        Ok(Some(Tensor::from_vec(gradient, shape, &device)?))
    }
}

struct CpuDivScalar(f32);

impl CustomOp1 for CpuDivScalar {
    fn name(&self) -> &'static str {
        "cpu_div_scalar"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(values) = storage else {
            candle_core::bail!("cpu_div_scalar expects f32")
        };
        let dims = layout.dims();
        let strides = layout.stride();
        let count = dims.iter().product::<usize>();
        let mut output = Vec::with_capacity(count);
        for flat_index in 0..count {
            let mut remainder = flat_index;
            let mut offset = layout.start_offset();
            for axis in (0..dims.len()).rev() {
                let coordinate = remainder % dims[axis];
                remainder /= dims[axis];
                offset += coordinate * strides[axis];
            }
            output.push(values[offset] / self.0);
        }
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }

    fn bwd(
        &self,
        _input: &Tensor,
        _output: &Tensor,
        _grad: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        Ok(None)
    }
}

struct CpuDivideScalar(f32);

impl CustomOp1 for CpuDivideScalar {
    fn name(&self) -> &'static str {
        "cpu_divide_scalar"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(values) = storage else {
            candle_core::bail!("cpu_divide_scalar expects f32")
        };
        let dims = layout.dims();
        let strides = layout.stride();
        let count = dims.iter().product::<usize>();
        let mut output = Vec::with_capacity(count);
        for flat_index in 0..count {
            let mut remainder = flat_index;
            let mut offset = layout.start_offset();
            for axis in (0..dims.len()).rev() {
                let coordinate = remainder % dims[axis];
                remainder /= dims[axis];
                offset += coordinate * strides[axis];
            }
            output.push(values[offset] / self.0);
        }
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }

    fn bwd(
        &self,
        _input: &Tensor,
        _output: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        Ok(Some(grad.apply_op1(CpuDivScalar(self.0))?))
    }
}

/// PyTorch differentiates reciprocal square root using its output cubed.
struct CpuRsqrt;

impl CustomOp1 for CpuRsqrt {
    fn name(&self) -> &'static str {
        "cpu_rsqrt"
    }
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(values) = storage else {
            candle_core::bail!("cpu_rsqrt expects f32")
        };
        let count = layout.dims().iter().product::<usize>();
        let start = layout.start_offset();
        let result = values[start..start + count]
            .iter()
            .map(|x| 1.0f32 / x.sqrt())
            .collect();
        Ok((CpuStorage::F32(result), layout.shape().clone()))
    }
    fn bwd(
        &self,
        _arg: &Tensor,
        result: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        Ok(Some(grad.affine(-0.5, 0.0)?.mul(&result.powf(3.0)?)?))
    }
}

impl CustomOp1 for CpuMeanLast {
    fn name(&self) -> &'static str {
        "cpu_mean_last"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(values) = storage else {
            candle_core::bail!("cpu_mean_last expects f32")
        };
        let width = *layout.dims().last().unwrap();
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().unwrap() = 1;
        let count = layout.dims().iter().product::<usize>();
        let start = layout.start_offset();
        let means = values[start..start + count]
            .chunks_exact(width)
            .map(|row| pytorch_cpu_f32_sum(row) / width as f32)
            .collect();
        Ok((CpuStorage::F32(means), Shape::from(dims)))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _res: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let width = *arg.dims().last().unwrap();
        let expanded = grad.broadcast_as(arg.shape())?.contiguous()?;
        Ok(Some(expanded.apply_op1(CpuDivScalar(width as f32))?))
    }
}

#[derive(Clone)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    proj: Linear,
    ve_gate: Option<Linear>,
}

#[derive(Clone)]
struct Mlp {
    fc: Linear,
    proj: Linear,
}

#[derive(Clone)]
struct Block {
    attn: Attention,
    mlp: Mlp,
}

#[derive(Clone)]
pub struct GptModel {
    pub config: GptConfig,
    dtype: DType,
    fp8_training: bool,
    inference_attention: bool,
    token_embedding: Embedding,
    value_embeddings: Vec<Option<Embedding>>,
    blocks: Vec<Block>,
    lm_head: Linear,
    resid_lambdas: Tensor,
    x0_lambdas: Tensor,
    smear_gate: Linear,
    smear_lambda: Tensor,
    backout_lambda: Tensor,
}

/// Per-sequence keys, values and the previous token embedding used by smear.
pub struct KvCache {
    max_seq_len: usize,
    position: usize,
    batch_size: usize,
    keys: Vec<Option<Tensor>>,
    values: Vec<Option<Tensor>>,
    previous_embedding: Option<Tensor>,
}

impl KvCache {
    pub fn new(config: &GptConfig) -> Self {
        Self {
            max_seq_len: config.sequence_len,
            position: 0,
            batch_size: 1,
            keys: (0..config.n_layer).map(|_| None).collect(),
            values: (0..config.n_layer).map(|_| None).collect(),
            previous_embedding: None,
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn replicate_batch(&mut self, batch_size: usize) -> Result<()> {
        ensure!(self.position > 0, "cannot replicate an empty KV cache");
        ensure!(
            self.batch_size == 1,
            "KV cache can only be replicated from one row"
        );
        ensure!(batch_size > 0, "KV cache batch size must be positive");
        if batch_size == 1 {
            return Ok(());
        }
        for key in self.keys.iter_mut().flatten() {
            let (_, heads, time, width) = key.dims4()?;
            *key = key
                .broadcast_as((batch_size, heads, time, width))?
                .contiguous()?;
        }
        for value in self.values.iter_mut().flatten() {
            let (_, heads, time, width) = value.dims4()?;
            *value = value
                .broadcast_as((batch_size, heads, time, width))?
                .contiguous()?;
        }
        if let Some(previous) = self.previous_embedding.as_mut() {
            let (_, time, width) = previous.dims3()?;
            *previous = previous
                .broadcast_as((batch_size, time, width))?
                .contiguous()?;
        }
        self.batch_size = batch_size;
        Ok(())
    }
}

impl GptModel {
    /// Use a lower precision for forward activations while retaining the
    /// model's original parameter storage dtype (normally FP32 for training).
    pub fn with_compute_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn with_fp8_training(mut self, enabled: bool) -> Self {
        self.fp8_training = enabled;
        self
    }

    /// Enable inference-only accelerator attention kernels for uncached
    /// forwards. Training keeps the differentiable fallback.
    pub fn with_inference_attention(mut self, enabled: bool) -> Self {
        self.inference_attention = enabled;
        self
    }

    pub fn device(&self) -> &Device {
        self.token_embedding.embeddings().device()
    }

    pub fn parameter_count(&self) -> usize {
        let config = &self.config;
        let padded_vocab = config.vocab_size.div_ceil(64) * 64;
        let width = config.n_embd;
        let kv_width = config.n_kv_head * (width / config.n_head);
        // matmul_parameter_count includes the output head for FLOP estimates,
        // while the two explicit vocab matrices below already count both the
        // token embedding and that head.
        let output_head_parameters = padded_vocab * width;
        let value_embedding_layers = (0..config.n_layer)
            .filter(|layer| has_value_embedding(*layer, config.n_layer))
            .count();
        padded_vocab * width * 2
            + value_embedding_layers * padded_vocab * kv_width
            + self
                .matmul_parameter_count()
                .saturating_sub(output_head_parameters)
            + 2 * config.n_layer
            + 2
    }

    pub fn weight_bytes(&self) -> usize {
        self.parameter_count() * dtype_bytes(self.dtype)
    }

    pub fn matmul_parameter_count(&self) -> usize {
        Self::matmul_parameter_count_for(&self.config)
    }

    pub fn scaling_parameter_count(config: &GptConfig) -> usize {
        // Python's scaling fit uses transformer block matrices plus the output
        // head, excluding the 24-weight smear gate included in matmul FLOPs.
        let model = Self::matmul_parameter_count_for(config);
        model.saturating_sub(24)
    }

    pub fn training_flops_per_token(config: &GptConfig) -> u64 {
        let head_dim = config.n_embd / config.n_head;
        let attention = (0..config.n_layer)
            .map(|layer| {
                let (left, _) = config.attention_window(layer);
                let window = if left < 0 {
                    config.sequence_len
                } else {
                    left as usize
                };
                12 * config.n_head * head_dim * window.min(config.sequence_len)
            })
            .sum::<usize>();
        (6 * Self::matmul_parameter_count_for(config) + attention) as u64
    }

    fn matmul_parameter_count_for(config: &GptConfig) -> usize {
        let width = config.n_embd;
        let kv_width = config.n_kv_head * (width / config.n_head);
        let padded_vocab = config.vocab_size.div_ceil(64) * 64;
        let per_layer = 2 * width * width + 2 * width * kv_width + 8 * width * width;
        let gates = (0..config.n_layer)
            .filter(|layer| has_value_embedding(*layer, config.n_layer))
            .count()
            * 12
            * config.n_kv_head;
        config.n_layer * per_layer + gates + width * padded_vocab + 24
    }

    pub fn kv_bytes_per_token(&self) -> usize {
        self.config.n_layer
            * 2
            * self.config.n_kv_head
            * (self.config.n_embd / self.config.n_head)
            * dtype_bytes(self.dtype)
    }

    pub fn kv_read_bytes(&self, context_len: usize) -> usize {
        let config = &self.config;
        let head_dim = config.n_embd / config.n_head;
        let short_window = config.sequence_len.div_ceil(512) * 128;
        (0..config.n_layer)
            .map(|layer| {
                let window = if config.is_long_window_layer(layer) {
                    config.sequence_len
                } else {
                    short_window
                };
                2 * config.n_kv_head * head_dim * dtype_bytes(self.dtype) * context_len.min(window)
            })
            .sum()
    }

    pub fn estimate_decode_flops(&self, context_len: usize) -> u64 {
        let config = &self.config;
        let head_dim = config.n_embd / config.n_head;
        let attention = (0..config.n_layer)
            .map(|layer| {
                let window = if config.is_long_window_layer(layer) {
                    config.sequence_len
                } else {
                    config.sequence_len.div_ceil(512) * 128
                };
                4 * config.n_head * head_dim * context_len.min(window)
            })
            .sum::<usize>();
        (2 * self.matmul_parameter_count() + attention) as u64
    }

    pub fn estimate_prefill_flops(&self, num_tokens: usize) -> u64 {
        let config = &self.config;
        let head_dim = config.n_embd / config.n_head;
        let attention = (0..config.n_layer)
            .map(|layer| {
                let window = if config.is_long_window_layer(layer) {
                    config.sequence_len
                } else {
                    config.sequence_len.div_ceil(512) * 128
                }
                .min(num_tokens);
                4 * config.n_head
                    * head_dim
                    * (window * (window + 1) / 2 + (num_tokens - window) * window)
            })
            .sum::<usize>();
        (2 * self.matmul_parameter_count() * num_tokens + attention) as u64
    }

    pub fn load_checkpoint(
        config: GptConfig,
        checkpoint: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let checkpoint = crate::checkpoint::resolve_model_file(checkpoint)?;
        if checkpoint
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("safetensors")
        {
            Self::load_safetensors(config, checkpoint, device, dtype)
        } else {
            Self::load_pth(config, checkpoint, device, dtype)
        }
    }

    /// Load the existing `torch.save(model.state_dict())` checkpoint format.
    /// Candle reads tensor state dictionaries directly, so inference no longer
    /// needs a Python process. `_orig_mod.` is removed for torch.compile saves.
    pub fn load_pth(
        config: GptConfig,
        checkpoint: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (model, _variables) =
            Self::load_trainable_pth(config, checkpoint.as_ref(), device, dtype)?;
        Ok(model)
    }

    pub fn load_safetensors(
        config: GptConfig,
        checkpoint: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (model, _variables) =
            Self::load_trainable_safetensors(config, checkpoint.as_ref(), device, dtype)?;
        Ok(model)
    }

    pub fn load_trainable_pth(
        config: GptConfig,
        checkpoint: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<(Self, VarMap)> {
        config.validate()?;
        let checkpoint = crate::checkpoint::resolve_model_file(checkpoint)?;
        let variables = VarMap::new();
        let target_builder = VarBuilder::from_varmap(&variables, dtype, device);
        let source = candle_core::pickle::PthTensors::new(&checkpoint, None)?;
        let source_names = source
            .tensor_infos()
            .keys()
            .cloned()
            .collect::<HashSet<_>>();
        let embedding_name = checkpoint_source_name("transformer.wte.weight", &source_names);
        let embedding_dtype = source
            .tensor_infos()
            .get(&embedding_name)
            .map(|info| info.dtype)
            .unwrap_or(dtype);
        let model = Self::from_var_builder(config, target_builder, dtype, embedding_dtype)?;
        validate_checkpoint_keys(
            source_names.iter().map(String::as_str),
            &model_variable_names(&variables),
        )?;
        let lookup_names = source_names.clone();
        let source_builder = VarBuilder::from_backend(Box::new(source), dtype, device.clone())
            .rename_f(move |name| checkpoint_source_name(name, &lookup_names));
        for (name, variable) in variables.data().lock().unwrap().iter() {
            if source_builder.contains_tensor(name) {
                variable.set(
                    &source_builder
                        .get_unchecked(name)?
                        .to_dtype(variable.dtype())?,
                )?;
            } else if name != "resid_lambdas" && name != "x0_lambdas" {
                anyhow::bail!("missing model parameter {name} in checkpoint");
            }
        }
        Ok((model, variables))
    }

    pub fn load_trainable_safetensors(
        config: GptConfig,
        checkpoint: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<(Self, VarMap)> {
        config.validate()?;
        let checkpoint = crate::checkpoint::resolve_model_file(checkpoint)?;
        let variables = VarMap::new();
        let target_builder = VarBuilder::from_varmap(&variables, dtype, device);
        let source =
            unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[checkpoint.as_path()])? };
        let source_tensors = source.tensors();
        let source_names = source_tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<HashSet<_>>();
        let embedding_name = checkpoint_source_name("transformer.wte.weight", &source_names);
        let embedding_dtype = source_tensors
            .iter()
            .find(|(name, _)| name == &embedding_name)
            .map(|(_, tensor)| DType::try_from(tensor.dtype()))
            .transpose()?
            .unwrap_or(dtype);
        let model = Self::from_var_builder(config, target_builder, dtype, embedding_dtype)?;
        validate_checkpoint_keys(
            source_names.iter().map(String::as_str),
            &model_variable_names(&variables),
        )?;
        let lookup_names = source_names.clone();
        let source_builder = VarBuilder::from_backend(Box::new(source), dtype, device.clone())
            .rename_f(move |name| checkpoint_source_name(name, &lookup_names));
        for (name, variable) in variables.data().lock().unwrap().iter() {
            if source_builder.contains_tensor(name) {
                variable.set(
                    &source_builder
                        .get_unchecked(name)?
                        .to_dtype(variable.dtype())?,
                )?;
            } else if name != "resid_lambdas" && name != "x0_lambdas" {
                anyhow::bail!("missing model parameter {name} in checkpoint");
            }
        }
        Ok((model, variables))
    }

    /// Construct model variables for gradient-based training. The returned
    /// `VarMap` owns the named parameters and is passed to the optimizer.
    /// Keep linear/scalar master parameters in FP32, matching PyTorch's
    /// reduced-precision token/value embeddings when `compute_dtype` is BF16.
    pub fn new_trainable_with_compute_dtype(
        config: GptConfig,
        device: &Device,
        compute_dtype: DType,
    ) -> Result<(Self, VarMap)> {
        config.validate()?;
        let mut variables = VarMap::new();
        let parameter_dtype = DType::F32;
        let vb = VarBuilder::from_varmap(&variables, parameter_dtype, device);
        let embedding_dtype = if compute_dtype == DType::BF16 {
            DType::BF16
        } else {
            DType::F32
        };
        let model = Self::from_var_builder(config.clone(), vb, compute_dtype, embedding_dtype)?;
        let denom = config.n_layer.saturating_sub(1).max(1) as f32;
        let resid = (0..config.n_layer)
            .map(|layer| 1.15 - 0.10 * layer as f32 / denom)
            .collect::<Vec<_>>();
        let x0 = (0..config.n_layer)
            .map(|layer| 0.20 - 0.15 * layer as f32 / denom)
            .collect::<Vec<_>>();
        variables.set_one(
            "resid_lambdas",
            Tensor::from_vec(resid, (config.n_layer,), device)?.to_dtype(parameter_dtype)?,
        )?;
        variables.set_one(
            "x0_lambdas",
            Tensor::from_vec(x0, (config.n_layer,), device)?.to_dtype(parameter_dtype)?,
        )?;
        initialize_trainable_variables(&variables, &config, device, 42)?;
        Ok((model, variables))
    }

    fn from_var_builder(
        config: GptConfig,
        vb: VarBuilder<'_>,
        dtype: DType,
        embedding_dtype: DType,
    ) -> Result<Self> {
        let padded_vocab = config.vocab_size.div_ceil(64) * 64;
        let n_embd = config.n_embd;
        let head_dim = n_embd / config.n_head;
        let kv_dim = config.n_kv_head * head_dim;
        let matrix_bound = 3.0f64.sqrt() / (n_embd as f64).sqrt();
        let matrix_init = Init::Uniform {
            lo: -matrix_bound,
            up: matrix_bound,
        };
        let zero_init = Init::Const(0.0);
        let small_positive_init = Init::Uniform { lo: 0.0, up: 0.02 };
        let token_embedding = embedding_at(
            padded_vocab,
            n_embd,
            vb.pp("transformer").pp("wte").to_dtype(embedding_dtype),
            Init::Randn {
                mean: 0.0,
                stdev: 0.8,
            },
        )?;
        let mut value_embeddings = Vec::with_capacity(config.n_layer);
        let mut blocks = Vec::with_capacity(config.n_layer);
        for layer in 0..config.n_layer {
            let layer_vb = vb.pp("transformer").pp("h").pp(layer);
            let attention_vb = layer_vb.pp("attn");
            let ve_gate = has_value_embedding(layer, config.n_layer)
                .then(|| {
                    linear(
                        12,
                        config.n_kv_head,
                        attention_vb.pp("ve_gate"),
                        small_positive_init,
                    )
                })
                .transpose()?;
            blocks.push(Block {
                attn: Attention {
                    q: linear(n_embd, n_embd, attention_vb.pp("c_q"), matrix_init)?,
                    k: linear(n_embd, kv_dim, attention_vb.pp("c_k"), matrix_init)?,
                    v: linear(n_embd, kv_dim, attention_vb.pp("c_v"), matrix_init)?,
                    proj: linear(n_embd, n_embd, attention_vb.pp("c_proj"), zero_init)?,
                    ve_gate,
                },
                mlp: Mlp {
                    fc: linear(
                        n_embd,
                        4 * n_embd,
                        layer_vb.pp("mlp").pp("c_fc"),
                        Init::Uniform {
                            lo: -matrix_bound * 0.4,
                            up: matrix_bound * 0.4,
                        },
                    )?,
                    proj: linear(
                        4 * n_embd,
                        n_embd,
                        layer_vb.pp("mlp").pp("c_proj"),
                        zero_init,
                    )?,
                },
            });
            let value_embedding = if has_value_embedding(layer, config.n_layer) {
                Some(embedding_at(
                    padded_vocab,
                    kv_dim,
                    vb.pp("value_embeds").pp(layer).to_dtype(embedding_dtype),
                    matrix_init,
                )?)
            } else {
                None
            };
            value_embeddings.push(value_embedding);
        }
        Ok(Self {
            lm_head: linear(
                n_embd,
                padded_vocab,
                vb.pp("lm_head"),
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.001,
                },
            )?,
            resid_lambdas: vb.get_with_hints(config.n_layer, "resid_lambdas", Init::Const(1.0))?,
            x0_lambdas: vb.get_with_hints(config.n_layer, "x0_lambdas", zero_init)?,
            smear_gate: linear(24, 1, vb.pp("smear_gate"), small_positive_init)?,
            smear_lambda: vb.get_with_hints(1, "smear_lambda", zero_init)?,
            backout_lambda: vb.get_with_hints(1, "backout_lambda", Init::Const(0.2))?,
            config,
            dtype,
            fp8_training: false,
            inference_attention: false,
            token_embedding,
            value_embeddings,
            blocks,
        })
    }

    /// Forward a row-major batch of token IDs and return fp32 logits with the
    /// original (unpadded) vocabulary width. Input shape is `[batch, time]`.
    pub fn forward(&self, token_ids: &[u32], batch: usize, time: usize) -> Result<Tensor> {
        self.forward_with_cache(token_ids, batch, time, None)
    }

    /// Forward the full sequence but project only selected flattened positions
    /// through the vocabulary head. Positions are indices into `[batch,time]`.
    pub fn forward_positions(
        &self,
        token_ids: &[u32],
        batch: usize,
        time: usize,
        positions: &[u32],
    ) -> Result<Tensor> {
        self.forward_internal(token_ids, batch, time, None, Some(positions), true)
    }

    /// Training-only logits before the output softcap. CUDA training applies
    /// the softcap inside its fused cross-entropy kernel.
    pub(crate) fn forward_training_logits(
        &self,
        token_ids: &[u32],
        batch: usize,
        time: usize,
    ) -> Result<Tensor> {
        self.forward_internal(token_ids, batch, time, None, None, false)
    }

    pub(crate) fn forward_positions_training_logits(
        &self,
        token_ids: &[u32],
        batch: usize,
        time: usize,
        positions: &[u32],
    ) -> Result<Tensor> {
        self.forward_internal(token_ids, batch, time, None, Some(positions), false)
    }

    pub fn forward_with_cache(
        &self,
        token_ids: &[u32],
        batch: usize,
        time: usize,
        cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        self.forward_internal(token_ids, batch, time, cache, None, true)
    }

    fn forward_internal(
        &self,
        token_ids: &[u32],
        batch: usize,
        time: usize,
        mut cache: Option<&mut KvCache>,
        positions: Option<&[u32]>,
        softcap_logits: bool,
    ) -> Result<Tensor> {
        ensure!(
            batch > 0 && time > 0,
            "batch and sequence length must be positive"
        );
        let start_position = cache.as_ref().map_or(0, |cache| cache.position);
        // The Python model permits longer uncached evaluation prompts; the
        // configured context limit is enforced for generation's KV cache.
        if cache.is_some() {
            ensure!(
                time + start_position <= self.config.sequence_len,
                "sequence exceeds model context"
            );
            ensure!(
                batch == cache.as_ref().unwrap().batch_size,
                "KV cache batch size mismatch"
            );
            ensure!(
                time + start_position <= cache.as_ref().unwrap().max_seq_len,
                "KV cache capacity exceeded"
            );
        }
        ensure!(
            token_ids.len() == batch * time,
            "token buffer does not match [batch, time]"
        );
        if let Some(positions) = positions {
            ensure!(
                !positions.is_empty()
                    && positions
                        .iter()
                        .all(|&position| (position as usize) < batch * time),
                "selected forward positions are empty or outside the input"
            );
        }
        ensure!(
            token_ids
                .iter()
                .all(|&id| id < self.config.vocab_size as u32),
            "token ID outside vocabulary"
        );

        let device = self.token_embedding.embeddings().device();
        let flat_ids = Tensor::from_vec(token_ids.to_vec(), (batch * time,), device)?;
        let mut x = self
            .token_embedding
            .forward(&flat_ids)?
            .to_dtype(self.dtype)?
            .reshape((batch, time, self.config.n_embd))?;
        x = rms_norm_for(&x, self.dtype, self.inference_attention)?;
        let normalized_embedding = x.clone();

        // The first token has no previous embedding; all later tokens use the
        // normalized embedding from their left neighbor.
        if time > 1 {
            let current = x.narrow(1, 1, time - 1)?;
            let previous = x.narrow(1, 0, time - 1)?;
            let gate_input = current.narrow(2, 0, 24)?;
            let gate = ops::sigmoid(&linear_forward(
                &self.smear_gate,
                &gate_input,
                self.dtype,
                self.fp8_training,
            )?)?
            .broadcast_mul(&self.smear_lambda.to_dtype(self.dtype)?)?;
            let smeared = current.broadcast_add(&gate.broadcast_mul(&previous)?)?;
            x = Tensor::cat(&[x.narrow(1, 0, 1)?, smeared], 1)?;
        } else if let Some(cache) = cache.as_deref_mut() {
            if let Some(previous) = cache.previous_embedding.as_ref() {
                let gate_input = x.narrow(2, 0, 24)?;
                let gate = ops::sigmoid(&linear_forward(
                    &self.smear_gate,
                    &gate_input,
                    self.dtype,
                    self.fp8_training,
                )?)?
                .broadcast_mul(&self.smear_lambda.to_dtype(self.dtype)?)?;
                x = x.broadcast_add(&gate.broadcast_mul(previous)?)?;
            }
        }
        if let Some(cache) = cache.as_deref_mut() {
            cache.previous_embedding = Some(normalized_embedding.narrow(1, time - 1, 1)?);
        }
        let x0 = x.clone();
        // The same RoPE tables are used by every transformer block for this
        // batch. Build and transfer them once per forward instead of once per
        // layer.
        let (rope_cos, rope_sin) = rope_tables(
            time,
            start_position,
            self.config.n_embd / self.config.n_head,
            device,
            self.dtype,
        )?;
        let mut backout = None;
        for (layer, block) in self.blocks.iter().enumerate() {
            let resid = self
                .resid_lambdas
                .narrow(0, layer, 1)?
                .to_dtype(self.dtype)?
                .reshape((1, 1, 1))?;
            let x0_scale = self
                .x0_lambdas
                .narrow(0, layer, 1)?
                .to_dtype(self.dtype)?
                .reshape((1, 1, 1))?;
            let cpu_scalar_reduction = device.is_cpu()
                && self.dtype == DType::F32
                && has_pytorch_cpu_f32_reduction_layout();
            let residual = if cpu_scalar_reduction {
                x.apply_op2(&resid, BackoutMul)?
            } else {
                x.broadcast_mul(&resid)?
            };
            let initial = if cpu_scalar_reduction {
                x0.apply_op2(&x0_scale, BackoutMul)?
            } else {
                x0.broadcast_mul(&x0_scale)?
            };
            x = residual.broadcast_add(&initial)?;
            let value = self.value_embeddings[layer]
                .as_ref()
                .map(|embedding| {
                    embedding.forward(&flat_ids).and_then(|tensor| {
                        tensor.to_dtype(self.dtype)?.reshape((
                            batch,
                            time,
                            self.config.n_kv_head,
                            self.config.n_embd / self.config.n_head,
                        ))
                    })
                })
                .transpose()?;
            x = block_forward_with_rope(
                block,
                &x,
                value.as_ref(),
                &self.config,
                layer,
                self.dtype,
                self.fp8_training,
                self.inference_attention,
                &rope_cos,
                &rope_sin,
                start_position,
                cache.as_deref_mut(),
            )?;
            if layer == self.config.n_layer / 2 {
                backout = Some(x.clone());
            }
        }
        if let Some(mid) = backout {
            let lambda = self
                .backout_lambda
                .to_dtype(self.dtype)?
                .reshape((1, 1, 1))?;
            let backout_update = if device.is_cpu()
                && self.dtype == DType::F32
                && has_pytorch_cpu_f32_reduction_layout()
            {
                mid.apply_op2(&lambda, BackoutMul)
            } else {
                mid.broadcast_mul(&lambda)
            }?;
            x = x.broadcast_sub(&backout_update)?;
        }
        x = rms_norm_for(&x, self.dtype, self.inference_attention)?;
        if let Some(positions) = positions {
            let indices = Tensor::from_vec(positions.to_vec(), (positions.len(),), device)?;
            x = x
                .reshape((batch * time, self.config.n_embd))?
                .index_select(&indices, 0)?;
        }
        let logits = linear_forward(&self.lm_head, &x, self.dtype, self.fp8_training)?;
        let logits = if positions.is_some() {
            logits.narrow(1, 0, self.config.vocab_size)?
        } else {
            logits.narrow(2, 0, self.config.vocab_size)?
        };
        let logits = logits.to_dtype(DType::F32)?;
        let logits = if softcap_logits {
            let scaled_logits = if device.is_cpu() && logits.dtype() == DType::F32 {
                logits.apply_op1(CpuDivideScalar(LOGIT_SOFTCAP as f32))?
            } else {
                (logits / LOGIT_SOFTCAP)?
            };
            scaled_logits.tanh()?.affine(LOGIT_SOFTCAP, 0.0)?
        } else {
            logits
        };
        if let Some(cache) = cache {
            cache.position += time;
        }
        Ok(logits)
    }
}

fn model_variable_names(variables: &VarMap) -> HashSet<String> {
    variables
        .data()
        .lock()
        .expect("model variable lock is not poisoned")
        .keys()
        .cloned()
        .collect()
}

fn normalized_checkpoint_name(name: &str) -> &str {
    name.strip_prefix("_orig_mod.").unwrap_or(name)
}

fn checkpoint_source_name(name: &str, source_names: &HashSet<String>) -> String {
    if source_names.contains(name) {
        name.to_owned()
    } else {
        let compiled_name = format!("_orig_mod.{name}");
        if source_names.contains(&compiled_name) {
            compiled_name
        } else {
            name.to_owned()
        }
    }
}

fn validate_checkpoint_keys<'a>(
    source_names: impl IntoIterator<Item = &'a str>,
    model_names: &HashSet<String>,
) -> Result<()> {
    for source_name in source_names {
        let name = normalized_checkpoint_name(source_name);
        ensure!(
            model_names.contains(name),
            "unexpected model parameter {name} in checkpoint"
        );
    }
    Ok(())
}

/// Recreate nanochat's explicit `init_weights` distributions from its fixed
/// seed using the corresponding CPU, CUDA, or MPS generator stream.
fn initialize_trainable_variables(
    variables: &VarMap,
    config: &GptConfig,
    device: &Device,
    seed: u64,
) -> Result<()> {
    use std::collections::HashMap;

    let mut named = variables
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(name, variable)| (name.clone(), variable.clone()))
        .collect::<Vec<_>>();
    named.sort_by(|left, right| left.0.cmp(&right.0));
    let by_name = named.into_iter().collect::<HashMap<_, _>>();
    let matrix_bound_exact = 3.0f64.sqrt() * (config.n_embd as f64).powf(-0.5);
    let matrix_bound = matrix_bound_exact as f32;
    let mlp_bound = (matrix_bound_exact * 0.4) as f32;
    let layer_denom = config.n_layer.saturating_sub(1).max(1) as f64;
    let mut ordered = vec![
        "transformer.wte.weight".to_owned(),
        "lm_head.weight".to_owned(),
    ];
    for layer in 0..config.n_layer {
        for parameter in ["c_q", "c_k", "c_v"] {
            ordered.push(format!("transformer.h.{layer}.attn.{parameter}.weight"));
        }
        ordered.push(format!("transformer.h.{layer}.mlp.c_fc.weight"));
    }
    ordered.push("smear_gate.weight".to_owned());
    for layer in 0..config.n_layer {
        if has_value_embedding(layer, config.n_layer) {
            ordered.push(format!("value_embeds.{layer}.weight"));
        }
    }
    for layer in 0..config.n_layer {
        if has_value_embedding(layer, config.n_layer) {
            ordered.push(format!("transformer.h.{layer}.attn.ve_gate.weight"));
        }
    }

    if device.is_cpu() {
        // compute_init() seeds PyTorch's CPU generator with 42. init_weights()
        // consumes it in this explicit order; keep that stream for CPU runs.
        let mut rng = crate::engine::TorchCpuRng::new(seed);
        for name in ordered {
            let Some(variable) = by_name.get(&name) else {
                continue;
            };
            let count = variable.as_tensor().elem_count();
            let values = if name == "transformer.wte.weight" {
                torch_cpu_normal_values(&mut rng, count, 0.8)
            } else if name == "lm_head.weight" {
                torch_cpu_normal_values(&mut rng, count, 0.001)
            } else if name == "smear_gate.weight" || name.ends_with(".attn.ve_gate.weight") {
                torch_cpu_uniform_values(&mut rng, count, 0.0, 0.02)
            } else if name.ends_with(".mlp.c_fc.weight") {
                torch_cpu_uniform_values(&mut rng, count, -mlp_bound, mlp_bound)
            } else {
                torch_cpu_uniform_values(&mut rng, count, -matrix_bound, matrix_bound)
            };
            set_variable_values(variable, values, device)?;
        }
    } else if device.is_cuda() {
        // CUDA initialization uses the same seeded Philox stream and per-tensor
        // launch layout as PyTorch's CUDA distribution kernels.
        let mut rng = crate::engine::TorchCudaRng::new(seed, true);
        for name in ordered {
            let Some(variable) = by_name.get(&name) else {
                continue;
            };
            let count = variable.as_tensor().elem_count();
            let values = if name == "transformer.wte.weight" {
                (0..count)
                    .map(|row| rng.normal_f32(row, count) * 0.8)
                    .collect()
            } else if name == "lm_head.weight" {
                (0..count)
                    .map(|row| rng.normal_f32(row, count) * 0.001)
                    .collect()
            } else if name == "smear_gate.weight" || name.ends_with(".attn.ve_gate.weight") {
                torch_cuda_uniform_values(&mut rng, count, 0.0, 0.02)
            } else if name.ends_with(".mlp.c_fc.weight") {
                torch_cuda_uniform_values(&mut rng, count, -mlp_bound, mlp_bound)
            } else {
                torch_cuda_uniform_values(&mut rng, count, -matrix_bound, matrix_bound)
            };
            set_variable_values(variable, values, device)?;
        }
    } else {
        // PyTorch 2.9.1 keeps MPS Philox counters in its generator but delegates
        // normal/uniform values to MPSGraph. Use the corresponding seeded
        // Philox stream and the same parameter order as the CPU/CUDA paths;
        // exact value parity still needs comparison against MPSGraph itself.
        let mut rng = crate::engine::TorchMpsRng::new(seed);
        for name in ordered {
            let Some(variable) = by_name.get(&name) else {
                continue;
            };
            let count = variable.as_tensor().elem_count();
            let values = if name == "transformer.wte.weight" {
                Some(rng.normal_values(count, 0.0, 0.8))
            } else if name == "lm_head.weight" {
                Some(rng.normal_values(count, 0.0, 0.001))
            } else if name.ends_with(".mlp.c_fc.weight") {
                Some(rng.uniform_values(count, -mlp_bound, mlp_bound))
            } else if name == "smear_gate.weight" || name.ends_with(".attn.ve_gate.weight") {
                Some(rng.uniform_values(count, 0.0, 0.02))
            } else {
                Some(rng.uniform_values(count, -matrix_bound, matrix_bound))
            };
            if let Some(values) = values {
                set_variable_values(variable, values, device)?;
            }
        }
    }

    for (name, variable) in &by_name {
        let shape = variable.as_tensor().shape().clone();
        let count = shape.elem_count();
        let values = if name == "transformer.wte.weight" {
            None
        } else if name == "lm_head.weight" {
            None
        } else if name == "resid_lambdas" {
            Some(
                (0..config.n_layer)
                    .map(|layer| (1.15f64 - 0.10f64 * layer as f64 / layer_denom) as f32)
                    .collect(),
            )
        } else if name == "x0_lambdas" {
            Some(
                (0..config.n_layer)
                    .map(|layer| (0.20f64 - 0.15f64 * layer as f64 / layer_denom) as f32)
                    .collect(),
            )
        } else if name == "backout_lambda" {
            Some(vec![0.2; count])
        } else if name == "smear_lambda"
            || name.ends_with(".attn.c_proj.weight")
            || name.ends_with(".mlp.c_proj.weight")
        {
            Some(vec![0.0; count])
        } else if name.ends_with(".attn.c_q.weight")
            || name.ends_with(".attn.c_k.weight")
            || name.ends_with(".attn.c_v.weight")
            || name.starts_with("value_embeds.")
        {
            None
        } else if name.ends_with(".mlp.c_fc.weight") {
            None
        } else if name == "smear_gate.weight" || name.ends_with(".attn.ve_gate.weight") {
            None
        } else {
            anyhow::bail!("no fresh-training initializer for parameter `{name}`")
        };
        if let Some(values) = values {
            set_variable_values(variable, values, device)?;
        }
    }
    Ok(())
}

fn set_variable_values(
    variable: &candle_core::Var,
    values: Vec<f32>,
    device: &Device,
) -> Result<()> {
    let tensor = Tensor::from_vec(values, variable.as_tensor().shape().clone(), device)?
        .to_dtype(variable.dtype())?;
    variable.set(&tensor)?;
    Ok(())
}

fn torch_cpu_uniform_values(
    rng: &mut crate::engine::TorchCpuRng,
    count: usize,
    low: f32,
    high: f32,
) -> Vec<f32> {
    (0..count)
        .map(|_| {
            let sample = rng.uniform_f32();
            sample * (high - low) + low
        })
        .collect()
}

fn torch_cuda_uniform_values(
    rng: &mut crate::engine::TorchCudaRng,
    count: usize,
    low: f32,
    high: f32,
) -> Vec<f32> {
    (0..count)
        .map(|row| {
            let value = rng.uniform_f32(row, count) * (high - low) + low;
            if value == high { low } else { value }
        })
        .collect()
}

fn torch_cpu_normal_values(
    rng: &mut crate::engine::TorchCpuRng,
    count: usize,
    standard_deviation: f32,
) -> Vec<f32> {
    let samples = (0..count).map(|_| rng.uniform_f32()).collect::<Vec<_>>();
    // PyTorch's CPU normal kernel transforms uniform values in blocks of 16,
    // placing the cosine half before the sine half.
    let mut result = vec![0.0; count];
    let mut start = 0;
    while start + 16 <= count {
        transform_normal_block(
            &samples[start..start + 16],
            &mut result[start..start + 16],
            standard_deviation,
        );
        start += 16;
    }
    if count % 16 != 0 {
        let output_start = count.saturating_sub(16);
        let tail = (0..16).map(|_| rng.uniform_f32()).collect::<Vec<_>>();
        transform_normal_block(
            &tail,
            &mut result[output_start..output_start + 16],
            standard_deviation,
        );
    }
    result
}

fn transform_normal_block(uniforms: &[f32], output: &mut [f32], standard_deviation: f32) {
    for index in 0..8 {
        let u1 = 1.0 - uniforms[index];
        let u2 = uniforms[index + 8];
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f32::consts::TAU * u2;
        output[index] = radius * angle.cos() * standard_deviation;
        output[index + 8] = radius * angle.sin() * standard_deviation;
    }
}

fn linear(input: usize, output: usize, vb: VarBuilder<'_>, init: Init) -> Result<Linear> {
    let weight = vb.get_with_hints((output, input), "weight", init)?;
    Ok(Linear::new(weight, None))
}

fn dtype_bytes(dtype: DType) -> usize {
    match dtype {
        DType::F16 | DType::BF16 => 2,
        DType::F32 => 4,
        DType::F64 => 8,
        _ => 4,
    }
}

fn embedding_at(vocab: usize, width: usize, vb: VarBuilder<'_>, init: Init) -> Result<Embedding> {
    let weight = vb.get_with_hints((vocab, width), "weight", init)?;
    Ok(Embedding::new(weight, width))
}

fn has_value_embedding(layer: usize, layers: usize) -> bool {
    layer % 2 == (layers - 1) % 2
}

#[cfg(test)]
fn rms_norm(x: &Tensor, dtype: DType) -> Result<Tensor> {
    rms_norm_for(x, dtype, false)
}

fn rms_norm_for(x: &Tensor, dtype: DType, use_fused_cuda_kernel: bool) -> Result<Tensor> {
    // PyTorch RMSNorm uses FP32 op-math epsilon and accumulation for F16/BF16
    // inputs, then returns the result in the input dtype. On representative
    // depth-12 traces, Candle's fused CUDA kernel matches the width-768 MLP
    // normalization, while the FP32 decomposition matches width-128 Q/K rows;
    // those paths can differ by one BF16 ULP when their shapes are crossed.
    if use_fused_cuda_kernel
        && x.device().is_cuda()
        && matches!(dtype, DType::F16 | DType::BF16)
        && x.dims().last().copied().unwrap_or(0) >= 256
    {
        let width = *x.dims().last().unwrap_or(&0);
        let alpha = Tensor::ones((width,), dtype, x.device())?;
        return Ok(candle_nn::ops::rms_norm(x, &alpha, RMS_EPS_F32 as f32)?);
    }
    if x.device().is_cpu()
        && has_pytorch_cpu_f32_reduction_layout()
        && dtype == DType::F32
        && x.dtype() == DType::F32
        && x.dims().last().copied().unwrap_or(0) >= 8
    {
        let input = x.contiguous()?;
        let width = *input.dims().last().unwrap();
        return Ok(input.apply_op1(CpuRmsNorm { width })?);
    }
    let x_f32 = x.to_dtype(DType::F32)?;
    let squares = x_f32.sqr()?;
    let mean_square = if x.device().is_cpu()
        && has_pytorch_cpu_f32_reduction_layout()
        && x.dims().last().copied().unwrap_or(0) >= 8
    {
        squares.contiguous()?.apply_op1(CpuMeanLast)?
    } else {
        squares.mean_keepdim(candle_core::D::Minus1)?
    };
    let adjusted = mean_square.affine(1.0, RMS_EPS_F32)?;
    let inverse_rms = if x.device().is_cpu() && has_pytorch_cpu_f32_reduction_layout() {
        adjusted.contiguous()?.apply_op1(CpuRsqrt)?
    } else {
        adjusted.sqrt()?.recip()?
    };
    Ok(x_f32.broadcast_mul(&inverse_rms)?.to_dtype(dtype)?)
}

/// Apply a linear layer in the activation dtype while leaving its stored
/// weights and optimizer master copy at their original precision.
fn linear_forward(
    layer: &Linear,
    input: &Tensor,
    dtype: DType,
    fp8_training: bool,
) -> Result<Tensor> {
    let master_weight = layer.weight();
    let (out_features, in_features) = master_weight.dims2()?;
    if fp8_training
        && input.device().is_cuda()
        && in_features % 16 == 0
        && out_features % 16 == 0
        && in_features.min(out_features) >= 128
    {
        #[cfg(feature = "cuda-base")]
        return Ok(crate::fp8::linear(
            input,
            master_weight,
            layer.bias(),
            dtype,
        )?);
        #[cfg(not(feature = "cuda-base"))]
        anyhow::bail!("FP8 training requires a build with --features cuda-base");
    }
    let weight = master_weight.to_dtype(dtype)?;
    let bias = layer.bias().map(|bias| bias.to_dtype(dtype)).transpose()?;
    let compute_layer = Linear::new(weight, bias);
    // CUDA matmul requires contiguous input layouts. Slices and narrow views
    // used by smear and gate projections are often strided, though CPU accepts
    // them directly.
    let input = input.to_dtype(dtype)?.contiguous()?;
    Ok(compute_layer.forward(&input)?)
}

fn relu_squared(input: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda-base")]
    if input.device().is_cuda() {
        return Ok(input.apply_op1(CudaReluSquared)?);
    }
    Ok(input.relu()?.sqr()?)
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let width = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, width)?;
    let x2 = x.narrow(3, width, width)?;
    let y1 = x1
        .broadcast_mul(cos)?
        .broadcast_add(&x2.broadcast_mul(sin)?)?;
    let y2 = x2
        .broadcast_mul(cos)?
        .broadcast_sub(&x1.broadcast_mul(sin)?)?;
    Ok(Tensor::cat(&[y1, y2], 3)?)
}

fn scale_qk(x: &Tensor, dtype: DType) -> Result<Tensor> {
    // PyTorch applies a float scalar in FP32 op-math before casting back to
    // BF16/F16; Candle's low-precision affine rounds the scalar first.
    Ok(x.to_dtype(DType::F32)?.affine(1.2, 0.0)?.to_dtype(dtype)?)
}

fn rope_tables(
    time: usize,
    start_position: usize,
    head_dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(time * half);
    let mut sin = Vec::with_capacity(time * half);
    let inv_freq = (0..half)
        .map(|channel| {
            let exponent = (2 * channel) as f32 / head_dim as f32;
            1.0f32 / (ROPE_BASE as f32).powf(exponent)
        })
        .collect::<Vec<_>>();
    for position in start_position..start_position + time {
        for &frequency in &inv_freq {
            let angle = position as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let cos = Tensor::from_vec(cos, (1, time, 1, half), device)?.to_dtype(dtype)?;
    let sin = Tensor::from_vec(sin, (1, time, 1, half), device)?.to_dtype(dtype)?;
    Ok((cos, sin))
}

fn mix_value_embeddings(v: &Tensor, embedding: &Tensor, gate_logits: &Tensor) -> Result<Tensor> {
    let gate = ops::sigmoid(gate_logits)?.affine(3.0, 0.0)?.unsqueeze(3)?;
    Ok(v.broadcast_add(&embedding.broadcast_mul(&gate)?)?)
}

#[cfg(test)]
fn block_forward(
    block: &Block,
    x: &Tensor,
    value_embedding: Option<&Tensor>,
    config: &GptConfig,
    layer: usize,
    dtype: DType,
    fp8_training: bool,
    inference_attention: bool,
    start_position: usize,
    cache: Option<&mut KvCache>,
) -> Result<Tensor> {
    let time = x.dim(1)?;
    let (cos, sin) = rope_tables(
        time,
        start_position,
        config.n_embd / config.n_head,
        x.device(),
        dtype,
    )?;
    block_forward_with_rope(
        block,
        x,
        value_embedding,
        config,
        layer,
        dtype,
        fp8_training,
        inference_attention,
        &cos,
        &sin,
        start_position,
        cache,
    )
}

fn block_forward_with_rope(
    block: &Block,
    x: &Tensor,
    value_embedding: Option<&Tensor>,
    config: &GptConfig,
    layer: usize,
    dtype: DType,
    fp8_training: bool,
    inference_attention: bool,
    cos: &Tensor,
    sin: &Tensor,
    start_position: usize,
    mut cache: Option<&mut KvCache>,
) -> Result<Tensor> {
    let (batch, time, _) = x.dims3()?;
    let head_dim = config.n_embd / config.n_head;
    let kv_heads = config.n_kv_head;
    let groups = config.n_head / kv_heads;
    let normalized = rms_norm_for(x, dtype, inference_attention)?;
    let mut q = linear_forward(&block.attn.q, &normalized, dtype, fp8_training)?.reshape((
        batch,
        time,
        config.n_head,
        head_dim,
    ))?;
    let mut k = linear_forward(&block.attn.k, &normalized, dtype, fp8_training)?
        .reshape((batch, time, kv_heads, head_dim))?;
    let mut v = linear_forward(&block.attn.v, &normalized, dtype, fp8_training)?
        .reshape((batch, time, kv_heads, head_dim))?;
    if let (Some(ve), Some(gate_layer)) = (value_embedding, block.attn.ve_gate.as_ref()) {
        let gate_input = normalized.narrow(2, 0, 12)?;
        let gate_logits = linear_forward(gate_layer, &gate_input, dtype, fp8_training)?;
        v = mix_value_embeddings(&v, ve, &gate_logits)?;
    }
    q = apply_rope(&q, cos, sin)?;
    k = apply_rope(&k, cos, sin)?;
    q = scale_qk(&rms_norm_for(&q, dtype, inference_attention)?, dtype)?.transpose(1, 2)?;
    k = scale_qk(&rms_norm_for(&k, dtype, inference_attention)?, dtype)?.transpose(1, 2)?;
    v = v.transpose(1, 2)?;

    let differentiable_attention = !inference_attention && cache.is_none();
    let mut key_start = 0;
    let mut key_time = time;
    if let Some(cache) = cache.as_deref_mut() {
        if let Some(previous_keys) = cache.keys[layer].as_ref() {
            key_time += previous_keys.dim(2)?;
            k = Tensor::cat(&[previous_keys, &k], 2)?;
        } else {
            key_start = start_position;
        }
        if let Some(previous_values) = cache.values[layer].as_ref() {
            v = Tensor::cat(&[previous_values, &v], 2)?;
        }
        cache.keys[layer] = Some(k.clone());
        cache.values[layer] = Some(v.clone());
    }

    let mask_width = if config.is_long_window_layer(layer) {
        config.sequence_len
    } else {
        ((config.sequence_len + 511) / 512) * 128
    };
    #[cfg(feature = "flash-attention")]
    let attended = if x.device().is_cuda() && matches!(q.dtype(), DType::F16 | DType::BF16) {
        let q_flash = q.transpose(1, 2)?;
        let k_flash = k.transpose(1, 2)?;
        let v_flash = v.transpose(1, 2)?;
        let left_window = (!config.is_long_window_layer(layer)).then_some(mask_width);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = if differentiable_attention {
            q_flash.apply_op3(
                &k_flash,
                &v_flash,
                TrainFlashAttention {
                    scale,
                    window: left_window,
                },
            )?
        } else {
            candle_flash_attn::flash_attn_windowed(
                &q_flash,
                &k_flash,
                &v_flash,
                scale,
                left_window,
                Some(0),
            )?
        };
        output.reshape((batch, time, config.n_embd))?
    } else {
        fallback_attention(
            &q,
            &k,
            &v,
            batch,
            time,
            start_position,
            key_start,
            key_time,
            mask_width,
            groups,
            head_dim,
            x.device(),
            differentiable_attention,
        )?
    };
    #[cfg(not(feature = "flash-attention"))]
    let attended = fallback_attention(
        &q,
        &k,
        &v,
        batch,
        time,
        start_position,
        key_start,
        key_time,
        mask_width,
        groups,
        head_dim,
        x.device(),
        differentiable_attention,
    )?;
    let attended = linear_forward(&block.attn.proj, &attended, dtype, fp8_training)?;
    let x = x.broadcast_add(&attended)?;

    let normalized = rms_norm_for(&x, dtype, inference_attention)?;
    let hidden = relu_squared(&linear_forward(
        &block.mlp.fc,
        &normalized,
        dtype,
        fp8_training,
    )?)?;
    let mlp = linear_forward(&block.mlp.proj, &hidden, dtype, fp8_training)?;
    Ok(x.broadcast_add(&mlp)?)
}

#[allow(clippy::too_many_arguments)]
fn fallback_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batch: usize,
    time: usize,
    start_position: usize,
    key_start: usize,
    key_time: usize,
    mask_width: usize,
    groups: usize,
    head_dim: usize,
    device: &Device,
    differentiable: bool,
) -> Result<Tensor> {
    // Explicit CUDA inference attention accumulates scores and the weighted
    // value sum in FP32, as PyTorch SDPA does for low-precision inputs. Keep
    // the differentiable training path in its activation dtype.
    let use_fp32 =
        !differentiable && device.is_cuda() && matches!(q.dtype(), DType::F16 | DType::BF16);
    let q_fp32 = use_fp32.then(|| q.to_dtype(DType::F32)).transpose()?;
    let k_fp32 = use_fp32.then(|| k.to_dtype(DType::F32)).transpose()?;
    let v_fp32 = use_fp32.then(|| v.to_dtype(DType::F32)).transpose()?;
    let q_work = q_fp32.as_ref().unwrap_or(q);
    let k_work = k_fp32.as_ref().unwrap_or(k);
    let v_work = v_fp32.as_ref().unwrap_or(v);
    let kv_heads = k_work.dim(1)?;
    let key_transposed = k_work.transpose(2, 3)?.contiguous()?;
    let values_contiguous = v_work.contiguous()?;
    const QUERY_CHUNK: usize = 128;
    let mut attended_chunks = Vec::with_capacity(time.div_ceil(QUERY_CHUNK));
    for query_start in (0..time).step_by(QUERY_CHUNK) {
        let query_count = QUERY_CHUNK.min(time - query_start);
        let query_chunk = q_work.narrow(2, query_start, query_count)?.contiguous()?;
        let mut mask = Vec::with_capacity(query_count * key_time);
        for query in query_start..query_start + query_count {
            let absolute_query = start_position + query;
            for key in 0..key_time {
                let absolute_key = key_start + key;
                mask.push(
                    if absolute_key <= absolute_query && absolute_query - absolute_key <= mask_width
                    {
                        0f32
                    } else {
                        -1.0e30
                    },
                );
            }
        }
        let mask = Tensor::from_vec(mask, (1, 1, query_count, key_time), device)?
            .to_dtype(q_work.dtype())?;
        let attended = if groups == 1 {
            let scores = query_chunk
                .matmul(&key_transposed)?
                .affine(1.0 / (head_dim as f64).sqrt(), 0.0)?
                .broadcast_add(&mask)?;
            let weights = attention_softmax(&scores, differentiable)?;
            weights.matmul(&values_contiguous)?
        } else {
            let mut grouped_outputs = Vec::with_capacity(kv_heads);
            for kv_head in 0..kv_heads {
                let query_heads = query_chunk
                    .narrow(1, kv_head * groups, groups)?
                    .contiguous()?;
                let key_head = k_work.narrow(1, kv_head, 1)?;
                let value_head = values_contiguous.narrow(1, kv_head, 1)?.contiguous()?;
                let scores = query_heads
                    .broadcast_matmul(&key_head.transpose(2, 3)?.contiguous()?)?
                    .affine(1.0 / (head_dim as f64).sqrt(), 0.0)?
                    .broadcast_add(&mask)?;
                let weights = attention_softmax(&scores, differentiable)?;
                grouped_outputs.push(weights.broadcast_matmul(&value_head)?);
            }
            Tensor::cat(&grouped_outputs, 1)?
        };
        attended_chunks.push(attended);
    }
    let attended = Tensor::cat(&attended_chunks, 2)?
        .transpose(1, 2)?
        .reshape((batch, time, q.dim(1)? * head_dim))?;
    Ok(if use_fp32 {
        attended.to_dtype(q.dtype())?
    } else {
        attended
    })
}

// FlashAttention's Candle binding supplies a forward kernel only. Recompute
// bounded query blocks in the backward pass instead of retaining every score
// matrix in the forward graph. This supports the same causal window and GQA
// layouts as the normal attention path.
#[cfg(feature = "flash-attention")]
struct CudaAttentionScaleMask {
    scale: f32,
    query_start: usize,
    key_start: usize,
    window: Option<usize>,
}

#[cfg(feature = "flash-attention")]
impl CustomOp1 for CudaAttentionScaleMask {
    fn name(&self) -> &'static str {
        "cuda_attention_scale_causal_mask"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA attention mask requires CUDA")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda::{
            CudaStorageSlice,
            cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg},
        };

        let dims = layout.dims();
        if dims.len() != 4 || !layout.is_contiguous() || layout.start_offset() != 0 {
            candle_core::bail!(
                "CUDA attention mask expects contiguous rank-4 scores, got {layout:?}"
            )
        }
        let [_, _, queries, keys] = [dims[0], dims[1], dims[2], dims[3]];
        let CudaStorageSlice::F32(input_slice) = &input.slice else {
            candle_core::bail!("CUDA attention mask expects FP32 scores")
        };
        let device = &input.device;
        let stream = device.cuda_stream();
        let (src, _src_record) = input_slice.device_ptr(&stream);
        let mut output = unsafe { device.alloc::<f32>(layout.shape().elem_count())? };
        let kernel = r#"
        extern "C" __global__ void scale_causal_mask(
            const float* x, float* y, unsigned long long n,
            unsigned int queries, unsigned int keys, unsigned int query_start,
            unsigned int key_start, unsigned int window, float scale
        ) {
          unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
          if (i >= n) return;
          unsigned int query = query_start + (unsigned int)((i / keys) % queries);
          unsigned int key = key_start + (unsigned int)(i % keys);
          bool masked = key > query || (window != 0xffffffffu && query - key > window);
          y[i] = masked ? -1.0e30f : x[i] * scale;
        }"#;
        use std::sync::OnceLock;
        static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
        let ptx = PTX.get_or_init(|| {
            candle_core::cuda::cudarc::nvrtc::compile_ptx(kernel)
                .map(|ptx| ptx.to_src())
                .map_err(|error| error.to_string())
        });
        let ptx = ptx.as_ref().map_err(|error| {
            candle_core::Error::Msg(format!("compile attention scale-mask kernel: {error}"))
        })?;
        let function = device.get_or_load_custom_func(
            "scale_causal_mask",
            "rust_agent_attention_scale_mask",
            ptx,
        )?;
        let n = layout.shape().elem_count() as u64;
        let queries = queries as u32;
        let keys = keys as u32;
        let query_start = self.query_start as u32;
        let key_start = self.key_start as u32;
        let window = self.window.map_or(u32::MAX, |value| value as u32);
        let scale = self.scale;
        let config = LaunchConfig::for_num_elems(n as u32);
        let mut launch = function.builder();
        launch.arg(&src);
        launch.arg(&mut output);
        launch.arg(&n);
        launch.arg(&queries);
        launch.arg(&keys);
        launch.arg(&query_start);
        launch.arg(&key_start);
        launch.arg(&window);
        launch.arg(&scale);
        unsafe { launch.launch(config) }.map_err(|error| {
            candle_core::Error::Msg(format!("launch attention scale-mask kernel: {error}"))
        })?;
        drop(_src_record);
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device.clone()),
            layout.shape().clone(),
        ))
    }
}

#[cfg(feature = "flash-attention")]
struct TrainFlashAttention {
    scale: f32,
    window: Option<usize>,
}

#[cfg(feature = "flash-attention")]
impl candle_core::CustomOp3 for TrainFlashAttention {
    fn name(&self) -> &'static str {
        "train-flash-attention"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("train-flash-attention requires CUDA")
    }

    fn cuda_fwd(
        &self,
        q: &candle_core::CudaStorage,
        ql: &Layout,
        k: &candle_core::CudaStorage,
        kl: &Layout,
        v: &candle_core::CudaStorage,
        vl: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        let op = candle_flash_attn::FlashAttn {
            softmax_scale: self.scale,
            alibi_slopes: None,
            window_size_left: self.window,
            window_size_right: Some(0),
            softcap: None,
        };
        candle_core::CustomOp3::cuda_fwd(&op, q, ql, k, kl, v, vl)
    }

    fn bwd(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let (batch, time, heads, width) = q.dims4()?;
        let (_, key_time, kv_heads, _) = k.dims4()?;
        if time != key_time || heads % kv_heads != 0 {
            candle_core::bail!(
                "training attention expects matching sequence lengths and grouped heads"
            );
        }
        let mut q_parts = Vec::new();
        let cuda_device = q.device().as_cuda_device()?;
        let mut k_grad = cuda_device.alloc_zeros::<f32>(k.elem_count())?;
        let mut v_grad = cuda_device.alloc_zeros::<f32>(v.elem_count())?;
        const BLOCK: usize = 256;
        let groups = heads / kv_heads;
        let scale = self.scale as f64;
        // Convert and expand the shared K/V heads only once. Doing this per
        // query block repeatedly copied a quadratic amount of data for dense
        // attention, even though the underlying keys and values are fixed.
        let k_full = k
            .to_dtype(DType::F16)?
            .transpose(1, 2)?
            .reshape((batch, kv_heads, 1, key_time, width))?
            .broadcast_as((batch, kv_heads, groups, key_time, width))?
            .contiguous()?
            .reshape((batch, heads, key_time, width))?;
        let v_full = v
            .to_dtype(DType::F16)?
            .transpose(1, 2)?
            .reshape((batch, kv_heads, 1, key_time, width))?
            .broadcast_as((batch, kv_heads, groups, key_time, width))?
            .contiguous()?
            .reshape((batch, heads, key_time, width))?;
        for start in (0..time).step_by(BLOCK) {
            let count = BLOCK.min(time - start);
            let key_start = self.window.map_or(0, |window| start.saturating_sub(window));
            let end = start + count;
            let key_count = end - key_start;
            // Use the standard attention derivative directly instead of
            // constructing an autograd graph for every query block. The
            // block keeps the score/probability tensors bounded at long
            // context lengths and removes the graph-building overhead from
            // the prior per-block autograd implementation.
            let q_block = q
                .narrow(1, start, count)?
                .to_dtype(DType::F16)?
                .transpose(1, 2)?
                .contiguous()?;
            let k_block = k_full.narrow(2, key_start, key_count)?.contiguous()?;
            let v_block = v_full.narrow(2, key_start, key_count)?.contiguous()?;
            let upstream = grad
                .narrow(1, start, count)?
                .to_dtype(DType::F16)?
                .transpose(1, 2)?
                .contiguous()?;
            let scores = q_block
                .broadcast_matmul(&k_block.transpose(2, 3)?.contiguous()?)?
                .to_dtype(DType::F32)?
                .apply_op1(CudaAttentionScaleMask {
                    scale: self.scale,
                    query_start: start,
                    key_start,
                    window: self.window,
                })?;
            let probabilities = ops::softmax_last_dim(&scores)?;
            let d_probabilities = upstream
                .broadcast_matmul(&v_block.transpose(2, 3)?.contiguous()?)?
                .to_dtype(DType::F32)?;
            let d_values = probabilities
                .to_dtype(DType::F16)?
                .transpose(2, 3)?
                .contiguous()?
                .broadcast_matmul(&upstream)?
                .to_dtype(DType::F32)?
                .reshape((batch, kv_heads, groups, key_count, width))?
                .sum(2)?;
            let d_scores = probabilities.broadcast_mul(
                &d_probabilities.broadcast_sub(
                    &d_probabilities
                        .broadcast_mul(&probabilities)?
                        .sum_keepdim(3)?,
                )?,
            )?;
            let d_queries = d_scores
                .to_dtype(DType::F16)?
                .broadcast_matmul(&k_block)?
                .to_dtype(DType::F32)?
                .affine(scale, 0.0)?
                .transpose(1, 2)?;
            let d_keys = d_scores
                .to_dtype(DType::F16)?
                .transpose(2, 3)?
                .contiguous()?
                .broadcast_matmul(&q_block)?
                .to_dtype(DType::F32)?
                .affine(scale, 0.0)?
                .reshape((batch, kv_heads, groups, key_count, width))?
                .sum(2)?;
            q_parts.push(d_queries.detach());
            accumulate_attention_gradient(
                &mut k_grad,
                &d_keys,
                batch,
                kv_heads,
                key_time,
                key_start,
                key_count,
                width,
            )?;
            accumulate_attention_gradient(
                &mut v_grad,
                &d_values,
                batch,
                kv_heads,
                key_time,
                key_start,
                key_count,
                width,
            )?;
        }
        let k_grad = cuda_f32_buffer_tensor(k_grad, k.shape().clone(), q.device())?;
        let v_grad = cuda_f32_buffer_tensor(v_grad, v.shape().clone(), q.device())?;
        Ok((
            Some(Tensor::cat(&q_parts, 1)?.to_dtype(q.dtype())?),
            Some(k_grad.to_dtype(k.dtype())?),
            Some(v_grad.to_dtype(v.dtype())?),
        ))
    }
}

#[cfg(feature = "flash-attention")]
fn cuda_f32_buffer_tensor(
    buffer: candle_core::cuda::cudarc::driver::CudaSlice<f32>,
    shape: Shape,
    device: &Device,
) -> candle_core::Result<Tensor> {
    use candle_core::{Storage, op::BackpropOp};

    Ok(Tensor::from_storage(
        Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            buffer,
            device.as_cuda_device()?.clone(),
        )),
        shape,
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "flash-attention")]
#[allow(clippy::too_many_arguments)]
fn accumulate_attention_gradient(
    destination: &mut candle_core::cuda::cudarc::driver::CudaSlice<f32>,
    contribution: &Tensor,
    batch: usize,
    heads: usize,
    full_time: usize,
    start_time: usize,
    time: usize,
    width: usize,
) -> candle_core::Result<()> {
    use std::sync::OnceLock;

    use candle_core::{
        Storage,
        cuda::{
            CudaStorageSlice,
            cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
        },
    };

    let contribution = contribution.contiguous()?;
    let (storage, layout) = contribution.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("attention gradient contribution must be on CUDA")
    };
    let CudaStorageSlice::F32(source) = &storage.slice else {
        candle_core::bail!("attention gradient contribution must be FP32")
    };
    if !layout.is_contiguous() || layout.start_offset() != 0 {
        candle_core::bail!("attention gradient contribution must be contiguous")
    }
    let device = &storage.device;
    let stream = device.cuda_stream();
    let (source_ptr, _source_record) = source.device_ptr(&stream);
    let (destination_ptr, _destination_record) = destination.device_ptr_mut(&stream);
    const KERNEL: &str = r#"
    extern "C" __global__ void accumulate_attention_gradient(
        const float* source, float* destination,
        unsigned long long n, unsigned int heads,
        unsigned int full_time, unsigned int start_time,
        unsigned int time, unsigned int width) {
      unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n) return;
      unsigned int dim = i % width;
      unsigned int token = (i / width) % time;
      unsigned int head = (i / (width * time)) % heads;
      unsigned int batch = i / ((unsigned long long)width * time * heads);
      unsigned long long dst = ((batch * full_time + start_time + token) * heads + head) * width + dim;
      destination[dst] += source[i];
    }"#;
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    let ptx = PTX.get_or_init(|| {
        candle_core::cuda::cudarc::nvrtc::compile_ptx(KERNEL)
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
    });
    let ptx = ptx.as_ref().map_err(|error| {
        candle_core::Error::Msg(format!(
            "compile attention gradient accumulation kernel: {error}"
        ))
    })?;
    let function = device.get_or_load_custom_func(
        "accumulate_attention_gradient",
        "rust_agent_attention_grad",
        ptx,
    )?;
    let count = (batch * heads * time * width) as u64;
    let heads = heads as u32;
    let full_time = full_time as u32;
    let start_time = start_time as u32;
    let time = time as u32;
    let width = width as u32;
    let cfg = LaunchConfig::for_num_elems(count as u32);
    let mut launch = function.builder();
    launch.arg(&source_ptr);
    launch.arg(&destination_ptr);
    launch.arg(&count);
    launch.arg(&heads);
    launch.arg(&full_time);
    launch.arg(&start_time);
    launch.arg(&time);
    launch.arg(&width);
    unsafe { launch.launch(cfg) }.map_err(|error| {
        candle_core::Error::Msg(format!(
            "launch attention gradient accumulation kernel: {error}"
        ))
    })?;
    drop(_source_record);
    drop(_destination_record);
    Ok(())
}

fn attention_softmax(scores: &Tensor, differentiable: bool) -> Result<Tensor> {
    if differentiable {
        Ok(ops::softmax(scores, scores.rank() - 1)?)
    } else {
        Ok(ops::softmax_last_dim(scores)?)
    }
}

#[cfg(test)]
mod attention_gradient_tests {
    use super::{
        CpuDivideScalar, LOGIT_SOFTCAP, attention_softmax, fallback_attention, rms_norm,
        rope_tables,
    };

    #[cfg(feature = "flash-attention")]
    #[test]
    fn cuda_train_flash_gradients_track_fallback_for_causal_and_windowed_attention() {
        use super::TrainFlashAttention;
        let device = Device::new_cuda(0).unwrap();
        for window in [None, Some(3)] {
            let make_var = |offset: f32| {
                let data = (0..(2 * 529 * 128))
                    .map(|i| ((i as f32 * 0.017 + offset).sin()) * 0.3)
                    .collect::<Vec<_>>();
                Var::from_tensor(
                    &Tensor::from_vec(data, (1, 529, 2, 128), &device)
                        .unwrap()
                        .to_dtype(DType::BF16)
                        .unwrap(),
                )
                .unwrap()
            };
            let (q, k, v) = (make_var(0.1), make_var(0.4), make_var(0.7));
            let flash = q
                .apply_op3(
                    &k,
                    &v,
                    TrainFlashAttention {
                        scale: 1.0 / (128.0f32).sqrt(),
                        window,
                    },
                )
                .unwrap();
            let reference = fallback_attention(
                &q.transpose(1, 2).unwrap(),
                &k.transpose(1, 2).unwrap(),
                &v.transpose(1, 2).unwrap(),
                1,
                529,
                0,
                0,
                529,
                window.unwrap_or(529),
                1,
                128,
                &device,
                true,
            )
            .unwrap()
            .reshape((1, 529, 2, 128))
            .unwrap();
            let flash_grads = flash
                .to_dtype(DType::F32)
                .unwrap()
                .sum_all()
                .unwrap()
                .backward()
                .unwrap();
            let ref_grads = reference
                .to_dtype(DType::F32)
                .unwrap()
                .sum_all()
                .unwrap()
                .backward()
                .unwrap();
            for var in [&q, &k, &v] {
                let a = flash_grads
                    .get(var)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let b = ref_grads
                    .get(var)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let max = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max < 0.04,
                    "window={window:?} gradient max difference={max}"
                );
            }
        }
    }

    #[cfg(feature = "cuda-base")]
    #[test]
    fn cuda_fused_cross_entropy_matches_candle_loss_and_gradients() {
        let device = Device::new_cuda(0).unwrap();
        let values = (0..(3 * 17))
            .map(|index| (index as f32 * 0.13).sin() * 4.0)
            .collect::<Vec<_>>();
        let targets = Tensor::from_vec(vec![1u32, 8, 16], (3,), &device).unwrap();
        let fused_logits =
            Var::from_tensor(&Tensor::from_vec(values.clone(), (3, 17), &device).unwrap()).unwrap();
        let reference_logits =
            Var::from_tensor(&Tensor::from_vec(values, (3, 17), &device).unwrap()).unwrap();
        let fused_loss = super::training_cross_entropy(fused_logits.as_tensor(), &targets).unwrap();
        let reference_loss =
            candle_nn::loss::cross_entropy(reference_logits.as_tensor(), &targets).unwrap();
        let loss_difference = (fused_loss.to_scalar::<f32>().unwrap()
            - reference_loss.to_scalar::<f32>().unwrap())
        .abs();
        assert!(loss_difference < 1e-5, "loss difference={loss_difference}");
        let fused_gradient = fused_loss
            .backward()
            .unwrap()
            .get(&fused_logits)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let reference_gradient = reference_loss
            .backward()
            .unwrap()
            .get(&reference_logits)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let gradient_difference = fused_gradient
            .iter()
            .flatten()
            .zip(reference_gradient.iter().flatten())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            gradient_difference < 1e-5,
            "gradient difference={gradient_difference}"
        );

        let softcap_values = (0..(3 * 17))
            .map(|index| (index as f32 * 0.13).sin() * 20.0)
            .collect::<Vec<_>>();
        let fused_softcap_logits =
            Var::from_tensor(&Tensor::from_vec(softcap_values.clone(), (3, 17), &device).unwrap())
                .unwrap();
        let reference_softcap_logits =
            Var::from_tensor(&Tensor::from_vec(softcap_values, (3, 17), &device).unwrap()).unwrap();
        let fused_softcap_loss =
            super::training_softcap_cross_entropy(fused_softcap_logits.as_tensor(), &targets)
                .unwrap();
        let reference_softcapped = reference_softcap_logits
            .as_tensor()
            .affine(1.0 / LOGIT_SOFTCAP, 0.0)
            .unwrap()
            .tanh()
            .unwrap()
            .affine(LOGIT_SOFTCAP, 0.0)
            .unwrap();
        let reference_softcap_loss =
            candle_nn::loss::cross_entropy(&reference_softcapped, &targets).unwrap();
        let softcap_loss_difference = (fused_softcap_loss.to_scalar::<f32>().unwrap()
            - reference_softcap_loss.to_scalar::<f32>().unwrap())
        .abs();
        assert!(
            softcap_loss_difference < 1e-5,
            "softcap loss difference={softcap_loss_difference}"
        );
        let fused_softcap_gradient = fused_softcap_loss
            .backward()
            .unwrap()
            .get(&fused_softcap_logits)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let reference_softcap_gradient = reference_softcap_loss
            .backward()
            .unwrap()
            .get(&reference_softcap_logits)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let softcap_gradient_difference = fused_softcap_gradient
            .iter()
            .flatten()
            .zip(reference_softcap_gradient.iter().flatten())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            softcap_gradient_difference < 1e-5,
            "softcap gradient difference={softcap_gradient_difference}"
        );
    }

    #[test]
    fn cpu_scalar_division_supports_strided_views() {
        let input = Tensor::from_vec(
            vec![3.0f32, 6.0, 9.0, 12.0, 15.0, 18.0],
            (2, 3),
            &Device::Cpu,
        )
        .unwrap()
        .narrow(1, 1, 2)
        .unwrap();
        let output = input.apply_op1(CpuDivideScalar(3.0)).unwrap();
        assert_eq!(
            output.to_vec2::<f32>().unwrap(),
            vec![vec![2.0, 3.0], vec![5.0, 6.0]]
        );
    }

    #[test]
    fn rotary_tables_follow_pytorch_float32_frequencies() {
        let (cos, sin) = rope_tables(1, 2047, 32, &Device::Cpu, DType::F32).unwrap();
        let cos = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sin = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (index, expected_cos, expected_sin) in [
            (1, -0.59164417, -0.8061992),
            (3, -0.7217759, -0.69212687),
            (7, 0.7475223, 0.6642367),
            (11, 0.73338294, 0.67981577),
            (15, 0.9991166, 0.04202328),
        ] {
            assert!((cos[index] - expected_cos).abs() < 1e-6);
            assert!((sin[index] - expected_sin).abs() < 1e-6);
        }
    }

    #[test]
    fn bf16_rotary_tables_track_cuda_reference() {
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cuda-rope-bf16.json")).unwrap();
        for (row, position) in reference["positions"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let position = position.as_u64().unwrap() as usize;
            let (cos, sin) = rope_tables(1, position, 128, &Device::Cpu, DType::BF16).unwrap();
            for (name, tensor) in [("cos", cos), ("sin", sin)] {
                let actual = tensor
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let expected = reference[name][row].as_array().unwrap();
                let expected = expected
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "position {position} {name}");
            }
        }
    }
    use candle_core::{DType, Device, Tensor, Var};

    #[cfg(feature = "cuda-base")]
    #[test]
    fn cuda_bf16_value_gate_tracks_checkpoint_reference() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cuda-value-gate-bf16.json"))
                .unwrap();
        assert_eq!(fixture["backend"], "PyTorch CUDA BF16 value gate");
        let config = crate::model_config::GptConfig::from_value(fixture["config"].clone()).unwrap();
        let time = fixture["time"].as_u64().unwrap() as usize;
        let read = |name: &str| {
            fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let shape = (1, time, config.n_kv_head, config.n_embd / config.n_head);
        let tensor = |name: &str| {
            Tensor::from_vec(read(name), shape, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let logits = Tensor::from_vec(read("gate_logits"), (1, time, config.n_kv_head), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let actual =
            super::mix_value_embeddings(&tensor("projection"), &tensor("value_embedding"), &logits)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
        let expected = read("output");
        assert_eq!(actual.len(), expected.len());
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max <= 0.015625, "value gate max error {max}");
    }

    #[cfg(feature = "cuda-base")]
    #[test]
    fn cuda_bf16_qk_scale_uses_pytorch_scalar_precision() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let x = Tensor::from_vec(
            vec![0.1f32, -0.1, 0.5, -1.0, 1.5, 3.25, -2.5, 0.003],
            (8,),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let actual = super::scale_qk(&x, DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // PyTorch CUDA BF16 `x * 1.2`, with the input converted to BF16 first.
        assert_eq!(
            actual,
            [
                0.12011719,
                -0.12011719,
                0.6015625,
                -1.203125,
                1.796875,
                3.90625,
                -3.0,
                0.0036010742
            ]
        );
    }

    #[cfg(feature = "cuda-base")]
    #[test]
    fn cuda_bf16_linear_prefill_and_decode_track_pytorch() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cuda-linear-bf16.json")).unwrap();
        assert_eq!(fixture["backend"], "PyTorch CUDA BF16 linear");
        let width = 768usize;
        let make_value = |i: usize, salt: usize| {
            let sign = (((i * 13 + salt) / 7) % 2) << 15;
            let exponent = (121 + (i * 17 + salt) % 8) << 7;
            let mantissa = (i * 53 + salt * 19) % 128;
            f32::from_bits(((sign | exponent | mantissa) as u32) << 16)
        };
        let weights = (0..width * width)
            .map(|i| make_value(i, 3))
            .collect::<Vec<_>>();
        let inputs = (0..13 * width)
            .map(|i| make_value(i, 5))
            .collect::<Vec<_>>();
        let layer = candle_nn::Linear::new(
            Tensor::from_vec(weights, (width, width), &device).unwrap(),
            None,
        );
        let x = Tensor::from_vec(inputs, (13, width), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        for (name, input) in [
            ("prefill", x.clone()),
            ("decode", x.narrow(0, 12, 1).unwrap()),
        ] {
            let actual = super::linear_forward(&layer, &input, DType::BF16, false)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let expected = fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), expected.len());
            let differing = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("CUDA BF16 linear {name}: {differing} different, max={max_error}");
            assert!(max_error <= 0.125, "{name}: max error {max_error}");
        }
    }

    #[cfg(feature = "cuda-base")]
    #[test]
    fn cuda_bf16_rms_norm_tracks_pytorch() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cuda-rms-bf16.json")).unwrap();
        assert_eq!(fixture["backend"], "PyTorch CUDA RMSNorm");
        for case in fixture["cases"].as_array().unwrap() {
            let width = case["width"].as_u64().unwrap() as usize;
            let read = |name: &str| {
                case[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>()
            };
            let x = Tensor::from_vec(read("input"), (13, width), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let actual = rms_norm(&x, DType::BF16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let expected = read("output");
            assert_eq!(actual.len(), expected.len());
            let mut differing = 0;
            let mut max_error = 0f32;
            for (a, b) in actual.iter().zip(&expected) {
                let error = (a - b).abs();
                differing += usize::from(error != 0.0);
                max_error = max_error.max(error);
            }
            eprintln!("CUDA BF16 RMSNorm width {width}: {differing} different, max={max_error}");
            assert!(
                max_error <= 0.0078125,
                "width {width}: max error {max_error}"
            );
            let alpha = Tensor::ones((width,), DType::BF16, &device).unwrap();
            let fused = candle_nn::ops::rms_norm(&x, &alpha, f32::EPSILON)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let fused_differing = fused.iter().zip(&expected).filter(|(a, b)| a != b).count();
            let fused_max = fused
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "CUDA BF16 fused RMSNorm width {width}: {fused_differing} different, max={fused_max}"
            );
            assert!(
                fused_max <= 0.0078125,
                "fused width {width}: max error {fused_max}"
            );
        }
    }

    #[cfg(feature = "flash-attention")]
    #[test]
    fn cuda_cached_attention_tracks_checkpoint_activations() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/cuda-sdpa-checkpoint-decode-bf16.json"
        ))
        .unwrap();
        assert_eq!(fixture["backend"], "PyTorch CUDA SDPA");
        let config = crate::model_config::GptConfig::from_value(fixture["config"].clone()).unwrap();
        let time = fixture["time"].as_u64().unwrap() as usize;
        let query_time = fixture["query_time"].as_u64().unwrap() as usize;
        let head_dim = config.n_embd / config.n_head;
        let values = |name: &str| {
            fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let tensor = |name: &str, t: usize, heads: usize| {
            Tensor::from_vec(values(name), (1, t, heads, head_dim), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let actual = candle_flash_attn::flash_attn_windowed(
            &tensor("q", query_time, config.n_head),
            &tensor("k", time, config.n_kv_head),
            &tensor("v", time, config.n_kv_head),
            1.0 / (head_dim as f32).sqrt(),
            Some(((config.sequence_len + 511) / 512) * 128),
            Some(0),
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let expected = values("output");
        assert_eq!(actual.len(), expected.len());
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let differing = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        eprintln!("cached attention: differing={differing} max={max}");
        assert!(
            max <= 0.015625,
            "checkpoint cached attention max error {max}"
        );
    }

    #[cfg(feature = "flash-attention")]
    #[test]
    fn cuda_attention_tracks_checkpoint_activations() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/cuda-sdpa-checkpoint-bf16.json"
        ))
        .unwrap();
        assert_eq!(fixture["backend"], "PyTorch CUDA SDPA");
        let config = crate::model_config::GptConfig::from_value(fixture["config"].clone()).unwrap();
        let time = fixture["time"].as_u64().unwrap() as usize;
        let head_dim = config.n_embd / config.n_head;
        let values = |name: &str| {
            fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let tensor = |name: &str| {
            Tensor::from_vec(values(name), (1, time, config.n_head, head_dim), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let actual = candle_flash_attn::flash_attn_windowed(
            &tensor("q"),
            &tensor("k"),
            &tensor("v"),
            1.0 / (head_dim as f32).sqrt(),
            Some(((config.sequence_len + 511) / 512) * 128),
            Some(0),
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let expected = values("output");
        assert_eq!(actual.len(), expected.len());
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max <= 0.0078125, "checkpoint attention max error {max}");
    }

    #[cfg(feature = "flash-attention")]
    #[test]
    fn cuda_attention_kernels_track_pytorch_sdpa() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cuda-sdpa-bf16.json")).unwrap();
        assert_eq!(fixture["backend"], "PyTorch CUDA SDPA");
        let values = |name: &str| {
            fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let tensor = |name: &str| {
            Tensor::from_vec(values(name), (1, 6, 13, 128), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let (q, k, v) = (tensor("q"), tensor("k"), tensor("v"));
        for (name, query, time, start) in [
            ("prefill", q.clone(), 13, 0),
            ("decode", q.narrow(2, 12, 1).unwrap(), 1, 12),
        ] {
            let flash = candle_flash_attn::flash_attn_windowed(
                &query.transpose(1, 2).unwrap(),
                &k.transpose(1, 2).unwrap(),
                &v.transpose(1, 2).unwrap(),
                1.0 / (128f32).sqrt(),
                None,
                Some(0),
            )
            .unwrap()
            .transpose(1, 2)
            .unwrap();
            let fallback = fallback_attention(
                &query, &k, &v, 1, time, start, 0, 13, 16384, 1, 128, &device, false,
            )
            .unwrap()
            .reshape((1, time, 6, 128))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
            let expected = values(name);
            for (path, output) in [("flash", flash), ("fallback", fallback)] {
                let actual = output
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                assert_eq!(actual.len(), expected.len());
                let max_error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let rms = (actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| ((*a - *b) as f64).powi(2))
                    .sum::<f64>()
                    / actual.len() as f64)
                    .sqrt();
                eprintln!("CUDA {name} {path}: max={max_error}, rms={rms}");
                let tolerance = if path == "flash" { 0.0078125 } else { 0.02 };
                assert!(
                    max_error <= tolerance,
                    "{name} {path}: max error {max_error}"
                );
            }
        }
    }

    #[test]
    fn bfloat16_attention_stays_close_to_pytorch_cpu_fixture() {
        let q = Var::from_vec(
            vec![
                0.13f32, -0.72, 1.11, 0.31, 0.91, 0.27, -0.44, 0.58, -0.39, 0.63, 0.74, -0.22,
            ],
            (1, 1, 3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let k = Var::from_vec(
            vec![
                0.25f32, -0.16, 0.81, 0.45, 0.33, 0.72, -0.59, -0.77, 0.88, -0.42, 0.34, 0.56,
            ],
            (1, 1, 3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let v = Var::from_vec(
            vec![
                0.4f32, -0.9, 0.7, 0.2, -0.6, 0.8, 0.1, 0.35, 0.45, 0.21, -0.3, 0.92,
            ],
            (1, 1, 3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let weights = Tensor::from_vec(
            vec![
                0.6f32, -0.4, 0.3, 0.7, -0.1, 0.8, -0.9, 0.2, 0.55, 0.33, -0.23, 0.77,
            ],
            (1, 3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let out = fallback_attention(
            &q.to_dtype(DType::BF16).unwrap(),
            &k.to_dtype(DType::BF16).unwrap(),
            &v.to_dtype(DType::BF16).unwrap(),
            1,
            3,
            0,
            0,
            3,
            8,
            1,
            4,
            &Device::Cpu,
            true,
        )
        .unwrap();
        let gradients = out
            .to_dtype(DType::F32)
            .unwrap()
            .mul(&weights)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let expected_output = [
            0.400390625,
            -0.8984375,
            0.69921875,
            0.2001953125,
            -0.1279296875,
            -0.0015258789,
            0.3828125,
            0.279296875,
            0.06640625,
            -0.018188477,
            0.228515625,
            0.44140625,
        ];
        let expected_q = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.01965332,
            0.22167969,
            -0.35351563,
            -0.30859375,
            0.06298828,
            -0.047851563,
            -0.014038086,
            0.040771484,
        ];
        let expected_k = [
            -0.20019531,
            -0.11767578,
            0.053955078,
            -0.12988281,
            0.23828125,
            0.052978516,
            -0.12792969,
            0.15136719,
            -0.040283203,
            0.064941406,
            0.076171875,
            -0.022583008,
        ];
        let expected_v = [
            0.76953125,
            0.106933594,
            -0.21386719,
            1.09375,
            0.13769531,
            0.53515625,
            -0.5546875,
            0.37109375,
            0.14550781,
            0.08691406,
            -0.060791016,
            0.203125,
        ];
        let output = out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut output_max_error = 0.0f32;
        for (index, (actual, expected)) in output.iter().zip(expected_output).enumerate() {
            output_max_error = output_max_error.max((actual - expected).abs());
            assert!(
                (actual - expected).abs() < 0.001,
                "attention output {index}: Rust={actual}, PyTorch={expected}, abs_error={}",
                (actual - expected).abs()
            );
        }
        let mut gradient_max_error = 0.0f32;
        for (var, expected) in [(q, expected_q), (k, expected_k), (v, expected_v)] {
            let actual = gradients
                .get(var.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                gradient_max_error = gradient_max_error.max((actual - expected).abs());
                assert!((actual - expected).abs() < 0.0021);
            }
        }
        eprintln!(
            "BF16 CPU attention max errors: output={output_max_error}, gradient={gradient_max_error}"
        );
    }

    #[test]
    fn rms_norm_matches_pytorch_forward_and_backward_fixture() {
        let input = Var::from_vec(
            vec![1.25f32, -0.75, 0.125, 3.0, -2.0, 0.5, 1.0, 0.25],
            (2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let weights = Tensor::from_vec(
            vec![0.5f32, -1.0, 0.25, 2.0, 1.5, 0.1, -0.5, 0.75],
            (2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let output = rms_norm(input.as_tensor(), DType::F32).unwrap();
        let gradient = output
            .mul(&weights)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let actual_output = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual_gradient = gradient
            .get(input.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected_output = [
            0.7490059,
            -0.44940352,
            0.07490059,
            1.7976141,
            -1.7354437,
            0.43386093,
            0.86772186,
            0.21693046,
        ];
        let expected_gradient = [
            -0.19833428,
            -0.30044273,
            0.10000752,
            0.0033614635,
            0.23581624,
            0.35321385,
            0.09902236,
            0.7840122,
        ];
        for (actual, expected) in actual_output.iter().zip(expected_output) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in actual_gradient.iter().zip(expected_gradient) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn cpu_width128_rms_norm_gradient_tracks_pytorch() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cpu-scalar-gradients.json"))
                .unwrap();
        let values = |name: &str| {
            fixture["rms_norm"][name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let input = Var::from_vec(values("input"), (13, 128), &Device::Cpu).unwrap();
        let upstream = Tensor::from_vec(values("upstream"), (13, 128), &Device::Cpu).unwrap();
        let output = rms_norm(input.as_tensor(), DType::F32).unwrap();
        let gradient = output
            .mul(&upstream)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let actual_output = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual_gradient = gradient
            .get(input.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected_output = values("output");
        let expected_gradient = values("gradient");
        let output_max = actual_output
            .iter()
            .zip(&expected_output)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let gradient_max = actual_gradient
            .iter()
            .zip(&expected_gradient)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        eprintln!("CPU width-128 RMSNorm: output max={output_max}, gradient max={gradient_max}");
        assert_eq!(output_max, 0.0);
        assert_eq!(gradient_max, 0.0);
    }

    #[test]
    fn attention_softmax_backward_matches_analytic_derivative() {
        let scores = Var::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();
        let weights = attention_softmax(scores.as_tensor(), true).unwrap();
        let coefficients =
            Tensor::from_vec(vec![0.25f32, -0.5, 1.0], (1, 3), &Device::Cpu).unwrap();
        let loss = weights.mul(&coefficients).unwrap().sum_all().unwrap();
        let gradients = loss.backward().unwrap();
        let actual = gradients
            .get(scores.as_tensor())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let probabilities = weights.to_vec2::<f32>().unwrap();
        let mean = probabilities[0]
            .iter()
            .zip([0.25f32, -0.5, 1.0])
            .map(|(p, c)| p * c)
            .sum::<f32>();
        for (index, coefficient) in [0.25f32, -0.5, 1.0].iter().enumerate() {
            let expected = probabilities[0][index] * (coefficient - mean);
            assert!((actual[0][index] - expected).abs() < 1e-6);
        }
        let cached = attention_softmax(scores.as_tensor(), false)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (actual, cached) in probabilities[0].iter().zip(&cached[0]) {
            assert!((actual - cached).abs() < 1e-6);
        }
    }
}

#[cfg(test)]
mod checkpoint_state_tests {
    use super::{
        GptModel, block_forward, checkpoint_source_name, pytorch_cpu_f32_sum,
        validate_checkpoint_keys,
    };

    #[cfg(feature = "flash-attention")]
    #[test]
    fn cuda_rl_policy_forward_propagates_attention_gradients() {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            return;
        };
        let config = crate::model_config::GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let (model, variables) =
            GptModel::load_trainable_pth(config, checkpoint, &device, candle_core::DType::F32)
                .unwrap();
        // The initialization fixture has a zero attention output projection;
        // use a nonzero projection so a missing Q gradient cannot pass silently.
        let mut projection = vec![0f32; 128 * 128];
        for i in 0..128 {
            projection[i * 128 + i] = 0.1;
        }
        variables.data().lock().unwrap()["transformer.h.0.attn.c_proj.weight"]
            .set(&candle_core::Tensor::from_vec(projection, (128, 128), &device).unwrap())
            .unwrap();
        let special_tokens = crate::tokenizer::SPECIAL_TOKENS
            .iter()
            .enumerate()
            .map(|(i, text)| serde_json::json!({"text": text, "id": 119 + i}))
            .collect::<Vec<_>>();
        let tokenizer_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            tokenizer_file.path(),
            serde_json::to_vec(&serde_json::json!({
                "pattern": ".", "mergeable_ranks": [{"bytes": "YQ==", "id": 0}],
                "special_tokens": special_tokens
            }))
            .unwrap(),
        )
        .unwrap();
        let tokenizer = crate::tokenizer::BpeTokenizer::load(tokenizer_file.path()).unwrap();
        let engine = crate::engine::Engine::new(
            model.with_compute_dtype(candle_core::DType::BF16),
            tokenizer,
        );
        let logits = engine
            .forward_train(&(0..16).collect::<Vec<u32>>(), 1, 16)
            .unwrap();
        let targets =
            candle_core::Tensor::from_vec((1..=16).collect::<Vec<u32>>(), (16,), &device).unwrap();
        let loss =
            candle_nn::loss::cross_entropy(&logits.reshape((16, 128)).unwrap(), &targets).unwrap();
        let gradients = loss.backward().unwrap();
        let variables = variables.data().lock().unwrap();
        let gradient = gradients
            .get(variables["transformer.h.0.attn.c_q.weight"].as_tensor())
            .expect("policy forward must retain attention gradients")
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            gradient.is_finite() && gradient > 0.0,
            "attention gradient max={gradient}"
        );
    }

    #[cfg(feature = "flash-attention")]
    #[test]
    #[ignore = "requires a local checkpoint and Python cached-decode trace"]
    fn diagnose_cuda_checkpoint_decode_parity() {
        use candle_core::{DType, Device, Tensor};
        let trace_path = std::env::var("RUST_AGENT_DECODE_TRACE").expect("RUST_AGENT_DECODE_TRACE");
        let checkpoint = std::env::var("RUST_AGENT_CHECKPOINT").expect("RUST_AGENT_CHECKPOINT");
        let trace: serde_json::Value =
            serde_json::from_slice(&std::fs::read(trace_path).unwrap()).unwrap();
        let config = crate::model_config::GptConfig::from_value(trace["config"].clone()).unwrap();
        let pos = trace["position"].as_u64().unwrap() as usize;
        let device = Device::new_cuda(0).unwrap();
        let model = GptModel::load_checkpoint(config.clone(), checkpoint, &device, DType::F32)
            .unwrap()
            .with_compute_dtype(DType::BF16)
            .with_inference_attention(true);
        let read = |value: &serde_json::Value| {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let make_cache = || {
            let mut cache = super::KvCache::new(&config);
            cache.position = pos;
            let shape = (1, pos, config.n_kv_head, config.n_embd / config.n_head);
            for layer in 0..config.n_layer {
                let load = |name: &str| {
                    Tensor::from_vec(read(&trace[name][layer]), shape, &device)
                        .unwrap()
                        .to_dtype(DType::BF16)
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap()
                };
                cache.keys[layer] = Some(load("keys"));
                cache.values[layer] = Some(load("values"));
            }
            cache.previous_embedding = Some(
                Tensor::from_vec(
                    read(&trace["previous_embedding"]),
                    (1, 1, config.n_embd),
                    &device,
                )
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
            );
            cache
        };
        let compare = |name: &str, output: Tensor, expected: Vec<f32>| {
            let actual = output
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(actual.len(), expected.len());
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let rms = (actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum::<f64>()
                / actual.len() as f64)
                .sqrt();
            let different = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            eprintln!("decode {name}: differing={different} max={max} rms={rms}");
            assert!(max.is_finite());
        };
        let mut cache = make_cache();
        for (layer, block) in model.blocks.iter().enumerate() {
            let record = &trace["blocks"][layer];
            let x = Tensor::from_vec(read(&record["input"]), (1, 1, config.n_embd), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let value = if record["value_embedding"].is_null() {
                None
            } else {
                Some(
                    Tensor::from_vec(
                        read(&record["value_embedding"]),
                        (1, 1, config.n_kv_head, config.n_embd / config.n_head),
                        &device,
                    )
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap(),
                )
            };
            let output = super::block_forward(
                block,
                &x,
                value.as_ref(),
                &config,
                layer,
                DType::BF16,
                false,
                true,
                pos,
                Some(&mut cache),
            )
            .unwrap();
            compare(&format!("block {layer}"), output, read(&record["output"]));
        }
        let layer = trace["detail_layer"].as_u64().unwrap() as usize;
        let detail = &trace["details"];
        let block = &model.blocks[layer];
        let input = Tensor::from_vec(
            read(&trace["blocks"][layer]["input"]),
            (1, 1, config.n_embd),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        compare(
            "initial norm",
            super::rms_norm(&input, DType::BF16).unwrap(),
            read(&detail["q"]["input"]),
        );
        for (name, linear) in [
            ("q", &block.attn.q),
            ("k", &block.attn.k),
            ("v", &block.attn.v),
            ("attn_proj", &block.attn.proj),
            ("mlp_fc", &block.mlp.fc),
            ("mlp_proj", &block.mlp.proj),
        ] {
            let values = read(&detail[name]["input"]);
            let width = values.len();
            let input = Tensor::from_vec(values, (1, 1, width), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            compare(
                name,
                super::linear_forward(linear, &input, DType::BF16, false).unwrap(),
                read(&detail[name]["output"]),
            );
        }
        let head_dim = config.n_embd / config.n_head;
        let (cos, sin) = super::rope_tables(1, pos, head_dim, &device, DType::BF16).unwrap();
        for name in ["q", "k"] {
            let heads = if name == "q" {
                config.n_head
            } else {
                config.n_kv_head
            };
            let projection = Tensor::from_vec(
                read(&detail[name]["output"]),
                (1, 1, heads, head_dim),
                &device,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
            let rotated = super::apply_rope(&projection, &cos, &sin).unwrap();
            compare(
                &format!("{name} rotary"),
                rotated.clone(),
                read(&detail[format!("{name}_rotary")]),
            );
            let normalized = super::rms_norm(&rotated, DType::BF16).unwrap();
            compare(
                &format!("{name} norm"),
                normalized.clone(),
                read(&detail[format!("{name}_norm")]),
            );
            let alpha = Tensor::ones((head_dim,), DType::BF16, &device).unwrap();
            let fused =
                candle_nn::ops::rms_norm(&rotated.contiguous().unwrap(), &alpha, f32::EPSILON)
                    .unwrap();
            compare(
                &format!("{name} fused norm"),
                fused,
                read(&detail[format!("{name}_norm")]),
            );
            let result = super::scale_qk(&normalized, DType::BF16).unwrap();
            let expected = read(&trace["attention"][name]);
            let expected = if name == "k" {
                expected[expected.len() - heads * head_dim..].to_vec()
            } else {
                expected
            };
            compare(&format!("{name} rotary and norm"), result, expected);
        }
        let hidden = read(&detail["mlp_fc"]["output"]);
        let width = hidden.len();
        let hidden = Tensor::from_vec(hidden, (1, 1, width), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        compare(
            "relu squared",
            hidden.relu().unwrap().sqr().unwrap(),
            read(&detail["mlp_proj"]["input"]),
        );
        let mut cache = make_cache();
        let token = trace["token"].as_u64().unwrap() as u32;
        let output = model
            .forward_with_cache(&[token], 1, 1, Some(&mut cache))
            .unwrap();
        compare("logits with Python cache", output, read(&trace["logits"]));
    }

    #[cfg(feature = "flash-attention")]
    #[test]
    #[ignore = "requires a local checkpoint and Python block trace"]
    fn diagnose_cuda_checkpoint_block_parity() {
        let trace_path = std::env::var("RUST_AGENT_BLOCK_TRACE").expect("RUST_AGENT_BLOCK_TRACE");
        let checkpoint = std::env::var("RUST_AGENT_CHECKPOINT").expect("RUST_AGENT_CHECKPOINT");
        let trace: serde_json::Value =
            serde_json::from_slice(&std::fs::read(trace_path).unwrap()).unwrap();
        let config = crate::model_config::GptConfig::from_value(trace["config"].clone()).unwrap();
        let time = trace["tokens"].as_array().unwrap().len();
        let device = candle_core::Device::new_cuda(0).unwrap();
        let model =
            GptModel::load_checkpoint(config.clone(), checkpoint, &device, candle_core::DType::F32)
                .unwrap()
                .with_compute_dtype(candle_core::DType::BF16)
                .with_inference_attention(true);
        for (layer, block) in model.blocks.iter().enumerate() {
            let record = &trace["blocks"][layer];
            let read = |key: &str| {
                record[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>()
            };
            let input =
                candle_core::Tensor::from_vec(read("input"), (1, time, config.n_embd), &device)
                    .unwrap()
                    .to_dtype(candle_core::DType::BF16)
                    .unwrap();
            let value = if record["value_embedding"].is_null() {
                None
            } else {
                Some(
                    candle_core::Tensor::from_vec(
                        read("value_embedding"),
                        (1, time, config.n_kv_head, config.n_embd / config.n_head),
                        &device,
                    )
                    .unwrap()
                    .to_dtype(candle_core::DType::BF16)
                    .unwrap(),
                )
            };
            let actual = super::block_forward(
                block,
                &input,
                value.as_ref(),
                &config,
                layer,
                candle_core::DType::BF16,
                false,
                true,
                0,
                None,
            )
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            let expected = read("output");
            assert_eq!(actual.len(), expected.len());
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let rms = (actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum::<f64>()
                / actual.len() as f64)
                .sqrt();
            let different = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            eprintln!("block {layer}: differing={different} max={max} rms={rms}");
            assert_eq!(max, 0.0, "block {layer} differs from Python");
        }
        let detail = &trace["first_block"];
        let detail_layer = trace["detail_layer"].as_u64().unwrap_or(0) as usize;
        let read = |record: &serde_json::Value, key: &str| {
            record[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let compare = |name: &str, result: candle_core::Tensor, expected: Vec<f32>| {
            let actual = result
                .to_dtype(candle_core::DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(actual.len(), expected.len());
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let different = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            eprintln!("block {detail_layer} {name}: differing={different} max={max}");
            max
        };
        let token_ids = trace["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let token_tensor =
            candle_core::Tensor::from_vec(token_ids.clone(), (time,), &device).unwrap();
        let embedded = candle_nn::Module::forward(&model.token_embedding, &token_tensor)
            .unwrap()
            .to_dtype(candle_core::DType::BF16)
            .unwrap()
            .reshape((1, time, config.n_embd))
            .unwrap();
        let normalized = super::rms_norm(&embedded, candle_core::DType::BF16).unwrap();
        let current = normalized.narrow(1, 1, time - 1).unwrap();
        let previous = normalized.narrow(1, 0, time - 1).unwrap();
        let gate = candle_nn::ops::sigmoid(
            &super::linear_forward(
                &model.smear_gate,
                &current.narrow(2, 0, 24).unwrap(),
                candle_core::DType::BF16,
                false,
            )
            .unwrap(),
        )
        .unwrap()
        .broadcast_mul(
            &model
                .smear_lambda
                .to_dtype(candle_core::DType::BF16)
                .unwrap(),
        )
        .unwrap();
        let smeared = current
            .broadcast_add(&gate.broadcast_mul(&previous).unwrap())
            .unwrap();
        let x0 =
            candle_core::Tensor::cat(&[normalized.narrow(1, 0, 1).unwrap(), smeared], 1).unwrap();
        let reference_x0 = trace["x0"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        compare("embedding and smear", x0, reference_x0.clone());
        let x0 =
            candle_core::Tensor::from_vec(reference_x0.clone(), (1, time, config.n_embd), &device)
                .unwrap()
                .to_dtype(candle_core::DType::BF16)
                .unwrap();
        for layer in 0..config.n_layer {
            let previous = if layer == 0 {
                reference_x0.clone()
            } else {
                read(&trace["blocks"][layer - 1], "output")
            };
            let previous =
                candle_core::Tensor::from_vec(previous, (1, time, config.n_embd), &device)
                    .unwrap()
                    .to_dtype(candle_core::DType::BF16)
                    .unwrap();
            let resid = model
                .resid_lambdas
                .narrow(0, layer, 1)
                .unwrap()
                .to_dtype(candle_core::DType::BF16)
                .unwrap()
                .reshape((1, 1, 1))
                .unwrap();
            let initial = model
                .x0_lambdas
                .narrow(0, layer, 1)
                .unwrap()
                .to_dtype(candle_core::DType::BF16)
                .unwrap()
                .reshape((1, 1, 1))
                .unwrap();
            let mixed = previous
                .broadcast_mul(&resid)
                .unwrap()
                .broadcast_add(&x0.broadcast_mul(&initial).unwrap())
                .unwrap();
            compare(
                &format!("layer {layer} residual mix"),
                mixed,
                read(&trace["blocks"][layer], "input"),
            );
        }
        let input = candle_core::Tensor::from_vec(
            read(&trace["blocks"][detail_layer], "input"),
            (1, time, config.n_embd),
            &device,
        )
        .unwrap()
        .to_dtype(candle_core::DType::BF16)
        .unwrap();
        compare(
            "initial norm",
            super::rms_norm(&input, candle_core::DType::BF16).unwrap(),
            read(&detail["q"], "input"),
        );
        let q_alpha =
            candle_core::Tensor::ones((config.n_embd,), candle_core::DType::BF16, &device).unwrap();
        compare(
            "initial fused norm",
            candle_nn::ops::rms_norm(&input.contiguous().unwrap(), &q_alpha, f32::EPSILON).unwrap(),
            read(&detail["q"], "input"),
        );
        let block = &model.blocks[detail_layer];
        for (name, linear) in [
            ("q", &block.attn.q),
            ("k", &block.attn.k),
            ("v", &block.attn.v),
            ("attn_proj", &block.attn.proj),
            ("mlp_fc", &block.mlp.fc),
            ("mlp_proj", &block.mlp.proj),
        ] {
            let record = &detail[name];
            let values = read(record, "input");
            let width = values.len() / time;
            let input = candle_core::Tensor::from_vec(values, (1, time, width), &device)
                .unwrap()
                .to_dtype(candle_core::DType::BF16)
                .unwrap();
            compare(
                name,
                super::linear_forward(linear, &input, candle_core::DType::BF16, false).unwrap(),
                read(record, "output"),
            );
        }
        if let Some(gate_layer) = &block.attn.ve_gate {
            let record = &detail["ve_gate"];
            let gate_input =
                candle_core::Tensor::from_vec(read(record, "input"), (1, time, 12), &device)
                    .unwrap()
                    .to_dtype(candle_core::DType::BF16)
                    .unwrap();
            let gate_logits =
                super::linear_forward(gate_layer, &gate_input, candle_core::DType::BF16, false)
                    .unwrap();
            compare(
                "value gate projection",
                gate_logits.clone(),
                read(record, "output"),
            );
            let shape = (1, time, config.n_kv_head, config.n_embd / config.n_head);
            let ve = candle_core::Tensor::from_vec(
                read(&trace["blocks"][detail_layer], "value_embedding"),
                shape,
                &device,
            )
            .unwrap()
            .to_dtype(candle_core::DType::BF16)
            .unwrap();
            let v = candle_core::Tensor::from_vec(read(&detail["v"], "output"), shape, &device)
                .unwrap()
                .to_dtype(candle_core::DType::BF16)
                .unwrap();
            compare(
                "mixed value",
                super::mix_value_embeddings(&v, &ve, &gate_logits).unwrap(),
                read(&detail["attention"], "v"),
            );
        }
        let head_dim = config.n_embd / config.n_head;
        let (cos, sin) =
            super::rope_tables(time, 0, head_dim, &device, candle_core::DType::BF16).unwrap();
        for name in ["q", "k"] {
            let heads = if name == "q" {
                config.n_head
            } else {
                config.n_kv_head
            };
            let projected = candle_core::Tensor::from_vec(
                read(&detail[name], "output"),
                (1, time, heads, head_dim),
                &device,
            )
            .unwrap()
            .to_dtype(candle_core::DType::BF16)
            .unwrap();
            let rotated = super::apply_rope(&projected, &cos, &sin).unwrap();
            let normalized = super::scale_qk(
                &super::rms_norm(&rotated, candle_core::DType::BF16).unwrap(),
                candle_core::DType::BF16,
            )
            .unwrap();
            compare(
                &format!("{name} rotary and norm"),
                normalized.clone(),
                read(&detail["attention"], name),
            );
            let alpha =
                candle_core::Tensor::ones((head_dim,), candle_core::DType::BF16, &device).unwrap();
            let fused = super::scale_qk(
                &candle_nn::ops::rms_norm(&rotated.contiguous().unwrap(), &alpha, f32::EPSILON)
                    .unwrap(),
                candle_core::DType::BF16,
            )
            .unwrap();
            compare(
                &format!("{name} fused rotary and norm"),
                fused,
                read(&detail["attention"], name),
            );
        }
        let attention_tensor = |name: &str| {
            let heads = if name == "q" {
                config.n_head
            } else {
                config.n_kv_head
            };
            candle_core::Tensor::from_vec(
                read(&detail["attention"], name),
                (1, time, heads, head_dim),
                &device,
            )
            .unwrap()
            .to_dtype(candle_core::DType::BF16)
            .unwrap()
        };
        let attended = candle_flash_attn::flash_attn_windowed(
            &attention_tensor("q"),
            &attention_tensor("k"),
            &attention_tensor("v"),
            1.0 / (head_dim as f32).sqrt(),
            (!config.is_long_window_layer(detail_layer))
                .then_some(((config.sequence_len + 511) / 512) * 128),
            Some(0),
        )
        .unwrap();
        compare("attention", attended, read(&detail["attention"], "output"));
        let fallback = super::fallback_attention(
            &attention_tensor("q").transpose(1, 2).unwrap(),
            &attention_tensor("k").transpose(1, 2).unwrap(),
            &attention_tensor("v").transpose(1, 2).unwrap(),
            1,
            time,
            0,
            0,
            time,
            config.sequence_len,
            config.n_head / config.n_kv_head,
            head_dim,
            &device,
            false,
        )
        .unwrap()
        .reshape((1, time, config.n_head, head_dim))
        .unwrap();
        compare(
            "fallback attention",
            fallback,
            read(&detail["attention"], "output"),
        );
        let input = candle_core::Tensor::from_vec(
            read(&trace["blocks"][detail_layer], "input"),
            (1, time, config.n_embd),
            &device,
        )
        .unwrap()
        .to_dtype(candle_core::DType::BF16)
        .unwrap();
        let projection = candle_core::Tensor::from_vec(
            read(&detail["attn_proj"], "output"),
            (1, time, config.n_embd),
            &device,
        )
        .unwrap()
        .to_dtype(candle_core::DType::BF16)
        .unwrap();
        let residual = input.broadcast_add(&projection).unwrap();
        compare(
            "mlp input norm",
            super::rms_norm(&residual, candle_core::DType::BF16).unwrap(),
            read(&detail["mlp_fc"], "input"),
        );
        let mlp_alpha =
            candle_core::Tensor::ones((config.n_embd,), candle_core::DType::BF16, &device).unwrap();
        compare(
            "mlp input fused norm",
            candle_nn::ops::rms_norm(&residual.contiguous().unwrap(), &mlp_alpha, f32::EPSILON)
                .unwrap(),
            read(&detail["mlp_fc"], "input"),
        );
        let mlp = candle_core::Tensor::from_vec(
            read(&detail["mlp_proj"], "output"),
            (1, time, config.n_embd),
            &device,
        )
        .unwrap()
        .to_dtype(candle_core::DType::BF16)
        .unwrap();
        compare(
            "residual output",
            residual.broadcast_add(&mlp).unwrap(),
            read(&trace["blocks"][detail_layer], "output"),
        );
        if let Some(expected_logits) = trace.get("logits") {
            let expected_logits = expected_logits
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            let logits = model.forward(&token_ids, 1, time).unwrap();
            let max = compare("full prefill logits", logits, expected_logits);
            assert_eq!(max, 0.0, "full prefill logits differ from Python");
        }
    }

    #[test]
    fn cpu_scalar_reduction_matches_pytorch_fp32_fixture() {
        if !super::has_pytorch_cpu_f32_reduction_layout() {
            return;
        }
        let values = (0..2048)
            .map(|index| (((index * 37 % 203) as i32 - 101) as f64 * 1e-8) as f32)
            .collect::<Vec<_>>();
        assert_eq!(pytorch_cpu_f32_sum(&values).to_bits(), 3_054_608_477);
        for (n, expected) in [
            (8, 3052333144u32),
            (12, 3053464973),
            (16, 3054784387),
            (24, 3059666218),
            (31, 3059974080),
            (32, 3058610687),
            (33, 3055620015),
            (64, 3064140564),
            (128, 3063392898),
            (129, 3064140565),
            (256, 3063304936),
            (512, 3064294498),
            (1024, 3053036850),
        ] {
            assert_eq!(
                pytorch_cpu_f32_sum(&values[..n]).to_bits(),
                expected,
                "width {n}"
            );
        }
    }

    #[test]
    fn cpu_embedding_mean_square_matches_pytorch_checkpoint() {
        if !super::has_pytorch_cpu_f32_reduction_layout() {
            return;
        }
        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let (_, variables) =
            GptModel::load_trainable_pth(config, checkpoint, &Device::Cpu, DType::F32).unwrap();
        let variables = variables.data().lock().unwrap();
        let embedding = variables["transformer.wte.weight"]
            .as_tensor()
            .narrow(0, 0, 16)
            .unwrap();
        let input =
            candle_core::Var::from_tensor(&embedding.to_dtype(DType::F32).unwrap()).unwrap();
        let coeff = candle_core::Tensor::from_vec(
            (0..2048)
                .map(|i| ((i * 37 % 203) as i32 - 101) as f32 / 101.0)
                .collect::<Vec<_>>(),
            (16, 128),
            &Device::Cpu,
        )
        .unwrap();
        let output = super::rms_norm(input.as_tensor(), DType::F32).unwrap();
        let grads = output
            .mul(&coeff)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let actual_grad = grads
            .get(input.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected_grad: Vec<f32> =
            serde_json::from_str(include_str!("../tests/fixtures/cpu-rms-backward.json")).unwrap();
        let max_error = actual_grad
            .iter()
            .zip(&expected_grad)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 1.2e-7,
            "CPU RMS gradient max error {max_error}"
        );
        let actual = embedding
            .to_dtype(DType::F32)
            .unwrap()
            .sqr()
            .unwrap()
            .contiguous()
            .unwrap()
            .apply_op1(super::CpuMeanLast)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = [
            1057435119, 1058408541, 1061631349, 1057760977, 1061328329, 1059014108, 1062596718,
            1062942668, 1058160323, 1059319644, 1056443706, 1058457239, 1058031085, 1060994777,
            1060889721, 1059582149,
        ];
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected
        );
        let normalized = super::rms_norm(&embedding.to_dtype(DType::F32).unwrap(), DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let expected_first = [
            1046138538, 1045667043, 3219158193, 3216228449, 1065188591, 1067765815, 1061193327,
            3205408015, 1063229649, 1075835936, 3214993626, 1050093251, 3215535173, 3192231823,
            1057640910, 1064030519,
        ];
        assert_eq!(
            normalized
                .iter()
                .map(|row| row[0].to_bits())
                .collect::<Vec<_>>(),
            expected_first
        );
    }

    #[test]
    fn cpu_f32_scalar_gradient_and_adam_match_pytorch_fixture() {
        if !super::has_pytorch_cpu_f32_reduction_layout() {
            return;
        }
        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let (model, variables) =
            GptModel::load_trainable_pth(config, checkpoint, &Device::Cpu, DType::F32).unwrap();
        let logits = model
            .forward(&(0..16).collect::<Vec<u32>>(), 1, 16)
            .unwrap();
        let targets =
            candle_core::Tensor::from_vec((1..=16).collect::<Vec<u32>>(), (16,), &Device::Cpu)
                .unwrap();
        let loss =
            super::training_cross_entropy(&logits.reshape((16, 128)).unwrap(), &targets).unwrap();
        let gradients = loss.backward().unwrap();
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cpu-scalar-gradients.json"))
                .unwrap();
        let actual_logits = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected_logits = reference["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let logits_max = actual_logits
            .iter()
            .zip(&expected_logits)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "CPU scalar batch: loss={:?}, logits max error={logits_max}",
            loss.to_scalar::<f32>().unwrap()
        );
        assert_eq!(logits_max, 0.0);
        let variables = variables.data().lock().unwrap();
        for name in ["backout_lambda", "x0_lambdas", "resid_lambdas"] {
            let parameter = &variables[name];
            let gradient = gradients
                .get(parameter.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let expected = reference["scalars"][name]["gradients"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            eprintln!(
                "CPU scalar {name} gradients: Rust={:?} Python={:?} Rust bits={:?}",
                gradient,
                expected,
                gradient.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(gradient, expected, "CPU scalar {name} gradient mismatch");
        }
        let parameter = &variables["backout_lambda"];
        let mut optimizer = crate::optim::MuonAdamW::new(
            ["backout_lambda", "x0_lambdas", "resid_lambdas"]
                .into_iter()
                .map(|name| (name.to_owned(), variables[name].clone()))
                .collect(),
            128,
            crate::optim::MuonAdamWConfig {
                embedding_lr: 0.2,
                unembedding_lr: 0.004,
                matrix_lr: 0.02,
                scalar_lr: 0.5,
                weight_decay: 0.0,
            },
        )
        .unwrap();
        optimizer
            .step(&gradients, 1.0, 0.95, 1.0, DType::F32)
            .unwrap();
        let updated = parameter.as_tensor().to_vec1::<f32>().unwrap()[0];
        let expected_update = reference["scalars"]["backout_lambda"]["updated"][0]
            .as_f64()
            .unwrap() as f32;
        eprintln!("CPU scalar backout_lambda update: Rust={updated:?} Python={expected_update:?}");
        assert_eq!(
            updated, expected_update,
            "backout_lambda Adam update mismatch"
        );
        for name in ["x0_lambdas", "resid_lambdas"] {
            let actual = variables[name].as_tensor().to_vec1::<f32>().unwrap();
            let expected = reference["scalars"][name]["updated"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            eprintln!("CPU scalar {name} update: Rust={actual:?} Python={expected:?}");
            assert_eq!(actual, expected, "CPU scalar {name} Adam update mismatch");
        }
    }

    #[test]
    fn cpu_f32_block_backward_tracks_pytorch_fixture() {
        use candle_core::{DType, Device, Tensor};
        use candle_nn::Module;

        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let (model, _) =
            GptModel::load_trainable_pth(config.clone(), checkpoint, &Device::Cpu, DType::F32)
                .unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cpu-scalar-gradients.json"))
                .unwrap();
        let token_ids = Tensor::from_vec((0..16u32).collect(), (16,), &Device::Cpu).unwrap();

        for layer in 0..config.n_layer {
            let read_tensor = |field: &str| {
                let values = fixture[field][layer]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>();
                Tensor::from_vec(values, (1, 16, 128), &Device::Cpu).unwrap()
            };
            let input = read_tensor("block_inputs");
            let upstream = read_tensor("block_output_gradients");
            let expected_input_gradient = read_tensor("mix_gradients");
            let value_embedding = model.value_embeddings[layer].as_ref().map(|embedding| {
                embedding
                    .forward(&token_ids)
                    .and_then(|value| value.to_dtype(DType::F32))
                    .and_then(|value| value.reshape((1, 16, 4, 32)))
                    .unwrap()
            });
            let output = block_forward(
                &model.blocks[layer],
                &input,
                value_embedding.as_ref(),
                &config,
                layer,
                DType::F32,
                false,
                false,
                0,
                None,
            )
            .unwrap();
            let expected_output = read_tensor("block_outputs");
            let output_error = output
                .sub(&expected_output)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let loss = output.broadcast_mul(&upstream).unwrap().sum_all().unwrap();
            let gradients = loss.backward().unwrap();
            let input_gradient = gradients.get(&input).unwrap();
            let gradient_error = input_gradient
                .sub(&expected_input_gradient)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            eprintln!(
                "CPU block {layer}: forward max error={output_error}, input-gradient max error={gradient_error}"
            );
            assert!(output_error <= 1e-5);
            assert!(gradient_error <= 1e-8);
        }
    }

    #[test]
    fn cpu_f32_final_head_backward_tracks_pytorch_fixture() {
        use candle_core::{DType, Device, Tensor, Var};

        if !super::has_pytorch_cpu_f32_reduction_layout() {
            return;
        }
        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let (model, _) =
            GptModel::load_trainable_pth(config.clone(), checkpoint, &Device::Cpu, DType::F32)
                .unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cpu-scalar-gradients.json"))
                .unwrap();
        let read_values = |field: &str| {
            fixture[field][1]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let block_output =
            Tensor::from_vec(read_values("block_outputs"), (1, 16, 128), &Device::Cpu).unwrap();
        let block_output = Var::from_tensor(&block_output).unwrap();
        let backout_lambda = model.backout_lambda.reshape((1, 1, 1)).unwrap();
        let backout = block_output
            .as_tensor()
            .apply_op2(&backout_lambda, super::BackoutMul)
            .unwrap();
        let final_hidden = block_output.as_tensor().broadcast_sub(&backout).unwrap();
        let normalized = super::rms_norm_for(&final_hidden, DType::F32, false).unwrap();
        let logits = super::linear_forward(&model.lm_head, &normalized, DType::F32, false)
            .unwrap()
            .apply_op1(super::CpuDivideScalar(super::LOGIT_SOFTCAP as f32))
            .unwrap()
            .tanh()
            .unwrap()
            .affine(super::LOGIT_SOFTCAP, 0.0)
            .unwrap();
        let targets = Tensor::from_vec((1..=16u32).collect(), (16,), &Device::Cpu).unwrap();
        let loss =
            super::training_cross_entropy(&logits.reshape((16, 128)).unwrap(), &targets).unwrap();
        let gradients = loss.backward().unwrap();
        let output_gradient = gradients.get(block_output.as_tensor()).unwrap();
        let backout_gradient = gradients
            .get(&model.backout_lambda)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        eprintln!(
            "CPU fixed-upstream backout_lambda gradient: Rust={backout_gradient:?} Python={:?}",
            fixture["scalars"]["backout_lambda"]["gradients"]
        );
        let expected = Tensor::from_vec(
            read_values("block_output_gradients"),
            (1, 16, 128),
            &Device::Cpu,
        )
        .unwrap();
        let max_error = output_gradient
            .sub(&expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!("CPU final-head block-output gradient max error={max_error}");
        assert_eq!(max_error, 0.0);
    }

    #[test]
    fn cpu_f32_final_hidden_backward_tracks_pytorch_fixture() {
        use candle_core::{DType, Device, Tensor, Var};

        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let (model, _) =
            GptModel::load_trainable_pth(config, checkpoint, &Device::Cpu, DType::F32).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cpu-scalar-gradients.json"))
                .unwrap();
        let hidden_values = fixture["final_hidden"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let hidden = Tensor::from_vec(hidden_values, (1, 16, 128), &Device::Cpu).unwrap();
        let hidden = Var::from_tensor(&hidden).unwrap();
        let normalized = super::rms_norm_for(hidden.as_tensor(), DType::F32, false).unwrap();
        let logits = super::linear_forward(&model.lm_head, &normalized, DType::F32, false)
            .unwrap()
            .apply_op1(super::CpuDivideScalar(super::LOGIT_SOFTCAP as f32))
            .unwrap()
            .tanh()
            .unwrap()
            .affine(super::LOGIT_SOFTCAP, 0.0)
            .unwrap();
        let targets = Tensor::from_vec((1..=16u32).collect(), (16,), &Device::Cpu).unwrap();
        let loss =
            super::training_cross_entropy(&logits.reshape((16, 128)).unwrap(), &targets).unwrap();
        let gradients = loss.backward().unwrap();
        let hidden_gradient = gradients.get(hidden.as_tensor()).unwrap();
        let expected = Tensor::from_vec(
            fixture["final_hidden_gradient"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect(),
            (1, 16, 128),
            &Device::Cpu,
        )
        .unwrap();
        let max_error = hidden_gradient
            .sub(&expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!("CPU final-hidden gradient max error={max_error}");
        assert_eq!(max_error, 0.0);
    }

    #[test]
    fn cpu_f32_output_head_backward_stages_track_pytorch_fixture() {
        use candle_core::{DType, Device, Tensor, Var};

        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cpu-scalar-gradients.json"))
                .unwrap();
        let values = |key: &str| {
            fixture[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let targets = Tensor::from_vec((1..=16u32).collect(), (16,), &Device::Cpu).unwrap();
        let raw_logits = Tensor::from_vec(values("raw_logits"), (16, 128), &Device::Cpu).unwrap();
        let raw_logits = Var::from_tensor(&raw_logits).unwrap();
        let logits = raw_logits
            .as_tensor()
            .apply_op1(super::CpuDivideScalar(super::LOGIT_SOFTCAP as f32))
            .unwrap()
            .tanh()
            .unwrap()
            .affine(super::LOGIT_SOFTCAP, 0.0)
            .unwrap();
        let loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
        let gradients = loss.backward().unwrap();
        let actual_raw_gradient = gradients.get(raw_logits.as_tensor()).unwrap();
        let expected_raw_gradient =
            Tensor::from_vec(values("raw_logits_gradient"), (16, 128), &Device::Cpu).unwrap();
        let raw_error = actual_raw_gradient
            .sub(&expected_raw_gradient)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let (model, _) =
            GptModel::load_trainable_pth(config, checkpoint, &Device::Cpu, DType::F32).unwrap();
        let head_input =
            Tensor::from_vec(values("lm_head_input"), (1, 16, 128), &Device::Cpu).unwrap();
        let head_input = Var::from_tensor(&head_input).unwrap();
        let raw_logits =
            super::linear_forward(&model.lm_head, head_input.as_tensor(), DType::F32, false)
                .unwrap()
                .reshape((16, 128))
                .unwrap();
        let logits = raw_logits
            .apply_op1(super::CpuDivideScalar(super::LOGIT_SOFTCAP as f32))
            .unwrap()
            .tanh()
            .unwrap()
            .affine(super::LOGIT_SOFTCAP, 0.0)
            .unwrap();
        let loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
        let gradients = loss.backward().unwrap();
        let actual_input_gradient = gradients.get(head_input.as_tensor()).unwrap();
        let expected_input_gradient =
            Tensor::from_vec(values("lm_head_input_gradient"), (1, 16, 128), &Device::Cpu).unwrap();
        let input_error = actual_input_gradient
            .sub(&expected_input_gradient)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!("CPU softcap/raw-logit gradient max error={raw_error}");
        eprintln!("CPU lm_head input-gradient max error={input_error}");
    }

    #[test]
    fn cpu_f32_softcap_and_lm_head_backward_match_fixed_pytorch_upstream() {
        use candle_core::{DType, Device, Tensor, Var};

        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cpu-scalar-gradients.json"))
                .unwrap();
        let values = |key: &str| {
            fixture[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let upstream_after_softcap =
            Tensor::from_vec(values("logits_gradient"), (16, 128), &Device::Cpu).unwrap();
        let raw_logits = Tensor::from_vec(values("raw_logits"), (16, 128), &Device::Cpu).unwrap();
        let raw_logits = Var::from_tensor(&raw_logits).unwrap();
        let softcapped = raw_logits
            .as_tensor()
            .apply_op1(super::CpuDivideScalar(super::LOGIT_SOFTCAP as f32))
            .unwrap()
            .tanh()
            .unwrap()
            .affine(super::LOGIT_SOFTCAP, 0.0)
            .unwrap();
        let softcap_loss = softcapped
            .broadcast_mul(&upstream_after_softcap)
            .unwrap()
            .sum_all()
            .unwrap();
        let softcap_gradients = softcap_loss.backward().unwrap();
        let actual_raw_gradient = softcap_gradients.get(raw_logits.as_tensor()).unwrap();
        let expected_raw_gradient =
            Tensor::from_vec(values("raw_logits_gradient"), (16, 128), &Device::Cpu).unwrap();
        let softcap_error = actual_raw_gradient
            .sub(&expected_raw_gradient)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let expected_softcapped =
            Tensor::from_vec(values("logits"), (16, 128), &Device::Cpu).unwrap();
        let softcap_forward_error = softcapped
            .sub(&expected_softcapped)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let raw_logits = Tensor::from_vec(values("raw_logits"), (16, 128), &Device::Cpu).unwrap();
        let raw_logits = Var::from_tensor(&raw_logits).unwrap();
        let torch_tanh = raw_logits
            .as_tensor()
            .apply_op1(super::CpuDivideScalar(super::LOGIT_SOFTCAP as f32))
            .unwrap()
            .apply_op1(super::TorchTanh)
            .unwrap()
            .affine(super::LOGIT_SOFTCAP, 0.0)
            .unwrap();
        let torch_softcap_loss = torch_tanh
            .broadcast_mul(&upstream_after_softcap)
            .unwrap()
            .sum_all()
            .unwrap();
        let torch_softcap_gradients = torch_softcap_loss.backward().unwrap();
        let torch_raw_gradient = torch_softcap_gradients.get(raw_logits.as_tensor()).unwrap();
        let torch_softcap_error = torch_raw_gradient
            .sub(&expected_raw_gradient)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let torch_softcap_forward_error = torch_tanh
            .sub(&expected_softcapped)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let (model, _) =
            GptModel::load_trainable_pth(config, checkpoint, &Device::Cpu, DType::F32).unwrap();
        let head_input =
            Tensor::from_vec(values("lm_head_input"), (1, 16, 128), &Device::Cpu).unwrap();
        let head_input = Var::from_tensor(&head_input).unwrap();
        let raw_logits =
            super::linear_forward(&model.lm_head, head_input.as_tensor(), DType::F32, false)
                .unwrap()
                .reshape((16, 128))
                .unwrap();
        let fixed_upstream =
            Tensor::from_vec(values("raw_logits_gradient"), (16, 128), &Device::Cpu).unwrap();
        let linear_loss = raw_logits
            .broadcast_mul(&fixed_upstream)
            .unwrap()
            .sum_all()
            .unwrap();
        let linear_gradients = linear_loss.backward().unwrap();
        let actual_input_gradient = linear_gradients.get(head_input.as_tensor()).unwrap();
        let expected_input_gradient =
            Tensor::from_vec(values("lm_head_input_gradient"), (1, 16, 128), &Device::Cpu).unwrap();
        let linear_error = actual_input_gradient
            .sub(&expected_input_gradient)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!("CPU softcap backward max error={softcap_error}");
        eprintln!("CPU softcap forward max error={softcap_forward_error}");
        eprintln!("CPU explicit tanh backward max error={torch_softcap_error}");
        eprintln!("CPU explicit tanh forward max error={torch_softcap_forward_error}");
        eprintln!("CPU lm_head backward with fixed upstream max error={linear_error}");
    }

    #[test]
    fn cpu_f32_cross_entropy_logits_gradient_tracks_pytorch_fixture() {
        use candle_core::{Device, Tensor, Var};

        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cpu-scalar-gradients.json"))
                .unwrap();
        let values = |key: &str| {
            fixture[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let logits = Tensor::from_vec(values("logits"), (16, 128), &Device::Cpu).unwrap();
        let logits = Var::from_tensor(&logits).unwrap();
        let candle_log_probs = candle_nn::ops::log_softmax(logits.as_tensor(), 1).unwrap();
        let expected_log_probs =
            Tensor::from_vec(values("log_probs"), (16, 128), &Device::Cpu).unwrap();
        let log_probs_error = candle_log_probs
            .sub(&expected_log_probs)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!("CPU log-softmax max error={log_probs_error}");
        let candle_probabilities = candle_log_probs.exp().unwrap();
        let expected_probabilities =
            Tensor::from_vec(values("log_probs_exp"), (16, 128), &Device::Cpu).unwrap();
        let probabilities_error = candle_probabilities
            .sub(&expected_probabilities)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!("CPU exp(log-softmax) max error={probabilities_error}");
        let targets = Tensor::from_vec((1..=16u32).collect(), (16,), &Device::Cpu).unwrap();
        let loss = candle_nn::loss::cross_entropy(logits.as_tensor(), &targets).unwrap();
        let gradients = loss.backward().unwrap();
        let actual = gradients.get(logits.as_tensor()).unwrap();
        let expected =
            Tensor::from_vec(values("logits_gradient"), (16, 128), &Device::Cpu).unwrap();
        let max_error = actual
            .sub(&expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!("CPU cross-entropy logits-gradient max error={max_error}");
        assert!(max_error <= 1e-8);
    }

    #[test]
    fn cpu_f32_fused_cross_entropy_backward_matches_pytorch_fixture() {
        use candle_core::{Device, Tensor, Var};

        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cpu-scalar-gradients.json"))
                .unwrap();
        let values = |key: &str| {
            fixture[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let logits = Tensor::from_vec(values("logits"), (16, 128), &Device::Cpu).unwrap();
        let logits = Var::from_tensor(&logits).unwrap();
        let targets = fixture["targets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect();
        let targets = Tensor::from_vec(targets, (16,), &Device::Cpu).unwrap();
        let loss = super::training_cross_entropy(logits.as_tensor(), &targets).unwrap();
        let gradients = loss.backward().unwrap();
        let actual = gradients.get(logits.as_tensor()).unwrap();
        let expected =
            Tensor::from_vec(values("logits_gradient"), (16, 128), &Device::Cpu).unwrap();
        let max_error = actual
            .sub(&expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!("CPU fused cross-entropy logits-gradient max error={max_error}");
        assert_eq!(max_error, 0.0);
    }
    #[cfg(feature = "cuda-base")]
    use crate::optim::{MuonAdamW, MuonAdamWConfig};
    use crate::{model_config::GptConfig, torch_save::TorchValue};
    use candle_core::{DType, Device};
    use candle_nn::VarMap;
    use std::collections::{BTreeMap, HashSet};

    fn fixture_config() -> GptConfig {
        GptConfig {
            sequence_len: 16,
            vocab_size: 64,
            n_layer: 1,
            n_head: 4,
            n_kv_head: 2,
            n_embd: 32,
            window_pattern: "L".into(),
        }
    }

    #[test]
    fn cuda_fresh_initialization_matches_pytorch_parameter_fixture() {
        // Reference values come from nanochat GPT.init_weights() with CUDA
        // seed 42 and NANOCHAT_DTYPE=float32 for this two-layer config.
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 64,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 2,
            n_embd: 32,
            window_pattern: "L".into(),
        };
        let (_model, variables) =
            GptModel::new_trainable_with_compute_dtype(config, &device, DType::F32).unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cuda-init-depth2-f32.json"))
                .unwrap();
        let variables = variables
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, variable)| (name.clone(), variable.clone()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(variables.len(), expected.as_object().unwrap().len());
        let mut max_abs_error = 0.0f64;
        let mut squared_error_sum = 0.0f64;
        let mut values_over_1e7 = 0usize;
        let mut values_over_1e6 = 0usize;
        let mut value_count = 0usize;
        for (name, variable) in &variables {
            let expected = &expected[name];
            let source = variable.as_tensor().to_dtype(DType::F32).unwrap();
            assert_eq!(
                source.dims(),
                expected["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|dim| dim.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>(),
                "shape mismatch for {name}"
            );
            let tensor = source.flatten_all().unwrap();
            let values = tensor.to_vec1::<f32>().unwrap();
            let reference = expected["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                values.len(),
                reference.len(),
                "value count mismatch for {name}"
            );
            for (actual, reference) in values.iter().zip(reference) {
                let error = (*actual as f64 - reference).abs();
                max_abs_error = max_abs_error.max(error);
                squared_error_sum += error * error;
                values_over_1e7 += usize::from(error > 1e-7);
                values_over_1e6 += usize::from(error > 1e-6);
                value_count += 1;
            }
        }
        let rms_error = (squared_error_sum / value_count as f64).sqrt();
        eprintln!(
            "PyTorch CUDA initialization fixture: values={value_count}, max_abs_error={max_abs_error}, rms_error={rms_error}, over_1e-7={values_over_1e7}, over_1e-6={values_over_1e6}"
        );
        assert!(
            max_abs_error <= 1e-5,
            "CUDA initialization max error: {max_abs_error}"
        );
    }

    #[cfg(feature = "cuda-base")]
    #[test]
    fn cuda_fp8_gpt_forward_and_backward_match_pytorch_fixture() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let (model, variables) =
            GptModel::load_trainable_pth(config, checkpoint, &device, DType::F32).unwrap();
        let model = model
            .with_compute_dtype(DType::BF16)
            .with_fp8_training(true);
        let input_ids = (0..16).collect::<Vec<u32>>();
        let target_ids = (1..=16).collect::<Vec<u32>>();
        let logits = model.forward(&input_ids, 1, 16).unwrap();
        let targets = candle_core::Tensor::from_vec(target_ids, (16,), &device).unwrap();
        let loss =
            candle_nn::loss::cross_entropy(&logits.reshape((16, 128)).unwrap(), &targets).unwrap();
        let gradients = loss.backward().unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/fp8-model-depth2-bf16.json"))
                .unwrap();
        let actual_logits = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected_logits = expected["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        assert_eq!(actual_logits.len(), expected_logits.len());
        let logits_max_error = actual_logits
            .iter()
            .zip(&expected_logits)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let expected_loss = expected["loss"].as_f64().unwrap() as f32;
        let actual_loss = loss.to_scalar::<f32>().unwrap();

        let variables = variables
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, variable)| (name.clone(), variable.clone()))
            .collect::<BTreeMap<_, _>>();
        let expected_grads = expected["grads"].as_object().unwrap();
        assert_eq!(variables.len(), expected_grads.len());
        let mut grads_max_error = 0.0f32;
        let mut grads_square_sum = 0.0f64;
        let mut grads_count = 0usize;
        for (name, variable) in &variables {
            let actual = gradients
                .get(variable.as_tensor())
                .unwrap_or_else(|| panic!("missing gradient for {name}"))
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let reference = expected_grads[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), reference.len(), "gradient shape for {name}");
            for (actual, reference) in actual.iter().zip(reference) {
                let error = (actual - reference).abs();
                grads_max_error = grads_max_error.max(error);
                grads_square_sum += (error as f64).powi(2);
                grads_count += 1;
            }
        }
        let grads_rms_error = (grads_square_sum / grads_count as f64).sqrt();
        eprintln!(
            "PyTorch FP8 GPT fixture: loss Rust={actual_loss} Python={expected_loss}; logits_max_abs={logits_max_error}; gradient_values={grads_count} max_abs={grads_max_error} rms={grads_rms_error}"
        );
        assert!((actual_loss - expected_loss).abs() <= 1e-5);
        assert!(logits_max_error <= 1e-4);
        assert!(grads_max_error <= 5e-5);
    }

    #[cfg(feature = "cuda-base")]
    #[test]
    fn cuda_fp8_gpt_optimizer_step_matches_pytorch_fixture() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let (model, variables) =
            GptModel::load_trainable_pth(config, checkpoint, &device, DType::F32).unwrap();
        let model = model
            .with_compute_dtype(DType::BF16)
            .with_fp8_training(true);
        let input_ids = (0..16).collect::<Vec<u32>>();
        let target_ids = (1..=16).collect::<Vec<u32>>();
        let logits = model.forward(&input_ids, 1, 16).unwrap();
        let targets = candle_core::Tensor::from_vec(target_ids, (16,), &device).unwrap();
        let loss =
            candle_nn::loss::cross_entropy(&logits.reshape((16, 128)).unwrap(), &targets).unwrap();
        let gradients = loss.backward().unwrap();
        let parameters = variables
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, variable)| (name.clone(), variable.clone()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(parameters["transformer.wte.weight"].dtype(), DType::BF16);
        assert_eq!(parameters["value_embeds.1.weight"].dtype(), DType::BF16);
        assert_eq!(parameters["lm_head.weight"].dtype(), DType::F32);
        let mut optimizer = MuonAdamW::new(
            parameters
                .iter()
                .map(|(name, variable)| (name.clone(), variable.clone()))
                .collect(),
            128,
            MuonAdamWConfig {
                embedding_lr: 0.2,
                unembedding_lr: 0.004,
                matrix_lr: 0.02,
                scalar_lr: 0.5,
                weight_decay: 0.0,
            },
        )
        .unwrap();
        optimizer
            .step(&gradients, 1.0, 0.95, 1.0, DType::BF16)
            .unwrap();

        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/fp8-model-depth2-step1.json"
        ))
        .unwrap();
        assert_eq!(
            expected["backend"],
            "PyTorch 2.9.1 CUDA tensorwise FP8 + MuonAdamW"
        );
        assert_eq!(expected["fp8_recipe"], "tensorwise");
        let expected_loss = expected["loss"].as_f64().unwrap() as f32;
        let actual_loss = loss.to_scalar::<f32>().unwrap();
        let expected_parameters = expected["updated_parameters"].as_object().unwrap();
        assert_eq!(parameters.len(), expected_parameters.len());
        let mut max_abs_error = 0.0f32;
        let mut squared_error_sum = 0.0f64;
        let mut values = 0usize;
        let mut largest_name = String::new();
        let mut parameter_errors = Vec::new();
        for (name, parameter) in &parameters {
            let actual = parameter
                .as_tensor()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let reference = expected_parameters[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), reference.len(), "parameter shape for {name}");
            let mut parameter_max_error = 0.0f32;
            for (actual, reference) in actual.iter().zip(reference) {
                let error = (actual - reference).abs();
                parameter_max_error = parameter_max_error.max(error);
                if error > max_abs_error {
                    max_abs_error = error;
                    largest_name = name.clone();
                }
                squared_error_sum += (error as f64).powi(2);
                values += 1;
            }
            parameter_errors.push((parameter_max_error, name));
        }
        let rms_error = (squared_error_sum / values as f64).sqrt();
        parameter_errors.sort_by(|left, right| right.0.total_cmp(&left.0));
        eprintln!(
            "PyTorch FP8 optimizer fixture: loss Rust={actual_loss} Python={expected_loss}; updated_parameters={values} max_abs={max_abs_error} rms={rms_error} largest={largest_name}; parameter_max_errors={:?}",
            &parameter_errors[..parameter_errors.len().min(8)]
        );
        assert!((actual_loss - expected_loss).abs() <= 1e-5);
        assert!(max_abs_error <= 2e-2);
        assert!(rms_error <= 1e-3);
    }

    #[cfg(feature = "cuda-base")]
    #[test]
    fn cuda_fp8_hundred_step_loss_tracks_pytorch() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let (model, variables) =
            GptModel::load_trainable_pth(config, checkpoint, &device, DType::F32).unwrap();
        let model = model
            .with_compute_dtype(DType::BF16)
            .with_fp8_training(true);
        let parameters = variables
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, variable)| (name.clone(), variable.clone()))
            .collect();
        let mut optimizer = MuonAdamW::new(
            parameters,
            128,
            MuonAdamWConfig {
                embedding_lr: 0.2,
                unembedding_lr: 0.004,
                matrix_lr: 0.02,
                scalar_lr: 0.5,
                weight_decay: 0.0,
            },
        )
        .unwrap();
        let inputs = (0..16).collect::<Vec<u32>>();
        let targets =
            candle_core::Tensor::from_vec((1..=16).collect::<Vec<u32>>(), (16,), &device).unwrap();
        // PyTorch 2.9.1 CUDA, tensorwise FP8, same checkpoint and batch each step.
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/fp8-model-depth2-100-step.json"
        ))
        .unwrap();
        let reference = reference["losses"].as_array().unwrap();
        let mut maximum_drift = 0.0f32;
        for (step, expected) in reference.iter().enumerate() {
            let expected = expected.as_f64().unwrap() as f32;
            let logits = model.forward(&inputs, 1, 16).unwrap();
            let loss =
                candle_nn::loss::cross_entropy(&logits.reshape((16, 128)).unwrap(), &targets)
                    .unwrap();
            let actual = loss.to_scalar::<f32>().unwrap();
            if step < 10 || step % 10 == 9 {
                eprintln!("FP8 step {step}: Rust={actual} Python={expected}");
            }
            maximum_drift = maximum_drift.max((actual - expected).abs());
            assert!(
                actual.is_finite() && (actual - expected).abs() < 0.02,
                "step {step}: Rust={actual} Python={expected}"
            );
            optimizer
                .step(&loss.backward().unwrap(), 1.0, 0.95, 1.0, DType::BF16)
                .unwrap();
        }
        eprintln!("100-step FP8 maximum loss drift: {maximum_drift}");
    }

    #[cfg(feature = "cuda-base")]
    #[test]
    #[ignore = "requires a local representative checkpoint and Python FP8 reference trajectory"]
    fn cuda_fp8_depth12_stability_tracks_pytorch() {
        use candle_core::Tensor;

        let checkpoint =
            std::env::var("RUST_AGENT_FP8_CHECKPOINT").expect("RUST_AGENT_FP8_CHECKPOINT");
        let meta_path = std::env::var("RUST_AGENT_FP8_META").expect("RUST_AGENT_FP8_META");
        let reference_path =
            std::env::var("RUST_AGENT_FP8_REFERENCE").expect("RUST_AGENT_FP8_REFERENCE");
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(meta_path).unwrap()).unwrap();
        let config = GptConfig::from_value(meta["model_config"].clone()).unwrap();
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
        assert_eq!(
            reference["steps"].as_u64().unwrap() as usize,
            reference["losses"].as_array().unwrap().len()
        );
        let device = Device::new_cuda(0).unwrap();
        let (model, variables) =
            GptModel::load_trainable_pth(config.clone(), checkpoint, &device, DType::F32).unwrap();
        let model = model
            .with_compute_dtype(DType::BF16)
            .with_fp8_training(true);
        let parameters = variables
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, variable)| (name.clone(), variable.clone()))
            .collect();
        let mut optimizer = MuonAdamW::new(
            parameters,
            config.n_embd,
            MuonAdamWConfig {
                embedding_lr: 0.3,
                unembedding_lr: 0.008,
                matrix_lr: 0.02,
                scalar_lr: 0.5,
                weight_decay: 0.28,
            },
        )
        .unwrap();
        let input_ids = reference["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|token| token.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let target_ids = reference["target_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|token| token.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let targets = Tensor::from_vec(target_ids, (input_ids.len(),), &device).unwrap();
        let expected = reference["losses"].as_array().unwrap();
        let mut maximum_drift = 0.0f32;
        for (step, expected) in expected.iter().enumerate() {
            let expected = expected.as_f64().unwrap() as f32;
            let logits = model.forward(&input_ids, 1, input_ids.len()).unwrap();
            let loss = candle_nn::loss::cross_entropy(
                &logits
                    .reshape((input_ids.len(), config.vocab_size))
                    .unwrap(),
                &targets,
            )
            .unwrap();
            let actual = loss.to_scalar::<f32>().unwrap();
            if step < 5 || step % 10 == 9 {
                eprintln!("FP8 representative Rust step {step}: loss={actual}, Python={expected}");
            }
            let drift = (actual - expected).abs();
            maximum_drift = maximum_drift.max(drift);
            assert!(
                actual.is_finite() && drift < 0.1,
                "step {step}: Rust={actual}, Python={expected}, drift={drift}"
            );
            optimizer
                .step(&loss.backward().unwrap(), 1.0, 0.95, 1.0, DType::BF16)
                .unwrap();
        }
        eprintln!("depth-12 100-step FP8 maximum loss drift: {maximum_drift}");
    }

    #[test]
    fn bf16_training_uses_pytorch_embedding_storage_dtypes() {
        let config = fixture_config();
        let (_model, variables) =
            GptModel::new_trainable_with_compute_dtype(config, &Device::Cpu, DType::BF16).unwrap();
        let variables = variables.data().lock().unwrap();
        assert_eq!(variables["transformer.wte.weight"].dtype(), DType::BF16);
        assert_eq!(variables["value_embeds.0.weight"].dtype(), DType::BF16);
        assert_eq!(
            variables["transformer.h.0.attn.c_q.weight"].dtype(),
            DType::F32
        );
        assert_eq!(variables["lm_head.weight"].dtype(), DType::F32);
    }

    #[test]
    fn cpu_bf16_gpt_forward_and_backward_match_pytorch_flash_fixture() {
        let device = Device::Cpu;
        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let (model, variables) =
            GptModel::load_trainable_pth(config, checkpoint, &device, DType::F32).unwrap();
        let model = model.with_compute_dtype(DType::BF16);
        let input_ids = (0..16).collect::<Vec<u32>>();
        let target_ids = (1..=16).collect::<Vec<u32>>();
        let logits = model.forward(&input_ids, 1, 16).unwrap();
        let targets = candle_core::Tensor::from_vec(target_ids, (16,), &device).unwrap();
        let loss =
            candle_nn::loss::cross_entropy(&logits.reshape((16, 128)).unwrap(), &targets).unwrap();
        let gradients = loss.backward().unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/bf16-model-depth2-cpu.json"))
                .unwrap();
        assert_eq!(expected["backend"], "PyTorch 2.9.1 CPU Flash SDPA");
        assert_eq!(expected["threads"], 1);
        assert_eq!(expected["source_checkpoint"], "fp8-model-depth2.pt");
        assert_eq!(expected["compute_dtype"], "bfloat16");

        let actual_logits = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected_logits = expected["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        assert_eq!(actual_logits.len(), expected_logits.len());
        let logits_max_error = actual_logits
            .iter()
            .zip(&expected_logits)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let expected_loss = expected["loss"].as_f64().unwrap() as f32;
        let actual_loss = loss.to_scalar::<f32>().unwrap();

        let variables = variables
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, variable)| (name.clone(), variable.clone()))
            .collect::<BTreeMap<_, _>>();
        let expected_grads = expected["grads"].as_object().unwrap();
        assert_eq!(variables.len(), expected_grads.len());
        let mut grads_max_error = 0.0f32;
        let mut grads_square_sum = 0.0f64;
        let mut grads_count = 0usize;
        for (name, variable) in &variables {
            let actual = gradients
                .get(variable.as_tensor())
                .unwrap_or_else(|| panic!("missing gradient for {name}"))
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let reference = expected_grads[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), reference.len(), "gradient shape for {name}");
            for (actual, reference) in actual.iter().zip(reference) {
                let error = (actual - reference).abs();
                grads_max_error = grads_max_error.max(error);
                grads_square_sum += (error as f64).powi(2);
                grads_count += 1;
            }
        }
        let grads_rms_error = (grads_square_sum / grads_count as f64).sqrt();
        eprintln!(
            "PyTorch CPU BF16 GPT fixture: loss Rust={actual_loss} Python={expected_loss}; logits_max_abs={logits_max_error}; gradient_values={grads_count} max_abs={grads_max_error} rms={grads_rms_error}"
        );
        assert!((actual_loss - expected_loss).abs() <= 1e-4);
        assert!(logits_max_error <= 1e-5);
        assert!(grads_max_error <= 5e-5);
    }

    #[cfg(feature = "flash-attention")]
    #[test]
    fn cuda_bf16_gpt_inference_matches_pytorch_sdpa_fixture() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let model = GptModel::load_checkpoint(config, checkpoint, &device, DType::F32).unwrap();
        let model = model
            .with_compute_dtype(DType::BF16)
            .with_inference_attention(true);
        let input_ids = (0..16).collect::<Vec<u32>>();
        let logits = model.forward(&input_ids, 1, 16).unwrap();
        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/bf16-model-depth2-cuda.json"
        ))
        .unwrap();
        assert_eq!(expected["backend"], "PyTorch 2.9.1 CUDA Flash SDPA");
        assert_eq!(expected["fa3_available"], false);
        assert_eq!(expected["compute_dtype"], "bfloat16");
        let actual = logits
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let reference = expected["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), reference.len());
        let max_abs_error = actual
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let rms_error = (actual
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| (actual - expected).powi(2) as f64)
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        let actual_top1 = actual
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0;
        let reference_top1 = reference
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0;
        eprintln!(
            "PyTorch CUDA BF16 GPT fixture: logits_max_abs={max_abs_error} rms={rms_error}; top1 Rust={actual_top1} Python={reference_top1}"
        );
        assert_eq!(actual_top1, reference_top1);
        assert!(max_abs_error <= 1e-2);
    }

    #[cfg(feature = "flash-attention")]
    #[test]
    fn cuda_bf16_cached_decode_matches_full_sequence_logits() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let config = GptConfig {
            sequence_len: 16,
            vocab_size: 128,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            n_embd: 128,
            window_pattern: "L".into(),
        };
        let checkpoint = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fp8-model-depth2.pt"
        );
        let model = GptModel::load_checkpoint(config.clone(), checkpoint, &device, DType::F32)
            .unwrap()
            .with_compute_dtype(DType::BF16)
            .with_inference_attention(true);
        let tokens = (0..16).collect::<Vec<u32>>();
        let full = model
            .forward(&tokens, 1, 16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/bf16-cuda-cache.json")).unwrap();
        assert_eq!(reference["backend"], "PyTorch CUDA SDPA");
        let mut cache = super::KvCache::new(&config);
        let prefix = model
            .forward_with_cache(&tokens[..8], 1, 8, Some(&mut cache))
            .unwrap();
        assert_eq!(cache.position, 8);
        let prefill_expected = reference["decode_logits"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let prefill_actual = prefix
            .narrow(1, 7, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let prefill_error = prefill_actual
            .iter()
            .zip(&prefill_expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let top_token = |logits: &[f32]| {
            logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0
        };
        assert!(
            prefill_error <= 1e-2,
            "Python prefill max logit error {prefill_error}"
        );
        assert_eq!(top_token(&prefill_actual), top_token(&prefill_expected));
        for position in 8..16 {
            let step = model
                .forward_with_cache(&tokens[position..position + 1], 1, 1, Some(&mut cache))
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let expected = full.narrow(1, position, 1).unwrap();
            let error = step
                .sub(&expected)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                error <= 0.01,
                "cached decode position {position} max logit error {error}"
            );
            let python_expected = reference["decode_logits"][position - 7]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            let actual = step.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let python_error = actual
                .iter()
                .zip(&python_expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                python_error <= 1e-2,
                "Python decode position {position} max logit error {python_error}"
            );
            assert_eq!(top_token(&actual), top_token(&python_expected));
        }
        assert_eq!(prefix.dim(1).unwrap(), 8);
        assert_eq!(cache.position, 16);
    }

    fn save_fixture_state(
        variables: &VarMap,
        path: &std::path::Path,
        compiled_names: bool,
        add_unexpected: bool,
    ) {
        let mut named = variables
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, value)| (name.clone(), value.as_tensor().clone()))
            .collect::<BTreeMap<_, _>>();
        let mut entries = Vec::new();
        for (name, tensor) in named.iter_mut() {
            let tensor = tensor.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
            entries.push((
                TorchValue::String(if compiled_names {
                    format!("_orig_mod.{name}")
                } else {
                    name.clone()
                }),
                TorchValue::Tensor {
                    shape: variables.data().lock().unwrap()[name]
                        .as_tensor()
                        .dims()
                        .to_vec(),
                    values: tensor.to_vec1::<f32>().unwrap(),
                },
            ));
        }
        if add_unexpected {
            entries.push((
                TorchValue::String("unused.weight".into()),
                TorchValue::Tensor {
                    shape: vec![1],
                    values: vec![0.0],
                },
            ));
        }
        crate::torch_save::save_torch_value(TorchValue::Dict(entries), path).unwrap();
    }

    #[test]
    fn compiled_checkpoint_keys_are_mapped_back_to_model_names() {
        let raw = HashSet::from(["_orig_mod.blocks.0.attn.q.weight".to_owned()]);
        assert_eq!(
            checkpoint_source_name("blocks.0.attn.q.weight", &raw),
            "_orig_mod.blocks.0.attn.q.weight"
        );
        let model = HashSet::from(["blocks.0.attn.q.weight".to_owned()]);
        assert!(validate_checkpoint_keys(raw.iter().map(String::as_str), &model).is_ok());
    }

    #[test]
    fn strict_checkpoint_validation_rejects_unexpected_model_keys() {
        let raw = HashSet::from(["lm_head.weight".to_owned(), "unused.weight".to_owned()]);
        let model = HashSet::from(["lm_head.weight".to_owned()]);
        let error = validate_checkpoint_keys(raw.iter().map(String::as_str), &model).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unexpected model parameter unused.weight")
        );
    }

    #[test]
    fn compiled_pytorch_state_dict_loads_with_original_tensor_values() {
        let config = fixture_config();
        let (_, source) =
            GptModel::new_trainable_with_compute_dtype(config.clone(), &Device::Cpu, DType::F32)
                .unwrap();
        let expected = source
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, value)| {
                (
                    name.clone(),
                    value
                        .as_tensor()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let temp = tempfile::tempdir().unwrap();
        let checkpoint = temp.path().join("compiled.pt");
        save_fixture_state(&source, &checkpoint, true, false);

        let (_, loaded) =
            GptModel::load_trainable_pth(config, &checkpoint, &Device::Cpu, DType::F32).unwrap();
        for (name, expected_values) in expected {
            let values = loaded
                .data()
                .lock()
                .unwrap()
                .get(&name)
                .unwrap()
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(values, expected_values, "tensor {name}");
        }
    }

    #[test]
    fn pytorch_state_dict_loader_rejects_an_extra_tensor() {
        let config = fixture_config();
        let (_, variables) =
            GptModel::new_trainable_with_compute_dtype(config.clone(), &Device::Cpu, DType::F32)
                .unwrap();
        let temp = tempfile::tempdir().unwrap();
        let checkpoint = temp.path().join("extra.pt");
        save_fixture_state(&variables, &checkpoint, false, true);
        let error = GptModel::load_trainable_pth(config, &checkpoint, &Device::Cpu, DType::F32)
            .err()
            .expect("unexpected checkpoint tensor should be rejected");
        assert!(
            error
                .to_string()
                .contains("unexpected model parameter unused.weight")
        );
    }
}
