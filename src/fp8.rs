//! FP8 quantization helpers for the CUDA linear-layer implementation.
//!
//! Dynamically scaled FP8 operands and the CUDA cuBLASLt GEMM path.

#[cfg(feature = "cuda-base")]
use anyhow::{Context, Result, ensure};
#[cfg(feature = "cuda-base")]
use candle_core::{DType, Tensor};

#[cfg(feature = "cuda-base")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fp8Format {
    E4M3,
    E5M2,
}

/// FP8 tensor plus the inverse of the scale applied before quantization.
#[cfg(feature = "cuda-base")]
pub struct ScaledFp8 {
    pub values: Tensor,
    pub inverse_scale: Tensor,
    format: Fp8Format,
}

/// Apply nanochat's tensorwise E4M3 scaling to a floating-point tensor.
///
/// The amax reduction uses FP32 values, and the scale division is performed
/// in FP64 before rounding back to FP32, matching `nanochat.fp8._to_fp8`. Scale
/// computation stays on the source device so each eligible layer avoids a
/// device-to-host synchronization.
#[cfg(feature = "cuda-base")]
fn quantize_e4m3_tensorwise(input: &Tensor) -> Result<ScaledFp8> {
    ensure!(
        matches!(
            input.dtype(),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        ),
        "FP8 input must be floating point, got {:?}",
        input.dtype()
    );

    let input_f32 = input
        .to_dtype(DType::F32)
        .context("cast E4M3 source to FP32")?;
    let absolute = input_f32
        .abs()
        .context("take E4M3 source absolute values")?;
    let amax = absolute
        .max_all()
        .context("reduce E4M3 source amax")?
        .to_dtype(DType::F64)
        .context("promote E4M3 source amax to FP64")?;
    let amax = amax
        .clamp(1e-12_f64, f64::MAX)
        .context("clamp E4M3 source amax")?;
    let fp8_max = Tensor::full(448.0_f64, (), amax.device())?;
    let scale = fp8_max.broadcast_div(&amax).context("compute E4M3 scale")?;
    let scale = scale
        .to_dtype(DType::F32)
        .context("cast E4M3 scale to FP32")?;
    let scaled = input_f32
        .broadcast_mul(&scale)
        .context("scale E4M3 source")?;
    let scaled = scaled
        .clamp(-448.0_f64, 448.0_f64)
        .context("clamp E4M3 source")?;
    let values = encode_e4m3(&scaled)?;

    Ok(ScaledFp8 {
        values,
        inverse_scale: scale.recip().context("invert E4M3 scale")?,
        format: Fp8Format::E4M3,
    })
}

/// Encode FP32 CUDA values as finite-only E4M3 bytes. Candle's CUDA dtype cast
/// for E4M3 is not available on all supported CUDA builds, while cuBLASLt
/// accepts the byte representation directly when its operand type is E4M3.
#[cfg(feature = "cuda-base")]
fn encode_e4m3(input: &Tensor) -> Result<Tensor> {
    use std::sync::OnceLock;

    use anyhow::Context;
    use candle_core::cuda::cudarc::{
        driver::{DevicePtr, LaunchConfig, PushKernelArg},
        nvrtc::compile_ptx,
    };
    use candle_core::{Storage, cuda::CudaStorageSlice};

    let input = input.to_dtype(DType::F32)?.contiguous()?;
    let (input_storage, layout) = input.storage_and_layout();
    ensure!(layout.is_contiguous(), "E4M3 input must be contiguous");
    let Storage::Cuda(input_storage) = &*input_storage else {
        anyhow::bail!("E4M3 conversion requires CUDA tensors")
    };
    let CudaStorageSlice::F32(input_slice) = &input_storage.slice else {
        anyhow::bail!("E4M3 input storage must be FP32")
    };
    let device = &input_storage.device;
    let stream = device.cuda_stream();
    let (src, _src_record) = input_slice.device_ptr(&stream);
    let mut output = unsafe { device.alloc::<u8>(layout.shape().elem_count()) }?;
    let kernel = r#"
    extern "C" __global__ void fp32_to_e4m3(const float* x, unsigned char* y, unsigned long long n) {
      unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n) return;
      unsigned int u = __float_as_uint(x[i]);
      unsigned int sign = (u >> 24) & 0x80;
      float a = fabsf(x[i]);
      if (a == 0.0f) { y[i] = (unsigned char)sign; return; }
      if (isnan(a)) { y[i] = (unsigned char)(sign | 0x7f); return; }
      if (isinf(a)) { y[i] = (unsigned char)(sign | 0x7f); return; }
      if (a >= 448.0f) { y[i] = (unsigned char)(sign | 0x7e); return; }
      if (a < 0.015625f) {
        int mant = __float2int_rn(ldexpf(a, 9));
        if (mant >= 8) y[i] = (unsigned char)(sign | 0x08);
        else y[i] = (unsigned char)(sign | (mant & 7));
        return;
      }
      int e;
      float m = frexpf(a, &e) * 2.0f;
      e -= 1;
      int exp = e + 7;
      int mant = __float2int_rn((m - 1.0f) * 8.0f);
      if (mant == 8) { mant = 0; exp += 1; }
      // E4M3FN uses exponent 0b1111 for finite values through mantissa 0b110;
      // only 0b111 is NaN. Preserve the final finite exponent instead of
      // prematurely saturating every value in [256, 448).
      if (exp > 15 || (exp == 15 && mant >= 7)) { y[i] = (unsigned char)(sign | 0x7e); return; }
      if (exp <= 0) {
        int sub = __float2int_rn(ldexpf(a, 9));
        if (sub >= 8) y[i] = (unsigned char)(sign | 0x08);
        else y[i] = (unsigned char)(sign | (sub & 7));
        return;
      }
      y[i] = (unsigned char)(sign | ((exp & 15) << 3) | (mant & 7));
    }"#;
    static E4M3_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    let ptx = E4M3_PTX.get_or_init(|| {
        compile_ptx(kernel)
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
    });
    let ptx = ptx
        .as_ref()
        .map_err(|error| anyhow::anyhow!("compile E4M3 conversion kernel: {error}"))?;
    let func = device.get_or_load_custom_func("fp32_to_e4m3", "nanochat_fp8_e4m3", ptx)?;
    let n = layout.shape().elem_count() as u64;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut launch = func.builder();
    launch.arg(&src);
    launch.arg(&mut output);
    launch.arg(&n);
    unsafe { launch.launch(cfg) }.context("launch E4M3 conversion kernel")?;
    drop(_src_record);
    Ok(Tensor::from_storage(
        Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            output,
            device.clone(),
        )),
        layout.shape().clone(),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// E5M2 quantization used for the output gradient in nanochat's FP8 backward.
#[cfg(feature = "cuda-base")]
fn quantize_e5m2_tensorwise(input: &Tensor) -> Result<ScaledFp8> {
    use anyhow::Context;
    use candle_core::{Storage, cuda::CudaStorageSlice};

    ensure!(
        matches!(
            input.dtype(),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        ),
        "FP8 input must be floating point"
    );
    let input = input.to_dtype(DType::F32)?.contiguous()?;
    let (input_storage, layout) = input.storage_and_layout();
    ensure!(layout.is_contiguous(), "E5M2 input must be contiguous");
    let Storage::Cuda(input_storage) = &*input_storage else {
        anyhow::bail!("E5M2 quantization requires CUDA tensors")
    };
    let CudaStorageSlice::F32(_) = &input_storage.slice else {
        anyhow::bail!("E5M2 input storage must be FP32")
    };
    let amax = input.abs()?.max_all()?.to_dtype(DType::F64)?;
    let amax = amax.clamp(1e-12_f64, f64::MAX)?;
    let scale = Tensor::full(57344.0_f64, (), amax.device())?
        .broadcast_div(&amax)?
        .to_dtype(DType::F32)?;
    let scaled = input
        .broadcast_mul(&scale)?
        .clamp(-57344.0_f64, 57344.0_f64)?
        .contiguous()
        .context("prepare scaled CUDA tensor for E5M2 conversion")?;
    let values = encode_e5m2(&scaled)?;
    Ok(ScaledFp8 {
        values,
        inverse_scale: scale.recip()?,
        format: Fp8Format::E5M2,
    })
}

/// Encode FP32 CUDA values as IEEE E5M2 bytes using round-to-nearest-even.
#[cfg(feature = "cuda-base")]
fn encode_e5m2(input: &Tensor) -> Result<Tensor> {
    use std::sync::OnceLock;

    use anyhow::Context;
    use candle_core::cuda::cudarc::{
        driver::{DevicePtr, LaunchConfig, PushKernelArg},
        nvrtc::compile_ptx,
    };
    use candle_core::{Storage, cuda::CudaStorageSlice};

    let input = input.to_dtype(DType::F32)?.contiguous()?;
    let (input_storage, layout) = input.storage_and_layout();
    ensure!(layout.is_contiguous(), "E5M2 input must be contiguous");
    let Storage::Cuda(input_storage) = &*input_storage else {
        anyhow::bail!("E5M2 conversion requires CUDA tensors")
    };
    let CudaStorageSlice::F32(input_slice) = &input_storage.slice else {
        anyhow::bail!("E5M2 input storage must be FP32")
    };
    let device = &input_storage.device;
    let stream = device.cuda_stream();
    let (src, _src_record) = input_slice.device_ptr(&stream);
    let mut output = unsafe { device.alloc::<u8>(layout.shape().elem_count()) }?;
    const KERNEL: &str = r#"
    extern "C" __global__ void fp32_to_e5m2(const float* x, unsigned char* y, unsigned long long n) {
      unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n) return;
      unsigned int u = __float_as_uint(x[i]);
      unsigned int sign = (u >> 24) & 0x80;
      float a = fabsf(x[i]);
      if (a == 0.0f) { y[i] = (unsigned char)sign; return; }
      if (isnan(a)) { y[i] = (unsigned char)(sign | 0x7f); return; }
      if (isinf(a)) { y[i] = (unsigned char)(sign | 0x7c); return; }
      int e;
      float m = frexpf(a, &e) * 2.0f;
      e -= 1;
      int exp = e + 15;
      if (exp <= 0) {
        float q = ldexpf(a, 16);
        int mant = __float2int_rn(q);
        if (mant >= 4) y[i] = (unsigned char)(sign | 0x04);
        else y[i] = (unsigned char)(sign | (mant & 3));
      } else {
        int mant = __float2int_rn((m - 1.0f) * 4.0f);
        if (mant == 4) { mant = 0; exp += 1; }
        if (exp >= 31) y[i] = (unsigned char)(sign | 0x7c);
        else y[i] = (unsigned char)(sign | ((exp & 31) << 2) | (mant & 3));
      }
    }"#;
    static E5M2_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    let ptx = E5M2_PTX.get_or_init(|| {
        compile_ptx(KERNEL)
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
    });
    let ptx = ptx
        .as_ref()
        .map_err(|error| anyhow::anyhow!("compile E5M2 conversion kernel: {error}"))?;
    let func = device.get_or_load_custom_func("fp32_to_e5m2", "nanochat_fp8_e5m2", &ptx)?;
    let n = layout.shape().elem_count() as u64;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut launch = func.builder();
    launch.arg(&src);
    launch.arg(&mut output);
    launch.arg(&n);
    unsafe { launch.launch(cfg) }.context("launch E5M2 conversion kernel")?;
    drop(_src_record);
    Ok(Tensor::from_storage(
        Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            output,
            device.clone(),
        )),
        layout.shape().clone(),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Multiply row-major FP8 matrices using tensorwise cuBLASLt scales.
///
/// `lhs` has shape `[M, K]`; `rhs` has shape `[N, K]` and is transposed by
/// cuBLASLt, producing FP32 `[M, N]` output. The scale tensors are the inverse
/// scales returned by [`quantize_e4m3_tensorwise`].
#[cfg(feature = "cuda-base")]
fn scaled_matmul(lhs: &ScaledFp8, rhs: &ScaledFp8, fast_accum: bool) -> Result<Tensor> {
    use anyhow::Context;
    use candle_core::cuda::cudarc::{
        cublas::sys::cublasOperation_t,
        cublaslt::sys as lt_sys,
        driver::{DevicePtr, DevicePtrMut},
    };
    use candle_core::{Storage, cuda::CudaStorageSlice};
    use std::mem::size_of;

    let (lhs_storage, lhs_layout) = lhs.values.storage_and_layout();
    let (rhs_storage, rhs_layout) = rhs.values.storage_and_layout();
    let (lhs_scale_storage, lhs_scale_layout) = lhs.inverse_scale.storage_and_layout();
    let (rhs_scale_storage, rhs_scale_layout) = rhs.inverse_scale.storage_and_layout();
    let Storage::Cuda(lhs_storage) = &*lhs_storage else {
        anyhow::bail!("FP8 cuBLASLt GEMM requires CUDA tensors")
    };
    let Storage::Cuda(rhs_storage) = &*rhs_storage else {
        anyhow::bail!("FP8 cuBLASLt GEMM requires CUDA tensors")
    };
    let Storage::Cuda(lhs_scale_storage) = &*lhs_scale_storage else {
        anyhow::bail!("FP8 cuBLASLt scales must be CUDA tensors")
    };
    let Storage::Cuda(rhs_scale_storage) = &*rhs_scale_storage else {
        anyhow::bail!("FP8 cuBLASLt scales must be CUDA tensors")
    };
    ensure!(
        matches!(
            (lhs.format, lhs.values.dtype()),
            (Fp8Format::E4M3, DType::U8) | (Fp8Format::E5M2, DType::U8)
        ),
        "lhs storage dtype does not match its FP8 format"
    );
    ensure!(
        matches!(
            (rhs.format, rhs.values.dtype()),
            (Fp8Format::E4M3, DType::U8) | (Fp8Format::E5M2, DType::U8)
        ),
        "rhs storage dtype does not match its FP8 format"
    );
    ensure!(
        lhs.inverse_scale.dtype() == DType::F32,
        "lhs scale must be FP32"
    );
    ensure!(
        rhs.inverse_scale.dtype() == DType::F32,
        "rhs scale must be FP32"
    );
    ensure!(
        lhs_layout.is_contiguous(),
        "lhs FP8 matrix must be contiguous"
    );
    ensure!(
        rhs_layout.is_contiguous(),
        "rhs FP8 matrix must be contiguous"
    );
    ensure!(
        lhs_scale_layout.shape().elem_count() == 1,
        "lhs scale must be scalar"
    );
    ensure!(
        rhs_scale_layout.shape().elem_count() == 1,
        "rhs scale must be scalar"
    );
    let (m, k) = lhs.values.dims2()?;
    let (n, rhs_k) = rhs.values.dims2()?;
    ensure!(
        k == rhs_k,
        "FP8 GEMM inner dimensions do not match: lhs={m}x{k}, rhs={n}x{rhs_k}"
    );
    ensure!(
        lhs_storage.device.id() == rhs_storage.device.id(),
        "FP8 GEMM tensors use different CUDA devices"
    );

    let device = &lhs_storage.device;
    let stream = device.cuda_stream();
    let (lhs_ptr, _lhs_record) = match (&lhs.format, &lhs_storage.slice) {
        (Fp8Format::E4M3 | Fp8Format::E5M2, CudaStorageSlice::U8(slice)) => {
            let (ptr, record) = slice.device_ptr(&stream);
            (ptr as *const std::ffi::c_void, record)
        }
        _ => anyhow::bail!("lhs FP8 storage does not match its format"),
    };
    let (rhs_ptr, _rhs_record) = match (&rhs.format, &rhs_storage.slice) {
        (Fp8Format::E4M3 | Fp8Format::E5M2, CudaStorageSlice::U8(slice)) => {
            let (ptr, record) = slice.device_ptr(&stream);
            (ptr as *const std::ffi::c_void, record)
        }
        _ => anyhow::bail!("rhs FP8 storage does not match its format"),
    };
    let CudaStorageSlice::F32(lhs_scale_slice) = &lhs_scale_storage.slice else {
        anyhow::bail!("lhs scale storage is not FP32")
    };
    let CudaStorageSlice::F32(rhs_scale_slice) = &rhs_scale_storage.slice else {
        anyhow::bail!("rhs scale storage is not FP32")
    };

    let (a_scale_ptr, _a_scale_record) = lhs_scale_slice.device_ptr(&stream);
    let (b_scale_ptr, _b_scale_record) = rhs_scale_slice.device_ptr(&stream);
    let mut output = unsafe { device.alloc::<f32>(m * n) }?;
    let (output_ptr, _output_record) = output.device_ptr_mut(&stream);
    const WORKSPACE_BYTES: usize = 4 * 1024 * 1024;
    let mut workspace = unsafe { device.alloc::<u8>(WORKSPACE_BYTES) }?;
    let (workspace_ptr, _workspace_record) = workspace.device_ptr_mut(&stream);

    let mut handle = std::ptr::null_mut();
    unsafe { lt_sys::cublasLtCreate(&mut handle).result() }.context("create cuBLASLt handle")?;
    let mut matmul_desc = std::ptr::null_mut();
    let mut a_layout = std::ptr::null_mut();
    let mut b_layout = std::ptr::null_mut();
    let mut c_layout = std::ptr::null_mut();
    let mut preference = std::ptr::null_mut();
    let operation = (|| -> Result<()> {
        unsafe {
            lt_sys::cublasLtMatmulDescCreate(
                &mut matmul_desc,
                lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                lt_sys::cudaDataType_t::CUDA_R_32F,
            )
            .result()
        }
        .context("create FP8 GEMM descriptor")?;
        let trans_b = cublasOperation_t::CUBLAS_OP_T;
        unsafe {
            lt_sys::cublasLtMatmulDescSetAttribute(
                matmul_desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                (&trans_b as *const cublasOperation_t).cast(),
                size_of::<cublasOperation_t>(),
            )
            .result()
        }
        .context("configure FP8 GEMM transpose")?;
        let a_scale_ptr = a_scale_ptr as *const std::ffi::c_void;
        let b_scale_ptr = b_scale_ptr as *const std::ffi::c_void;
        for (attribute, pointer) in [
            (
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                a_scale_ptr,
            ),
            (
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                b_scale_ptr,
            ),
        ] {
            unsafe {
                lt_sys::cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    attribute,
                    (&pointer as *const *const std::ffi::c_void).cast(),
                    size_of::<*const std::ffi::c_void>(),
                )
                .result()
            }
            .context("configure FP8 GEMM scale")?;
        }
        unsafe {
            lt_sys::cublasLtMatmulDescSetAttribute(
                matmul_desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                (&fast_accum as *const bool).cast(),
                size_of::<bool>(),
            )
            .result()
        }
        .context("enable FP8 fast accumulation")?;
        for (rows, cols, ld, dtype, layout) in [
            (
                m as u64,
                k as u64,
                k as i64,
                if lhs.format == Fp8Format::E4M3 {
                    lt_sys::cudaDataType_t::CUDA_R_8F_E4M3
                } else {
                    lt_sys::cudaDataType_t::CUDA_R_8F_E5M2
                },
                &mut a_layout,
            ),
            (
                n as u64,
                k as u64,
                k as i64,
                if rhs.format == Fp8Format::E4M3 {
                    lt_sys::cudaDataType_t::CUDA_R_8F_E4M3
                } else {
                    lt_sys::cudaDataType_t::CUDA_R_8F_E5M2
                },
                &mut b_layout,
            ),
            (
                m as u64,
                n as u64,
                n as i64,
                lt_sys::cudaDataType_t::CUDA_R_32F,
                &mut c_layout,
            ),
        ] {
            unsafe { lt_sys::cublasLtMatrixLayoutCreate(layout, dtype, rows, cols, ld).result() }
                .context("create FP8 GEMM matrix layout")?;
            let order = lt_sys::cublasLtOrder_t::CUBLASLT_ORDER_ROW;
            unsafe {
                lt_sys::cublasLtMatrixLayoutSetAttribute(
                    *layout,
                    lt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
                    (&order as *const lt_sys::cublasLtOrder_t).cast(),
                    size_of::<lt_sys::cublasLtOrder_t>(),
                )
                .result()
            }
            .context("set row-major FP8 GEMM layout")?;
        }
        unsafe { lt_sys::cublasLtMatmulPreferenceCreate(&mut preference).result() }
            .context("create FP8 GEMM preference")?;
        let max_workspace = WORKSPACE_BYTES;
        unsafe {
            lt_sys::cublasLtMatmulPreferenceSetAttribute(
                preference,
                lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&max_workspace as *const usize).cast(),
                size_of::<usize>(),
            )
            .result()
        }
        .context("configure FP8 GEMM workspace")?;
        let mut heuristic = std::mem::MaybeUninit::uninit();
        let mut heuristic_count = 0;
        unsafe {
            lt_sys::cublasLtMatmulAlgoGetHeuristic(
                handle,
                matmul_desc,
                a_layout,
                b_layout,
                c_layout,
                c_layout,
                preference,
                1,
                heuristic.as_mut_ptr(),
                &mut heuristic_count,
            )
            .result()
        }
        .context("select FP8 GEMM kernel")?;
        ensure!(heuristic_count > 0, "cuBLASLt found no FP8 GEMM kernel");
        let heuristic = unsafe { heuristic.assume_init() };
        heuristic
            .state
            .result()
            .context("FP8 GEMM kernel is unsupported")?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            lt_sys::cublasLtMatmul(
                handle,
                matmul_desc,
                (&alpha as *const f32).cast(),
                lhs_ptr,
                a_layout,
                rhs_ptr,
                b_layout,
                (&beta as *const f32).cast(),
                (output_ptr as *const f32).cast(),
                c_layout,
                output_ptr as *mut std::ffi::c_void,
                c_layout,
                &heuristic.algo,
                workspace_ptr as *mut std::ffi::c_void,
                WORKSPACE_BYTES,
                stream.cu_stream() as lt_sys::cudaStream_t,
            )
            .result()
        }
        .context("run scaled FP8 GEMM")?;
        Ok(())
    })();

    unsafe {
        for layout in [a_layout, b_layout, c_layout] {
            if !layout.is_null() {
                lt_sys::cublasLtMatrixLayoutDestroy(layout);
            }
        }
        if !preference.is_null() {
            lt_sys::cublasLtMatmulPreferenceDestroy(preference);
        }
        if !matmul_desc.is_null() {
            lt_sys::cublasLtMatmulDescDestroy(matmul_desc);
        }
        lt_sys::cublasLtDestroy(handle);
    }
    operation?;
    drop(_output_record);
    drop(_workspace_record);

    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            output,
            device.clone(),
        )),
        (m, n),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Apply FP8 compute to a trainable linear layer.
#[cfg(feature = "cuda-base")]
pub fn linear(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    dtype: DType,
) -> Result<Tensor> {
    let input_dims = input.dims();
    ensure!(
        input_dims.len() >= 2,
        "FP8 linear input needs at least two dimensions"
    );
    let (out_features, in_features) = weight.dims2()?;
    ensure!(
        *input_dims.last().unwrap() == in_features,
        "FP8 linear input width does not match weight"
    );
    let rows = input_dims[..input_dims.len() - 1].iter().product::<usize>();
    let input_2d = input
        .to_dtype(dtype)
        .context("cast FP8 linear activation to compute dtype")?
        .reshape((rows, in_features))?;
    let output = input_2d
        .apply_op2(
            weight,
            Fp8Matmul {
                output_dtype: dtype,
            },
        )
        .context("run FP8 autograd linear operation")?;
    let mut output_shape = input_dims[..input_dims.len() - 1].to_vec();
    output_shape.push(out_features);
    let mut output = output.to_dtype(dtype)?.reshape(output_shape)?;
    if let Some(bias) = bias {
        output = output.broadcast_add(&bias.to_dtype(dtype)?.broadcast_as(output.dims())?)?;
    }
    Ok(output)
}

#[cfg(feature = "cuda-base")]
struct Fp8Matmul {
    output_dtype: DType,
}

#[cfg(feature = "cuda-base")]
impl candle_core::CustomOp2 for Fp8Matmul {
    fn name(&self) -> &'static str {
        "nanochat_fp8_linear"
    }

    fn cpu_fwd(
        &self,
        _lhs: &candle_core::CpuStorage,
        _lhs_layout: &candle_core::Layout,
        _rhs: &candle_core::CpuStorage,
        _rhs_layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("FP8 linear is supported only on CUDA")
    }

    fn cuda_fwd(
        &self,
        lhs: &candle_core::CudaStorage,
        lhs_layout: &candle_core::Layout,
        rhs: &candle_core::CudaStorage,
        rhs_layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        let lhs = tensor_from_cuda_storage(lhs, lhs_layout)?;
        let rhs = tensor_from_cuda_storage(rhs, rhs_layout)?;
        let lhs = quantize_e4m3_tensorwise(&lhs)
            .context("quantize FP8 activation as E4M3")
            .map_err(candle_core::Error::wrap)?;
        let rhs = quantize_e4m3_tensorwise(&rhs)
            .context("quantize FP8 weight as E4M3")
            .map_err(candle_core::Error::wrap)?;
        let output = scaled_matmul(&lhs, &rhs, true)
            .context("run E4M3 forward GEMM")
            .map_err(candle_core::Error::wrap)?
            .to_dtype(self.output_dtype)
            .map_err(candle_core::Error::wrap)?;
        let (storage, layout) = output.storage_and_layout();
        let candle_core::Storage::Cuda(storage) = &*storage else {
            candle_core::bail!("FP8 GEMM did not return CUDA storage")
        };
        let storage = storage.transfer_to_device(&storage.device)?;
        Ok((storage, layout.shape().clone()))
    }

    fn bwd(
        &self,
        lhs: &Tensor,
        rhs: &Tensor,
        _output: &Tensor,
        grad_output: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        let lhs_q = quantize_e4m3_tensorwise(lhs)
            .context("quantize saved activation as E4M3 for backward")
            .map_err(candle_core::Error::wrap)?;
        let rhs_q = quantize_e4m3_tensorwise(rhs)
            .context("quantize saved weight as E4M3 for backward")
            .map_err(candle_core::Error::wrap)?;
        let grad_q = quantize_e5m2_tensorwise(grad_output)
            .context("quantize output gradient as E5M2")
            .map_err(candle_core::Error::wrap)?;
        let rhs_q_t = ScaledFp8 {
            values: rhs_q.values.t()?.contiguous()?,
            inverse_scale: rhs_q.inverse_scale.clone(),
            format: rhs_q.format,
        };
        let grad_lhs = scaled_matmul(&grad_q, &rhs_q_t, false)
            .map_err(|error| {
                anyhow::anyhow!("run E5M2 by E4M3 activation-gradient GEMM: {error:#}")
            })
            .map_err(candle_core::Error::wrap)?
            .to_dtype(lhs.dtype())?;
        let grad_q_t = ScaledFp8 {
            values: grad_q.values.t()?.contiguous()?,
            inverse_scale: grad_q.inverse_scale.clone(),
            format: grad_q.format,
        };
        // scaled_matmul interprets its right operand as [output, reduction]
        // and transposes it for cuBLASLt. For dW = grad_output.T @ input,
        // that means passing input.T with shape [width, batch].
        let lhs_q_t = ScaledFp8 {
            values: lhs_q.values.t()?.contiguous()?,
            inverse_scale: lhs_q.inverse_scale.clone(),
            format: lhs_q.format,
        };
        let grad_rhs = scaled_matmul(&grad_q_t, &lhs_q_t, false)
            .context("run E5M2 by E4M3 weight-gradient GEMM")
            .map_err(candle_core::Error::wrap)?
            .to_dtype(grad_output.dtype())?
            .to_dtype(rhs.dtype())?;
        Ok((Some(grad_lhs), Some(grad_rhs)))
    }
}

#[cfg(feature = "cuda-base")]
fn tensor_from_cuda_storage(
    storage: &candle_core::CudaStorage,
    layout: &candle_core::Layout,
) -> candle_core::Result<Tensor> {
    use candle_core::op::BackpropOp;

    if !layout.is_contiguous() || layout.start_offset() != 0 {
        candle_core::bail!("FP8 linear operands must be contiguous and unoffset")
    }
    let storage = storage.transfer_to_device(&storage.device)?;
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        layout.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(all(test, feature = "cuda-base"))]
mod tests {
    use super::{encode_e4m3, encode_e5m2, linear};
    use candle_core::{DType, Device, Tensor, Var};

    #[test]
    fn tensorwise_fp8_linear_matches_pytorch_forward_and_backward_fixture() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let input = (0..16 * 16)
            .map(|i| ((i % 31) as i32 - 15) as f32 / 17.0)
            .collect::<Vec<_>>();
        let weight = (0..16 * 16)
            .map(|i| ((i % 43) as i32 - 21) as f32 / 23.0)
            .collect::<Vec<_>>();
        let output_gradient = (0..16 * 16)
            .map(|i| ((i % 17) as i32 - 8) as f32 / 7.0)
            .collect::<Vec<_>>();
        let input = Var::from_tensor(
            &Tensor::from_vec(input, (16, 16), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        )
        .unwrap();
        // Python keeps master linear weights in FP32 while its activations
        // and matmul output use BF16.
        let weight =
            Var::from_tensor(&Tensor::from_vec(weight, (16, 16), &device).unwrap()).unwrap();
        let output_gradient = Tensor::from_vec(output_gradient, (16, 16), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let output = linear(input.as_tensor(), weight.as_tensor(), None, DType::BF16).unwrap();
        let loss = output
            .to_dtype(DType::F32)
            .unwrap()
            .broadcast_mul(&output_gradient.to_dtype(DType::F32).unwrap())
            .unwrap()
            .sum_all()
            .unwrap();
        let gradients = loss.backward().unwrap();
        let actual = [
            output.to_dtype(DType::F32).unwrap().flatten_all().unwrap(),
            gradients
                .get(input.as_tensor())
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap(),
            gradients
                .get(weight.as_tensor())
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap(),
        ];
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/fp8-linear-bf16.json")).unwrap();
        for (tensor, key) in actual.iter().zip(["out", "gx", "gw"]) {
            let actual = tensor.to_vec1::<f32>().unwrap();
            let expected = expected[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), expected.len());
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);
            if key == "gw" {
                let (index, actual, expected) = actual
                    .iter()
                    .zip(&expected)
                    .enumerate()
                    .map(|(index, (actual, expected))| (index, actual, expected))
                    .max_by(|(_, a0, e0), (_, a1, e1)| {
                        (*a0 - *e0).abs().total_cmp(&(*a1 - *e1).abs())
                    })
                    .unwrap();
                eprintln!("gw max diff at {index}: actual={actual} expected={expected}");
            }
            eprintln!("PyTorch CUDA FP8 fixture {key}: max_abs_error={max_error}");
            assert!(max_error <= 1e-6, "{key} max absolute error: {max_error}");
        }
    }

    #[test]
    fn fp8_encoders_match_pytorch_cuda_rounding_fixtures() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };

        let e4m3_values = [
            0.0,
            2.0f32.powi(-9),
            3.0 * 2.0f32.powi(-9),
            2.0f32.powi(-6),
            1.5 * 2.0f32.powi(-6),
            1.0,
            256.0,
            320.0,
            352.0,
            384.0,
            416.0,
            432.0,
            440.0,
            447.0,
            448.0,
            -1.0,
            f32::INFINITY,
            f32::NAN,
        ];
        let e4m3_input =
            Tensor::from_vec(e4m3_values.to_vec(), (e4m3_values.len(),), &device).unwrap();
        let e4m3 = encode_e4m3(&e4m3_input).unwrap().to_vec1::<u8>().unwrap();
        assert_eq!(
            e4m3,
            [
                0x00, 0x01, 0x03, 0x08, 0x0c, 0x38, 0x78, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7e, 0x7e,
                0x7e, 0xb8, 0x7f, 0x7f,
            ]
        );

        let e5m2_values = [
            0.0,
            2.0f32.powi(-16),
            2.0f32.powi(-14),
            1.5 * 2.0f32.powi(-14),
            1.0,
            57344.0,
            -1.0,
            f32::INFINITY,
            f32::NAN,
        ];
        let e5m2_input =
            Tensor::from_vec(e5m2_values.to_vec(), (e5m2_values.len(),), &device).unwrap();
        let e5m2 = encode_e5m2(&e5m2_input).unwrap().to_vec1::<u8>().unwrap();
        assert_eq!(e5m2, [0x00, 0x01, 0x04, 0x06, 0x3c, 0x7b, 0xbc, 0x7c, 0x7f]);
    }
}
