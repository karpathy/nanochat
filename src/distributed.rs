//! Synchronous multi-process data-parallel collectives for the Rust trainer.
//!
//! Every rank keeps a full model, while optimizer moments are rank-sharded.
//! TCP supplies the portable fallback; Linux CUDA builds can opt into NCCL for
//! device-side parameter broadcast, gradient reduction, and parameter gathers.

use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor, Var, backprop::GradStore};

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
use std::{
    ffi::{c_char, c_int, c_void},
    sync::Arc,
};

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
struct NcclApi {
    _library: libloading::Library,
    get_unique_id: unsafe extern "C" fn(*mut NcclUniqueId) -> c_int,
    comm_init_rank: unsafe extern "C" fn(*mut *mut c_void, c_int, NcclUniqueId, c_int) -> c_int,
    comm_destroy: unsafe extern "C" fn(*mut c_void) -> c_int,
    broadcast: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        c_int,
        c_int,
        *mut c_void,
        *mut c_void,
    ) -> c_int,
    all_reduce: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        c_int,
        c_int,
        *mut c_void,
        *mut c_void,
    ) -> c_int,
    all_gather: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        c_int,
        *mut c_void,
        *mut c_void,
    ) -> c_int,
    get_error_string: unsafe extern "C" fn(c_int) -> *const c_char,
}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
#[repr(C)]
#[derive(Clone, Copy)]
struct NcclUniqueId {
    internal: [c_char; 128],
}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
struct NcclCommunicator {
    api: Arc<NcclApi>,
    handle: *mut c_void,
    stream: Arc<candle_core::cuda::cudarc::driver::CudaStream>,
    world_size: usize,
}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
unsafe impl Send for NcclCommunicator {}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
impl Drop for NcclCommunicator {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.api.comm_destroy)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
impl NcclApi {
    fn load() -> Result<Arc<Self>> {
        let library = unsafe { libloading::Library::new("libnccl.so.2") }
            .or_else(|_| unsafe { libloading::Library::new("libnccl.so") })
            .context("load libnccl.so.2 or libnccl.so")?;
        unsafe fn load<T: Copy>(library: &libloading::Library, symbol: &[u8]) -> Result<T> {
            let value = unsafe { library.get::<T>(symbol) }
                .with_context(|| format!("load NCCL symbol {}", String::from_utf8_lossy(symbol)))?;
            Ok(*value)
        }
        Ok(Arc::new(unsafe {
            Self {
                get_unique_id: load(&library, b"ncclGetUniqueId\0")?,
                comm_init_rank: load(&library, b"ncclCommInitRank\0")?,
                comm_destroy: load(&library, b"ncclCommDestroy\0")?,
                broadcast: load(&library, b"ncclBroadcast\0")?,
                all_reduce: load(&library, b"ncclAllReduce\0")?,
                all_gather: load(&library, b"ncclAllGather\0")?,
                get_error_string: load(&library, b"ncclGetErrorString\0")?,
                _library: library,
            }
        }))
    }

    fn error(&self, code: c_int) -> anyhow::Error {
        let message = unsafe { (self.get_error_string)(code) };
        let message = if message.is_null() {
            "unknown NCCL error".to_owned()
        } else {
            unsafe { std::ffi::CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        };
        anyhow::anyhow!("NCCL error {code}: {message}")
    }

    fn check(&self, code: c_int) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(self.error(code))
        }
    }
}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
impl NcclCommunicator {
    fn broadcast(&self, send: *const c_void, receive: *mut c_void, count: usize) -> Result<()> {
        let code = unsafe {
            (self.api.broadcast)(
                send,
                receive,
                count,
                7, // ncclFloat32
                0,
                self.handle,
                self.stream.cu_stream() as *mut c_void,
            )
        };
        self.api.check(code)
    }

    fn all_reduce_sum(
        &self,
        send: *const c_void,
        receive: *mut c_void,
        count: usize,
    ) -> Result<()> {
        let code = unsafe {
            (self.api.all_reduce)(
                send,
                receive,
                count,
                7, // ncclFloat32
                0, // ncclSum
                self.handle,
                self.stream.cu_stream() as *mut c_void,
            )
        };
        self.api.check(code)
    }

    fn all_gather(&self, send: *const c_void, receive: *mut c_void, count: usize) -> Result<()> {
        let code = unsafe {
            (self.api.all_gather)(
                send,
                receive,
                count,
                7, // ncclFloat32
                self.handle,
                self.stream.cu_stream() as *mut c_void,
            )
        };
        self.api.check(code)
    }
}

pub struct DistributedContext {
    rank: usize,
    world_size: usize,
    backend: String,
    root_streams: Vec<Option<TcpStream>>,
    worker_stream: Option<TcpStream>,
    #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
    nccl: Option<NcclCommunicator>,
}

impl DistributedContext {
    pub fn env_group() -> Result<(usize, usize)> {
        let world_size = env_usize("WORLD_SIZE", 1)?;
        let rank = env_usize("RANK", 0)?;
        ensure!(world_size > 0, "WORLD_SIZE must be positive");
        ensure!(rank < world_size, "RANK must be less than WORLD_SIZE");
        Ok((rank, world_size))
    }

    pub fn from_env() -> Result<Self> {
        let (rank, world_size) = Self::env_group()?;
        let backend =
            std::env::var("RUST_AGENT_DISTRIBUTED_BACKEND").unwrap_or_else(|_| "auto".to_owned());
        ensure!(
            matches!(backend.as_str(), "auto" | "tcp" | "nccl"),
            "RUST_AGENT_DISTRIBUTED_BACKEND must be auto, tcp, or nccl"
        );
        #[cfg(not(all(feature = "cuda-nccl", target_os = "linux")))]
        ensure!(
            backend != "nccl",
            "NCCL backend requires building with --features cuda-nccl"
        );
        if world_size == 1 {
            return Ok(Self {
                rank,
                world_size,
                backend,
                root_streams: Vec::new(),
                worker_stream: None,
                #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
                nccl: None,
            });
        }

        let master_addr = std::env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".into());
        let master_port = std::env::var("MASTER_PORT").unwrap_or_else(|_| "29500".into());
        let explicit_port = std::env::var("RUST_AGENT_PORT")
            .ok()
            .filter(|value| !value.is_empty());
        let agent_store = std::env::var("TORCHELASTIC_USE_AGENT_STORE")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
        let port = collective_port(&master_port, explicit_port.as_deref(), agent_store)?;
        let timeout = Duration::from_secs(180);
        if rank == 0 {
            let listener = TcpListener::bind((master_addr.as_str(), port))
                .with_context(|| format!("bind distributed rendezvous at {master_addr}:{port}"))?;
            listener.set_nonblocking(true)?;
            let deadline = Instant::now() + timeout;
            let mut streams = (0..world_size)
                .map(|_| None)
                .collect::<Vec<Option<TcpStream>>>();
            let mut connected = 0usize;
            while connected < world_size - 1 {
                if Instant::now() >= deadline {
                    anyhow::bail!(
                        "distributed rendezvous timed out: connected {connected}/{} workers",
                        world_size - 1
                    );
                }
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        // Windows may propagate the listener's nonblocking
                        // mode to accepted sockets; collectives use read_exact
                        // and write_all and therefore require blocking streams.
                        stream.set_nonblocking(false)?;
                        stream.set_nodelay(true)?;
                        stream.set_read_timeout(Some(timeout))?;
                        stream.set_write_timeout(Some(timeout))?;
                        let mut rank_bytes = [0u8; 8];
                        stream.read_exact(&mut rank_bytes)?;
                        let worker_rank = u64::from_le_bytes(rank_bytes) as usize;
                        ensure!(
                            worker_rank > 0 && worker_rank < world_size,
                            "invalid distributed worker rank {worker_rank}"
                        );
                        ensure!(
                            streams[worker_rank].is_none(),
                            "distributed rank {worker_rank} connected more than once"
                        );
                        streams[worker_rank] = Some(stream);
                        connected += 1;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(error) => return Err(error).context("accept distributed worker"),
                }
            }
            Ok(Self {
                rank,
                world_size,
                backend: backend.clone(),
                root_streams: streams,
                worker_stream: None,
                #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
                nccl: None,
            })
        } else {
            let deadline = Instant::now() + timeout;
            let mut last_error = None;
            let mut connected = None;
            while connected.is_none() {
                match TcpStream::connect((master_addr.as_str(), port)) {
                    Ok(stream) => {
                        if stream.local_addr()? == stream.peer_addr()? {
                            last_error = Some(std::io::Error::new(
                                std::io::ErrorKind::ConnectionRefused,
                                "distributed worker connected to itself before rank zero was listening",
                            ));
                        } else {
                            connected = Some(stream);
                        }
                    }
                    Err(error) => last_error = Some(error),
                }
                if connected.is_none() {
                    if Instant::now() >= deadline {
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            }
            let mut stream = connected.with_context(|| {
                format!(
                    "connect to distributed rank zero at {master_addr}:{port}: {}",
                    last_error
                        .as_ref()
                        .map_or_else(|| "timeout".to_owned(), ToString::to_string)
                )
            })?;
            stream.set_nodelay(true)?;
            stream.set_read_timeout(Some(timeout))?;
            stream.set_write_timeout(Some(timeout))?;
            stream.write_all(&(rank as u64).to_le_bytes())?;
            Ok(Self {
                rank,
                world_size,
                backend,
                root_streams: Vec::new(),
                worker_stream: Some(stream),
                #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
                nccl: None,
            })
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Copy rank zero's initialized or restored parameters to every worker.
    pub fn broadcast_parameters(&mut self, parameters: &[(String, Var)]) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }
        let local_manifest = parameter_manifest(parameters, &self.backend)?;
        self.validate_parameter_manifest(&local_manifest)?;
        #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
        if let Some((_, parameter)) = parameters.first() {
            self.initialize_nccl(parameter.as_tensor().device())?;
        }
        for (_, parameter) in parameters {
            let tensor = parameter.as_tensor();
            let count = tensor.elem_count();
            #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
            if let Some(comm) = &self.nccl {
                let distributed = nccl_broadcast_tensor(comm, self.rank, tensor)?;
                parameter.set(&distributed.to_dtype(tensor.dtype())?)?;
                continue;
            }
            if self.rank == 0 {
                let values = tensor.flatten_all()?.to_vec1::<f32>()?;
                let bytes = encode_f32(&values);
                for worker in 1..self.world_size {
                    self.write_rank(worker, &bytes)?;
                }
            } else {
                let mut bytes = vec![0u8; count * 4];
                self.read_rank(0, &mut bytes)?;
                let values = decode_f32(&bytes);
                let next = Tensor::from_vec(values, tensor.shape().clone(), tensor.device())?
                    .to_dtype(tensor.dtype())?;
                parameter.set(&next)?;
            }
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
    fn initialize_nccl(&mut self, device: &candle_core::Device) -> Result<()> {
        use std::ffi::c_char;

        if self.nccl.is_some() || self.world_size == 1 {
            return Ok(());
        }
        if self.backend == "tcp" {
            return Ok(());
        }
        let cuda_device = match device {
            candle_core::Device::Cuda(cuda_device) => Some(cuda_device),
            _ => None,
        };
        let api = if cuda_device.is_some() {
            NcclApi::load()
        } else {
            Err(anyhow::anyhow!("NCCL requires a CUDA model device"))
        };
        let all_ranks_available = self.agree_on_nccl_availability(api.is_ok())?;
        if !all_ranks_available {
            if self.backend == "nccl" {
                return Err(api.err().unwrap_or_else(|| {
                    anyhow::anyhow!("NCCL is unavailable on at least one rank")
                }));
            }
            return Ok(());
        }
        let api = api.expect("all-rank NCCL availability implies a local API");
        let cuda_device = cuda_device.expect("all-rank NCCL availability implies a CUDA device");
        let id = if self.rank == 0 {
            let mut id = NcclUniqueId { internal: [0; 128] };
            api.check(unsafe { (api.get_unique_id)(&mut id) })
                .context("create NCCL unique ID")?;
            let id_bytes = id.internal.map(|byte| byte as u8);
            for worker in 1..self.world_size {
                self.write_rank(worker, &id_bytes)?;
            }
            id
        } else {
            let mut bytes = [0u8; 128];
            self.read_rank(0, &mut bytes)?;
            NcclUniqueId {
                internal: bytes.map(|byte| byte as c_char),
            }
        };
        let mut handle = std::ptr::null_mut();
        api.check(unsafe {
            (api.comm_init_rank)(
                &mut handle,
                self.world_size
                    .try_into()
                    .context("NCCL world size exceeds i32")?,
                id,
                self.rank.try_into().context("NCCL rank exceeds i32")?,
            )
        })
        .context("initialize NCCL communicator")?;
        self.nccl = Some(NcclCommunicator {
            api,
            handle,
            stream: cuda_device.cuda_stream(),
            world_size: self.world_size,
        });
        Ok(())
    }

    #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
    fn agree_on_nccl_availability(&mut self, local_available: bool) -> Result<bool> {
        if self.rank == 0 {
            let mut all_available = local_available;
            for worker in 1..self.world_size {
                let mut status = [0u8; 1];
                self.read_rank(worker, &mut status)?;
                all_available &= status[0] == 1;
            }
            let status = [u8::from(all_available)];
            for worker in 1..self.world_size {
                self.write_rank(worker, &status)?;
            }
            Ok(all_available)
        } else {
            self.write_rank(0, &[u8::from(local_available)])?;
            let mut status = [0u8; 1];
            self.read_rank(0, &mut status)?;
            Ok(status[0] == 1)
        }
    }

    /// Ensure every rank agrees on model layout and collective backend before
    /// starting either the unframed TCP stream or device collectives.
    fn validate_parameter_manifest(&mut self, local_manifest: &[u8]) -> Result<()> {
        if self.rank == 0 {
            for worker in 1..self.world_size {
                self.write_rank(worker, &(local_manifest.len() as u64).to_le_bytes())?;
                self.write_rank(worker, local_manifest)?;
            }
            let mut all_match = true;
            for worker in 1..self.world_size {
                let mut reply = [0u8; 1];
                self.read_rank(worker, &mut reply)?;
                all_match &= reply[0] == 1;
            }
            let status = [u8::from(all_match)];
            for worker in 1..self.world_size {
                self.write_rank(worker, &status)?;
            }
            ensure!(
                all_match,
                "distributed parameter manifests differ across ranks; model names, shapes, dtypes, and ordering must match"
            );
        } else {
            let mut length_bytes = [0u8; 8];
            self.read_rank(0, &mut length_bytes)?;
            let length = usize::try_from(u64::from_le_bytes(length_bytes))
                .context("parameter manifest length exceeds this platform")?;
            let mut root_manifest = vec![0u8; length];
            self.read_rank(0, &mut root_manifest)?;
            let matches = root_manifest == local_manifest;
            self.write_rank(0, &[u8::from(matches)])?;
            let mut status = [0u8; 1];
            self.read_rank(0, &mut status)?;
            ensure!(
                matches && status[0] == 1,
                "distributed parameter manifest differs from rank zero; model names, shapes, dtypes, and ordering must match"
            );
        }
        Ok(())
    }

    /// Verify equal-sized buffers before variable-length float collectives.
    fn validate_collective_length(&mut self, length: usize) -> Result<()> {
        if self.rank == 0 {
            let mut all_match = true;
            for worker in 1..self.world_size {
                let mut bytes = [0u8; 8];
                self.read_rank(worker, &mut bytes)?;
                all_match &= u64::from_le_bytes(bytes) == length as u64;
            }
            let status = [u8::from(all_match)];
            for worker in 1..self.world_size {
                self.write_rank(worker, &status)?;
            }
            ensure!(
                all_match,
                "distributed collective buffer lengths differ across ranks"
            );
        } else {
            self.write_rank(0, &(length as u64).to_le_bytes())?;
            let mut status = [0u8; 1];
            self.read_rank(0, &mut status)?;
            ensure!(
                status[0] == 1,
                "distributed collective buffer length differs from rank zero"
            );
        }
        Ok(())
    }

    /// Average rank-local gradients equally, matching PyTorch DDP's reduction.
    pub fn average_gradients(
        &mut self,
        parameters: &[(String, Var)],
        gradients: &GradStore,
    ) -> Result<GradStore> {
        let mut averaged = GradStore::default();
        if self.world_size == 1 {
            for (_, parameter) in parameters {
                if let Some(gradient) = gradients.get(parameter.as_tensor()) {
                    averaged.insert(parameter.as_tensor(), gradient.clone());
                }
            }
            return Ok(averaged);
        }

        // Keep large tensors separate and combine adjacent small tensors to
        // avoid one network round trip per scalar or small matrix.
        for bucket in gradient_buckets(parameters, 256 * 1024) {
            #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
            if let Some(comm) = &self.nccl {
                for (index, gradient) in nccl_average_gradient_bucket(
                    comm,
                    self.world_size,
                    &parameters[bucket.clone()],
                    gradients,
                )?
                .into_iter()
                .enumerate()
                {
                    averaged.insert(parameters[bucket.start + index].1.as_tensor(), gradient);
                }
                continue;
            }
            let mut values = Vec::new();
            for (_, parameter) in &parameters[bucket.clone()] {
                let tensor = parameter.as_tensor();
                let count = tensor.elem_count();
                let local = match gradients.get(tensor) {
                    Some(gradient) => gradient
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    None => vec![0.0f32; count],
                };
                ensure!(local.len() == count, "local gradient shape mismatch");
                values.extend(local);
            }
            if self.rank == 0 {
                for worker in 1..self.world_size {
                    let mut bytes = vec![0u8; values.len() * 4];
                    self.read_rank(worker, &mut bytes)?;
                    for (sum_value, chunk) in values.iter_mut().zip(bytes.chunks_exact(4)) {
                        *sum_value += f32::from_le_bytes(chunk.try_into().expect("4-byte float"));
                    }
                }
                for value in &mut values {
                    *value /= self.world_size as f32;
                }
                let bytes = encode_f32(&values);
                for worker in 1..self.world_size {
                    self.write_rank(worker, &bytes)?;
                }
            } else {
                self.write_rank(0, &encode_f32(&values))?;
                let mut bytes = vec![0u8; values.len() * 4];
                self.read_rank(0, &mut bytes)?;
                values = decode_f32(&bytes);
            }
            if bucket.len() == 1 {
                let tensor = parameters[bucket.start].1.as_tensor();
                let gradient = Tensor::from_vec(values, tensor.shape().clone(), tensor.device())?
                    .to_dtype(tensor.dtype())?;
                averaged.insert(tensor, gradient);
                continue;
            }
            let mut offset = 0;
            for (_, parameter) in &parameters[bucket] {
                let tensor = parameter.as_tensor();
                let count = tensor.elem_count();
                let gradient = Tensor::from_vec(
                    values[offset..offset + count].to_vec(),
                    tensor.shape().clone(),
                    tensor.device(),
                )?
                .to_dtype(tensor.dtype())?;
                averaged.insert(tensor, gradient);
                offset += count;
            }
        }
        Ok(averaged)
    }

    pub fn all_reduce_counts(&mut self, local_a: u64, local_b: u64) -> Result<(u64, u64)> {
        if self.world_size == 1 {
            return Ok((local_a, local_b));
        }
        if self.rank == 0 {
            let mut sum_a = local_a;
            let mut sum_b = local_b;
            for worker in 1..self.world_size {
                let mut bytes = [0u8; 16];
                self.read_rank(worker, &mut bytes)?;
                sum_a += u64::from_le_bytes(bytes[..8].try_into().expect("8-byte count"));
                sum_b += u64::from_le_bytes(bytes[8..].try_into().expect("8-byte count"));
            }
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&sum_a.to_le_bytes());
            bytes.extend_from_slice(&sum_b.to_le_bytes());
            for worker in 1..self.world_size {
                self.write_rank(worker, &bytes)?;
            }
            Ok((sum_a, sum_b))
        } else {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&local_a.to_le_bytes());
            bytes.extend_from_slice(&local_b.to_le_bytes());
            self.write_rank(0, &bytes)?;
            let mut reduced = [0u8; 16];
            self.read_rank(0, &mut reduced)?;
            Ok((
                u64::from_le_bytes(reduced[..8].try_into().expect("8-byte count")),
                u64::from_le_bytes(reduced[8..].try_into().expect("8-byte count")),
            ))
        }
    }

    /// Sum float32 values across ranks and broadcast, matching PyTorch's
    /// float32 `ReduceOp.SUM` metric path.
    pub fn all_reduce_f32_sum(&mut self, local: &[f32]) -> Result<Vec<f32>> {
        if self.world_size == 1 {
            return Ok(local.to_vec());
        }
        self.validate_collective_length(local.len())?;
        if self.rank == 0 {
            let mut sum = local.to_vec();
            for worker in 1..self.world_size {
                let mut bytes = vec![0u8; local.len() * 4];
                self.read_rank(worker, &mut bytes)?;
                for (value, bytes) in sum.iter_mut().zip(bytes.chunks_exact(4)) {
                    *value += f32::from_le_bytes(bytes.try_into().expect("4-byte float"));
                }
            }
            let bytes = encode_f32(&sum);
            for worker in 1..self.world_size {
                self.write_rank(worker, &bytes)?;
            }
            Ok(sum)
        } else {
            self.write_rank(0, &encode_f32(local))?;
            let mut bytes = vec![0u8; local.len() * 4];
            self.read_rank(0, &mut bytes)?;
            Ok(decode_f32(&bytes))
        }
    }

    /// Sum float32 values across ranks, divide by world size, and broadcast.
    /// The wire format and arithmetic match PyTorch's float32 `ReduceOp.AVG`.
    pub fn all_reduce_f32_average(&mut self, local: &[f32]) -> Result<Vec<f32>> {
        let mut sum = self.all_reduce_f32_sum(local)?;
        let scale = 1.0f32 / self.world_size as f32;
        for value in &mut sum {
            *value *= scale;
        }
        Ok(sum)
    }

    /// Gather equal-sized rank slices in rank order and return the complete
    /// concatenated value on every process. Used when importing stacked Muon
    /// moment shards from Python checkpoints.
    pub fn all_gather_f32(&mut self, local: &[f32]) -> Result<Vec<f32>> {
        if self.world_size == 1 {
            return Ok(local.to_vec());
        }
        self.validate_collective_length(local.len())?;
        let local_bytes = encode_f32(local);
        if self.rank == 0 {
            let mut gathered = Vec::with_capacity(local.len() * self.world_size);
            gathered.extend_from_slice(local);
            for worker in 1..self.world_size {
                let mut bytes = vec![0u8; local_bytes.len()];
                self.read_rank(worker, &mut bytes)?;
                gathered.extend(decode_f32(&bytes));
            }
            let bytes = encode_f32(&gathered);
            for worker in 1..self.world_size {
                self.write_rank(worker, &bytes)?;
            }
            Ok(gathered)
        } else {
            self.write_rank(0, &local_bytes)?;
            let mut bytes = vec![0u8; local.len() * self.world_size * 4];
            self.read_rank(0, &mut bytes)?;
            Ok(decode_f32(&bytes))
        }
    }

    /// Gather an equal-shaped float tensor from every rank, prepending rank as
    /// the outer dimension. NCCL keeps the payload on-device; TCP is the
    /// portable fallback.
    pub fn all_gather_tensor(&mut self, local: &Tensor) -> Result<Tensor> {
        if self.world_size == 1 {
            return Ok(local.clone());
        }
        let mut output_shape = vec![self.world_size];
        output_shape.extend_from_slice(local.dims());
        #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
        if self.nccl.is_some() {
            self.validate_collective_length(local.elem_count())?;
            let comm = self.nccl.as_ref().expect("NCCL communicator checked");
            return nccl_all_gather_tensor(comm, local, &output_shape);
        }

        let values = local
            .to_dtype(DType::F32)?
            .to_device(&candle_core::Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let gathered = self.all_gather_f32(&values)?;
        let result = Tensor::from_vec(gathered, output_shape, &candle_core::Device::Cpu)?;
        Ok(result.to_device(local.device())?)
    }

    fn write_rank(&mut self, rank: usize, bytes: &[u8]) -> Result<()> {
        let stream = if self.rank == 0 {
            self.root_streams[rank]
                .as_mut()
                .context("worker connection is missing")?
        } else {
            ensure!(rank == 0, "workers can communicate only with rank zero");
            self.worker_stream
                .as_mut()
                .context("rank zero connection is missing")?
        };
        stream.write_all(bytes)?;
        Ok(())
    }

    fn read_rank(&mut self, rank: usize, bytes: &mut [u8]) -> Result<()> {
        let stream = if self.rank == 0 {
            self.root_streams[rank]
                .as_mut()
                .context("worker connection is missing")?
        } else {
            ensure!(rank == 0, "workers can communicate only with rank zero");
            self.worker_stream
                .as_mut()
                .context("rank zero connection is missing")?
        };
        stream.read_exact(bytes)?;
        Ok(())
    }
}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
fn nccl_broadcast_tensor(comm: &NcclCommunicator, rank: usize, tensor: &Tensor) -> Result<Tensor> {
    use candle_core::cuda::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::{Device, Storage, cuda::CudaStorageSlice, op::BackpropOp};

    let Device::Cuda(device) = tensor.device() else {
        anyhow::bail!("NCCL parameter broadcast requires CUDA tensors")
    };
    let source = tensor.to_dtype(DType::F32)?.contiguous()?;
    let (source_storage, layout) = source.storage_and_layout();
    let Storage::Cuda(source_storage) = &*source_storage else {
        anyhow::bail!("NCCL parameter source is not CUDA storage")
    };
    let CudaStorageSlice::F32(source_slice) = &source_storage.slice else {
        anyhow::bail!("NCCL parameter source is not FP32")
    };
    let mut output = unsafe { device.alloc::<f32>(source.elem_count()) }?;
    let (source_ptr, _source_record) = source_slice.device_ptr(&comm.stream);
    let (output_ptr, _output_record) = output.device_ptr_mut(&comm.stream);
    comm.broadcast(
        source_ptr as *const c_void,
        output_ptr as *mut c_void,
        source.elem_count(),
    )?;
    drop(_source_record);
    drop(_output_record);
    ensure!(
        rank < comm.world_size,
        "NCCL rank is outside its communicator"
    );
    Ok(Tensor::from_storage(
        Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            output,
            device.clone(),
        )),
        layout.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
fn nccl_all_gather_tensor(
    comm: &NcclCommunicator,
    local: &Tensor,
    output_shape: &[usize],
) -> Result<Tensor> {
    use candle_core::cuda::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::{Device, Storage, cuda::CudaStorageSlice, op::BackpropOp};

    let Device::Cuda(device) = local.device() else {
        anyhow::bail!("NCCL optimizer gather requires CUDA tensors")
    };
    let source = local.to_dtype(DType::F32)?.contiguous()?;
    let (source_storage, _) = source.storage_and_layout();
    let Storage::Cuda(source_storage) = &*source_storage else {
        anyhow::bail!("NCCL optimizer gather source is not CUDA storage")
    };
    let CudaStorageSlice::F32(source_slice) = &source_storage.slice else {
        anyhow::bail!("NCCL optimizer gather source is not FP32")
    };
    let mut output = unsafe { device.alloc::<f32>(source.elem_count() * comm.world_size) }?;
    let (source_ptr, _source_record) = source_slice.device_ptr(&comm.stream);
    let (output_ptr, _output_record) = output.device_ptr_mut(&comm.stream);
    comm.all_gather(
        source_ptr as *const c_void,
        output_ptr as *mut c_void,
        source.elem_count(),
    )?;
    drop(_source_record);
    drop(_output_record);
    ensure!(
        output_shape.iter().product::<usize>() == source.elem_count() * comm.world_size,
        "NCCL optimizer gather output shape does not match its rank payloads"
    );
    Ok(Tensor::from_storage(
        Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            output,
            device.clone(),
        )),
        output_shape,
        BackpropOp::none(),
        false,
    ))
}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
fn nccl_average_gradient_bucket(
    comm: &NcclCommunicator,
    world_size: usize,
    parameters: &[(String, Var)],
    gradients: &GradStore,
) -> Result<Vec<Tensor>> {
    use candle_core::cuda::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::{Device, Storage, cuda::CudaStorageSlice, op::BackpropOp};

    ensure!(!parameters.is_empty(), "NCCL gradient bucket is empty");
    let device = parameters[0].1.as_tensor().device();
    let Device::Cuda(cuda_device) = device else {
        anyhow::bail!("NCCL gradient reduction requires CUDA tensors")
    };
    let mut flat_gradients = Vec::with_capacity(parameters.len());
    for (_, parameter) in parameters {
        let tensor = parameter.as_tensor();
        let gradient = match gradients.get(tensor) {
            Some(gradient) => gradient.to_dtype(DType::F32)?,
            None => Tensor::zeros(tensor.shape(), DType::F32, tensor.device())?,
        };
        flat_gradients.push(gradient.flatten_all()?.contiguous()?);
    }
    let refs = flat_gradients.iter().collect::<Vec<_>>();
    let input = if refs.len() == 1 {
        flat_gradients.remove(0)
    } else {
        Tensor::cat(&refs, 0)?.contiguous()?
    };
    let (input_storage, layout) = input.storage_and_layout();
    let Storage::Cuda(input_storage) = &*input_storage else {
        anyhow::bail!("NCCL gradient bucket is not CUDA storage")
    };
    let CudaStorageSlice::F32(input_slice) = &input_storage.slice else {
        anyhow::bail!("NCCL gradient bucket is not FP32")
    };
    let mut output = unsafe { cuda_device.alloc::<f32>(layout.shape().elem_count()) }?;
    let (input_ptr, _input_record) = input_slice.device_ptr(&comm.stream);
    let (output_ptr, _output_record) = output.device_ptr_mut(&comm.stream);
    comm.all_reduce_sum(
        input_ptr as *const c_void,
        output_ptr as *mut c_void,
        layout.shape().elem_count(),
    )?;
    drop(_input_record);
    drop(_output_record);
    let summed = Tensor::from_storage(
        Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            output,
            cuda_device.clone(),
        )),
        layout.shape().clone(),
        BackpropOp::none(),
        false,
    );
    let averaged = summed.affine(1.0 / world_size as f64, 0.0)?;
    let mut offset = 0usize;
    let mut result = Vec::with_capacity(parameters.len());
    for (_, parameter) in parameters {
        let tensor = parameter.as_tensor();
        let count = tensor.elem_count();
        result.push(
            averaged
                .narrow(0, offset, count)?
                .reshape(tensor.shape())?
                .to_dtype(tensor.dtype())?,
        );
        offset += count;
    }
    ensure!(
        offset == summed.elem_count(),
        "NCCL gradient bucket size differs from its parameters"
    );
    Ok(result)
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .with_context(|| format!("{name} must be an integer"))
        })
        .unwrap_or(Ok(default))
}

fn collective_port(master: &str, explicit: Option<&str>, agent_store: bool) -> Result<u16> {
    if let Some(value) = explicit {
        let port = value
            .parse::<u16>()
            .context("RUST_AGENT_PORT must be a TCP port")?;
        ensure!(port != 0, "RUST_AGENT_PORT must be nonzero");
        return Ok(port);
    }
    let port = master
        .parse::<u16>()
        .context("MASTER_PORT must be a TCP port")?;
    ensure!(port != 0, "MASTER_PORT must be nonzero");
    Ok(if agent_store {
        port.checked_add(1).unwrap_or(port - 1)
    } else {
        port
    })
}

fn parameter_manifest(parameters: &[(String, Var)], backend: &str) -> Result<Vec<u8>> {
    let entries = parameters
        .iter()
        .map(|(name, parameter)| {
            let tensor = parameter.as_tensor();
            (name, tensor.dims(), format!("{:?}", tensor.dtype()))
        })
        .collect::<Vec<_>>();
    Ok(serde_json::to_vec(&(backend, entries))?)
}

fn gradient_buckets(
    parameters: &[(String, Var)],
    max_elements: usize,
) -> Vec<std::ops::Range<usize>> {
    let mut buckets = Vec::new();
    let mut start = 0;
    while start < parameters.len() {
        let mut end = start + 1;
        let mut count = parameters[start].1.as_tensor().elem_count();
        while end < parameters.len() {
            let next = parameters[end].1.as_tensor().elem_count();
            if count.saturating_add(next) > max_elements {
                break;
            }
            count += next;
            end += 1;
        }
        buckets.push(start..end);
        start = end;
    }
    buckets
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{DistributedContext, collective_port, gradient_buckets};
    use candle_core::{DType, Device, Tensor, Var, backprop::GradStore};
    use std::{net::TcpListener, thread};

    fn context_pair() -> (DistributedContext, DistributedContext) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let connector = thread::spawn(move || std::net::TcpStream::connect(address).unwrap());
        let (root_stream, _) = listener.accept().unwrap();
        let worker_stream = connector.join().unwrap();
        (
            DistributedContext {
                rank: 0,
                world_size: 2,
                backend: "auto".to_owned(),
                root_streams: vec![None, Some(root_stream)],
                worker_stream: None,
                #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
                nccl: None,
            },
            DistributedContext {
                rank: 1,
                world_size: 2,
                backend: "auto".to_owned(),
                root_streams: Vec::new(),
                worker_stream: Some(worker_stream),
                #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
                nccl: None,
            },
        )
    }

    #[test]
    fn torchrun_agent_store_uses_a_separate_collective_port() {
        assert_eq!(collective_port("29500", None, false).unwrap(), 29500);
        assert_eq!(collective_port("29500", None, true).unwrap(), 29501);
        assert_eq!(collective_port("65535", None, true).unwrap(), 65534);
        assert_eq!(
            collective_port("29500", Some("31000"), true).unwrap(),
            31000
        );
        assert!(collective_port("29500", Some("0"), true).is_err());
    }

    #[test]
    fn parameter_manifest_handshake_accepts_matching_ranks() {
        let (mut root, mut worker) = context_pair();
        let worker_check = thread::spawn(move || {
            worker.validate_parameter_manifest(b"same model").unwrap();
        });
        root.validate_parameter_manifest(b"same model").unwrap();
        worker_check.join().unwrap();
    }

    #[test]
    fn parameter_manifest_handshake_rejects_mismatched_ranks() {
        let (mut root, mut worker) = context_pair();
        let worker_check =
            thread::spawn(move || worker.validate_parameter_manifest(b"other model"));
        let root_error = root.validate_parameter_manifest(b"root model").unwrap_err();
        let worker_error = worker_check.join().unwrap().unwrap_err();
        assert!(root_error.to_string().contains("manifests differ"));
        assert!(worker_error.to_string().contains("differs from rank zero"));
    }

    #[test]
    fn parameter_manifest_rejects_different_collective_backends() {
        let (mut root, mut worker) = context_pair();
        worker.backend = "tcp".to_owned();
        let parameters = || {
            vec![(
                "weight".to_owned(),
                Var::from_vec(vec![0.0f32; 2], (2,), &Device::Cpu).unwrap(),
            )]
        };
        let worker_check = thread::spawn(move || worker.broadcast_parameters(&parameters()));
        let root_error = root.broadcast_parameters(&parameters()).unwrap_err();
        let worker_error = worker_check.join().unwrap().unwrap_err();
        assert!(root_error.to_string().contains("manifests differ"));
        assert!(worker_error.to_string().contains("differs from rank zero"));
    }

    #[test]
    fn collective_length_handshake_accepts_equal_slices() {
        let (mut root, mut worker) = context_pair();
        let worker_check = thread::spawn(move || worker.validate_collective_length(17).unwrap());
        root.validate_collective_length(17).unwrap();
        worker_check.join().unwrap();
    }

    #[test]
    fn collective_length_handshake_rejects_unequal_slices() {
        let (mut root, mut worker) = context_pair();
        let worker_check = thread::spawn(move || worker.validate_collective_length(8));
        let root_error = root.validate_collective_length(12).unwrap_err();
        let worker_error = worker_check.join().unwrap().unwrap_err();
        assert!(root_error.to_string().contains("lengths differ"));
        assert!(worker_error.to_string().contains("differs from rank zero"));
    }

    #[test]
    fn broadcast_and_gradient_average_keep_two_ranks_in_sync() {
        fn run(
            mut context: DistributedContext,
            initial: Vec<f32>,
            local_grad: Vec<f32>,
        ) -> (Vec<f32>, Vec<f32>) {
            let parameter = Var::from_vec(initial, (2,), &Device::Cpu).unwrap();
            let parameters = vec![("weight".to_owned(), parameter.clone())];
            context.broadcast_parameters(&parameters).unwrap();
            let mut gradients = GradStore::default();
            gradients.insert(
                parameter.as_tensor(),
                Tensor::from_vec(local_grad, (2,), &Device::Cpu).unwrap(),
            );
            let averaged = context.average_gradients(&parameters, &gradients).unwrap();
            (
                parameter.as_tensor().to_vec1::<f32>().unwrap(),
                averaged
                    .get(parameter.as_tensor())
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            )
        }
        let (root, worker) = context_pair();
        let worker = thread::spawn(move || run(worker, vec![9.0, 9.0], vec![3.0, 7.0]));
        let root = run(root, vec![1.0, 2.0], vec![1.0, 5.0]);
        let worker = worker.join().unwrap();
        assert_eq!(root, (vec![1.0, 2.0], vec![2.0, 6.0]));
        assert_eq!(worker, root);
    }

    #[test]
    fn sharded_optimizer_updates_match_replicated_reference() {
        fn params() -> Vec<(String, Var)> {
            let mut parameters = vec![(
                "lm_head.weight".to_owned(),
                Var::from_vec(
                    (0..1024).map(|index| index as f32 * 1e-4).collect(),
                    (4, 256),
                    &Device::Cpu,
                )
                .unwrap(),
            )];
            for index in 0..3 {
                parameters.push((
                    format!("transformer.h.{index}.attn.c_q.weight"),
                    Var::from_vec(
                        (0..16).map(|value| value as f32 * 0.01).collect(),
                        (4, 4),
                        &Device::Cpu,
                    )
                    .unwrap(),
                ));
            }
            parameters.extend([
                (
                    "smear_gate.weight".to_owned(),
                    Var::from_vec(vec![0.01f32; 24 * 24], (24, 24), &Device::Cpu).unwrap(),
                ),
                (
                    "smear_lambda".to_owned(),
                    Var::from_vec(vec![0.1f32], (1,), &Device::Cpu).unwrap(),
                ),
                (
                    "backout_lambda".to_owned(),
                    Var::from_vec(vec![0.1f32], (1,), &Device::Cpu).unwrap(),
                ),
            ]);
            parameters
        }

        fn config() -> crate::optim::MuonAdamWConfig {
            crate::optim::MuonAdamWConfig {
                embedding_lr: 0.2,
                unembedding_lr: 0.004,
                matrix_lr: 0.02,
                scalar_lr: 0.5,
                weight_decay: 0.01,
            }
        }

        fn run(
            mut context: DistributedContext,
            rank: usize,
            checkpoint_dir: &std::path::Path,
        ) -> Vec<Vec<f32>> {
            let parameters = params();
            context.broadcast_parameters(&parameters).unwrap();
            let mut optimizer =
                crate::optim::MuonAdamW::new(parameters.clone(), 768, config()).unwrap();
            optimizer.shard_state(rank, 2).unwrap();
            let mut local_gradients = GradStore::default();
            for (_, parameter) in &parameters {
                let values = vec![rank as f32 + 1.0; parameter.as_tensor().elem_count()];
                local_gradients.insert(
                    parameter.as_tensor(),
                    Tensor::from_vec(values, parameter.shape().clone(), &Device::Cpu).unwrap(),
                );
            }
            let averaged = context
                .average_gradients(&parameters, &local_gradients)
                .unwrap();
            optimizer
                .step_distributed(&averaged, 1.0, 0.95, 1.0, DType::F32, &mut context)
                .unwrap();
            optimizer.save_state(checkpoint_dir, 1, rank, 2).unwrap();
            let step_one = parameters
                .iter()
                .map(|(_, parameter)| {
                    parameter
                        .as_tensor()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                })
                .collect::<Vec<_>>();

            let mut second_local = GradStore::default();
            for (_, parameter) in &parameters {
                second_local.insert(
                    parameter.as_tensor(),
                    Tensor::full(rank as f32 + 2.0, parameter.shape(), &Device::Cpu).unwrap(),
                );
            }
            let second_average = context
                .average_gradients(&parameters, &second_local)
                .unwrap();
            optimizer
                .step_distributed(&second_average, 0.75, 0.95, 0.8, DType::F32, &mut context)
                .unwrap();
            let uninterrupted = parameters
                .iter()
                .map(|(_, parameter)| {
                    parameter
                        .as_tensor()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                })
                .collect::<Vec<_>>();

            let resumed_parameters = params();
            for ((_, parameter), values) in resumed_parameters.iter().zip(&step_one) {
                parameter
                    .set(
                        &Tensor::from_vec(values.clone(), parameter.shape().clone(), &Device::Cpu)
                            .unwrap(),
                    )
                    .unwrap();
            }
            let mut resumed =
                crate::optim::MuonAdamW::new(resumed_parameters.clone(), 768, config()).unwrap();
            resumed.shard_state(rank, 2).unwrap();
            assert_eq!(
                resumed
                    .load_state(checkpoint_dir, Some(1), &mut context)
                    .unwrap(),
                Some(1)
            );
            let mut resumed_local = GradStore::default();
            for (_, parameter) in &resumed_parameters {
                resumed_local.insert(
                    parameter.as_tensor(),
                    Tensor::full(rank as f32 + 2.0, parameter.shape(), &Device::Cpu).unwrap(),
                );
            }
            let resumed_average = context
                .average_gradients(&resumed_parameters, &resumed_local)
                .unwrap();
            resumed
                .step_distributed(&resumed_average, 0.75, 0.95, 0.8, DType::F32, &mut context)
                .unwrap();
            let resumed_values = resumed_parameters
                .iter()
                .map(|(_, parameter)| {
                    parameter
                        .as_tensor()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(resumed_values, uninterrupted);
            uninterrupted
        }

        let (root, worker) = context_pair();
        let checkpoint_dir =
            std::env::temp_dir().join(format!("rust-agent-zero2-{}", std::process::id()));
        std::fs::create_dir_all(&checkpoint_dir).unwrap();
        let worker_dir = checkpoint_dir.clone();
        let worker = thread::spawn(move || run(worker, 1, &worker_dir));
        let root = run(root, 0, &checkpoint_dir);
        assert_eq!(worker.join().unwrap(), root);
        assert!(
            checkpoint_dir
                .join("optimizer_000001_rank0.safetensors")
                .is_file()
        );
        assert!(
            checkpoint_dir
                .join("optimizer_000001_rank1.safetensors")
                .is_file()
        );
        assert!(checkpoint_dir.join("optim_000001_rank0.pt").is_file());
        assert!(checkpoint_dir.join("optim_000001_rank1.pt").is_file());

        let parameters = params();
        let mut reference =
            crate::optim::MuonAdamW::new(parameters.clone(), 768, config()).unwrap();
        for (grad_value, lr, decay) in [(1.5, 1.0, 1.0), (2.5, 0.75, 0.8)] {
            let mut average = GradStore::default();
            for (_, parameter) in &parameters {
                average.insert(
                    parameter.as_tensor(),
                    Tensor::full(grad_value, parameter.shape(), &Device::Cpu).unwrap(),
                );
            }
            reference
                .step(&average, lr, 0.95, decay, DType::F32)
                .unwrap();
        }
        let expected = parameters
            .iter()
            .map(|(_, parameter)| {
                parameter
                    .as_tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(root, expected);
    }

    #[test]
    fn bucketed_gradient_average_preserves_per_parameter_values() {
        fn run(mut context: DistributedContext, rank: usize) -> Vec<Vec<f32>> {
            let parameters: Vec<_> = [2usize, 3, 1]
                .into_iter()
                .enumerate()
                .map(|(index, len)| {
                    (
                        format!("p{index}"),
                        Var::from_vec(vec![0.0f32; len], (len,), &Device::Cpu).unwrap(),
                    )
                })
                .collect();
            assert_eq!(gradient_buckets(&parameters, 4), vec![0..1, 1..3]);
            context.broadcast_parameters(&parameters).unwrap();
            let mut local = GradStore::default();
            let values = if rank == 0 {
                vec![Some(vec![1.0, 3.0]), None, Some(vec![4.0])]
            } else {
                vec![Some(vec![3.0, 5.0]), Some(vec![2.0, 4.0, 6.0]), None]
            };
            for ((_, parameter), value) in parameters.iter().zip(values) {
                if let Some(value) = value {
                    local.insert(
                        parameter.as_tensor(),
                        Tensor::from_vec(value, parameter.shape().clone(), &Device::Cpu).unwrap(),
                    );
                }
            }
            let averaged = context.average_gradients(&parameters, &local).unwrap();
            parameters
                .iter()
                .map(|(_, parameter)| {
                    averaged
                        .get(parameter.as_tensor())
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                })
                .collect()
        }
        let (root, worker) = context_pair();
        let worker = thread::spawn(move || run(worker, 1));
        let root = run(root, 0);
        assert_eq!(root, vec![vec![2.0, 4.0], vec![1.0, 2.0, 3.0], vec![2.0]]);
        assert_eq!(worker.join().unwrap(), root);
    }

    #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
    #[test]
    fn nccl_availability_falls_back_collectively_when_one_rank_lacks_it() {
        let (mut root, mut worker) = context_pair();
        let worker_check = thread::spawn(move || worker.agree_on_nccl_availability(false));
        assert!(!root.agree_on_nccl_availability(true).unwrap());
        assert!(!worker_check.join().unwrap().unwrap());
    }

    #[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
    #[test]
    fn nccl_cuda_broadcast_and_average_run_on_one_rank() {
        let Ok(Device::Cuda(device)) = Device::new_cuda(0) else {
            return;
        };
        let Ok(api) = super::NcclApi::load() else {
            return;
        };
        use std::ptr;

        let mut id = super::NcclUniqueId { internal: [0; 128] };
        api.check(unsafe { (api.get_unique_id)(&mut id) }).unwrap();
        let mut handle = ptr::null_mut();
        api.check(unsafe { (api.comm_init_rank)(&mut handle, 1, id, 0) })
            .unwrap();
        let comm = super::NcclCommunicator {
            api,
            handle,
            stream: device.cuda_stream(),
            world_size: 1,
        };
        let cuda = Device::Cuda(device);
        let parameter = Var::from_vec(vec![2.0f32, -4.0], (2,), &cuda).unwrap();
        let broadcast = super::nccl_broadcast_tensor(&comm, 0, parameter.as_tensor()).unwrap();
        assert_eq!(broadcast.to_vec1::<f32>().unwrap(), [2.0, -4.0]);
        let gathered =
            super::nccl_all_gather_tensor(&comm, parameter.as_tensor(), &[1, 2]).unwrap();
        assert_eq!(
            gathered.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [2.0, -4.0]
        );

        let parameters = vec![("weight".to_owned(), parameter.clone())];
        let mut local = GradStore::default();
        local.insert(
            parameter.as_tensor(),
            Tensor::from_vec(vec![3.0f32, 7.0], (2,), &cuda).unwrap(),
        );
        let average = super::nccl_average_gradient_bucket(&comm, 1, &parameters, &local).unwrap();
        assert_eq!(average[0].to_vec1::<f32>().unwrap(), [3.0, 7.0]);
    }
}
