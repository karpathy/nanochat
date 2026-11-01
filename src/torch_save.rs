//! Minimal native writer for PyTorch state-dict checkpoints.
//!
//! nanochat's Python checkpoint manager expects `torch.save(state_dict, path)`
//! archives. This writes the tensor-only subset of that format without loading
//! Python or libtorch, keeping Rust-produced model checkpoints readable by the
//! Python reference during migration.

use std::{collections::BTreeMap, fs::File, io::Write, path::Path};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device};
use candle_nn::VarMap;
use zip::{CompressionMethod, ZipWriter, write::SimpleFileOptions};

/// Values supported by the tensor-only subset of PyTorch's pickle archive.
pub enum TorchValue {
    Dict(Vec<(TorchValue, TorchValue)>),
    List(Vec<TorchValue>),
    Tuple(Vec<TorchValue>),
    String(String),
    Integer(i64),
    Float(f64),
    Tensor { shape: Vec<usize>, values: Vec<f32> },
    IntTensor { shape: Vec<usize>, values: Vec<i32> },
}

enum TensorStorage {
    Float(Vec<f32>),
    Int(Vec<i32>),
}

/// Save the variables in `variables` as a CPU float32 `torch.save` archive.
pub fn save_state_dict(variables: &VarMap, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let named = variables
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("model variable lock is poisoned"))?
        .iter()
        .map(|(name, value)| (name.clone(), value.as_tensor().detach().clone()))
        .collect::<BTreeMap<_, _>>();
    ensure!(
        !named.is_empty(),
        "refusing to save an empty PyTorch state dict"
    );

    let mut entries = Vec::with_capacity(named.len());
    for (name, tensor) in named {
        let flat = tensor
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?;
        let values = flat.to_vec1::<f32>()?;
        let shape = tensor.dims().to_vec();
        let expected_elements = shape
            .iter()
            .try_fold(1usize, |count, size| count.checked_mul(*size))
            .context("tensor element count overflow")?;
        ensure!(
            values.len() == expected_elements,
            "tensor `{name}` has inconsistent flattened length"
        );
        entries.push((
            TorchValue::String(name),
            TorchValue::Tensor { shape, values },
        ));
    }
    save_torch_value(TorchValue::Dict(entries), path)
}

/// Write a supported Python value tree to a CPU `torch.save` archive.
pub fn save_torch_value(value: TorchValue, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let mut pickle = vec![0x80, 2]; // PROTO 2
    let mut tensors = Vec::new();
    encode_value(value, &mut pickle, &mut tensors)?;
    pickle.push(b'.'); // STOP
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut archive = ZipWriter::new(file);
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Stored)
        .with_alignment(64);
    let prefix = "rustagent";

    archive.start_file(format!("{prefix}/data.pkl"), options)?;
    archive.write_all(&pickle)?;
    archive.start_file(format!("{prefix}/byteorder"), options)?;
    archive.write_all(b"little")?;
    for (index, values) in tensors.iter().enumerate() {
        archive.start_file(format!("{prefix}/data/{index}"), options)?;
        match values {
            TensorStorage::Float(values) => {
                write_f32_storage(&mut archive, values)?;
            }
            TensorStorage::Int(values) => {
                write_i32_storage(&mut archive, values)?;
            }
        }
    }
    archive.start_file(format!("{prefix}/version"), options)?;
    archive.write_all(b"3\n")?;
    archive.finish()?;
    Ok(())
}

fn write_f32_storage(archive: &mut ZipWriter<File>, values: &[f32]) -> Result<()> {
    const CHUNK_ELEMENTS: usize = 16 * 1024;
    let mut bytes = Vec::with_capacity(CHUNK_ELEMENTS * std::mem::size_of::<f32>());
    for chunk in values.chunks(CHUNK_ELEMENTS) {
        bytes.clear();
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        archive.write_all(&bytes)?;
    }
    Ok(())
}

fn write_i32_storage(archive: &mut ZipWriter<File>, values: &[i32]) -> Result<()> {
    const CHUNK_ELEMENTS: usize = 16 * 1024;
    let mut bytes = Vec::with_capacity(CHUNK_ELEMENTS * std::mem::size_of::<i32>());
    for chunk in values.chunks(CHUNK_ELEMENTS) {
        bytes.clear();
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        archive.write_all(&bytes)?;
    }
    Ok(())
}

fn encode_value(
    value: TorchValue,
    output: &mut Vec<u8>,
    tensors: &mut Vec<TensorStorage>,
) -> Result<()> {
    match value {
        TorchValue::Dict(entries) => {
            output.push(b'}'); // EMPTY_DICT
            if !entries.is_empty() {
                output.push(b'('); // MARK
                for (key, value) in entries {
                    encode_value(key, output, tensors)?;
                    encode_value(value, output, tensors)?;
                }
                output.push(b'u'); // SETITEMS
            }
        }
        TorchValue::List(values) => {
            output.push(b']'); // EMPTY_LIST
            if !values.is_empty() {
                output.push(b'('); // MARK
                for value in values {
                    encode_value(value, output, tensors)?;
                }
                output.push(b'e'); // APPENDS
            }
        }
        TorchValue::Tuple(values) => {
            output.push(b'('); // MARK
            for value in values {
                encode_value(value, output, tensors)?;
            }
            output.push(b't'); // TUPLE
        }
        TorchValue::String(value) => put_unicode(output, &value)?,
        TorchValue::Integer(value) => put_signed_integer(output, value),
        TorchValue::Float(value) => {
            output.push(b'G'); // BINFLOAT uses big-endian IEEE-754
            output.extend_from_slice(&value.to_be_bytes());
        }
        TorchValue::Tensor { shape, values } => {
            let expected_elements = shape
                .iter()
                .try_fold(1usize, |count, size| count.checked_mul(*size))
                .context("tensor element count overflow")?;
            ensure!(
                values.len() == expected_elements,
                "tensor has inconsistent flattened length"
            );
            let storage_index = tensors.len();
            tensors.push(TensorStorage::Float(values));

            // torch._utils._rebuild_tensor_v2(storage, offset, size, stride,
            // requires_grad, hooks) reconstructs an ordinary dense tensor.
            output.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n(");
            output.push(b'('); // persistent storage ID tuple
            put_unicode(output, "storage")?;
            output.extend_from_slice(b"ctorch\nFloatStorage\n");
            put_unicode(output, &storage_index.to_string())?;
            put_unicode(output, "cpu")?;
            put_integer(
                output,
                match &tensors[storage_index] {
                    TensorStorage::Float(values) => values.len(),
                    TensorStorage::Int(_) => unreachable!(),
                },
            )?;
            output.push(b't'); // TUPLE storage ID
            output.push(b'Q'); // BINPERSID
            put_integer(output, 0)?; // storage offset
            put_dimensions(output, &shape)?;
            put_dimensions(output, &contiguous_strides(&shape)?)?;
            output.push(0x89); // NEWFALSE: requires_grad
            output.extend_from_slice(b"ccollections\nOrderedDict\n)R");
            output.extend_from_slice(b"tR"); // argument TUPLE, REDUCE
        }
        TorchValue::IntTensor { shape, values } => {
            let expected_elements = shape
                .iter()
                .try_fold(1usize, |count, size| count.checked_mul(*size))
                .context("tensor element count overflow")?;
            ensure!(
                values.len() == expected_elements,
                "tensor has inconsistent flattened length"
            );
            let storage_index = tensors.len();
            tensors.push(TensorStorage::Int(values));

            output.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n(");
            output.push(b'(');
            put_unicode(output, "storage")?;
            output.extend_from_slice(b"ctorch\nIntStorage\n");
            put_unicode(output, &storage_index.to_string())?;
            put_unicode(output, "cpu")?;
            put_integer(
                output,
                match &tensors[storage_index] {
                    TensorStorage::Int(values) => values.len(),
                    TensorStorage::Float(_) => unreachable!(),
                },
            )?;
            output.push(b't');
            output.push(b'Q');
            put_integer(output, 0)?;
            put_dimensions(output, &shape)?;
            put_dimensions(output, &contiguous_strides(&shape)?)?;
            output.push(0x89);
            output.extend_from_slice(b"ccollections\nOrderedDict\n)R");
            output.extend_from_slice(b"tR");
        }
    }
    Ok(())
}

fn put_unicode(output: &mut Vec<u8>, value: &str) -> Result<()> {
    let len = u32::try_from(value.len()).context("pickle string is too long")?;
    output.push(b'X'); // BINUNICODE
    output.extend_from_slice(&len.to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_integer(output: &mut Vec<u8>, value: usize) -> Result<()> {
    if value <= u8::MAX as usize {
        output.push(b'K'); // BININT1
        output.push(value as u8);
    } else if value <= u16::MAX as usize {
        output.push(b'M'); // BININT2
        output.extend_from_slice(&(value as u16).to_le_bytes());
    } else if let Ok(value) = i32::try_from(value) {
        output.push(b'J'); // BININT
        output.extend_from_slice(&value.to_le_bytes());
    } else {
        // LONG1 with the shortest positive little-endian two's-complement
        // representation. Tensor element counts are nonnegative.
        let mut bytes = value.to_le_bytes().to_vec();
        while bytes.len() > 1 && bytes.last() == Some(&0) {
            bytes.pop();
        }
        if bytes.last().is_some_and(|byte| byte & 0x80 != 0) {
            bytes.push(0);
        }
        let len = u8::try_from(bytes.len()).context("pickle integer is too large")?;
        output.extend_from_slice(&[0x8a, len]); // LONG1
        output.extend_from_slice(&bytes);
    }
    Ok(())
}

fn put_signed_integer(output: &mut Vec<u8>, value: i64) {
    if (0..=u8::MAX as i64).contains(&value) {
        output.extend_from_slice(&[b'K', value as u8]); // BININT1
    } else if (0..=u16::MAX as i64).contains(&value) {
        output.push(b'M'); // BININT2
        output.extend_from_slice(&(value as u16).to_le_bytes());
    } else if let Ok(value) = i32::try_from(value) {
        output.push(b'J'); // BININT
        output.extend_from_slice(&value.to_le_bytes());
    } else {
        let mut bytes = value.to_le_bytes().to_vec();
        while bytes.len() > 1
            && ((value >= 0 && bytes.last() == Some(&0))
                || (value < 0 && bytes.last() == Some(&0xff)))
        {
            let previous = bytes[bytes.len() - 2];
            if (value >= 0 && previous & 0x80 == 0) || (value < 0 && previous & 0x80 != 0) {
                bytes.pop();
            } else {
                break;
            }
        }
        output.extend_from_slice(&[0x8a, bytes.len() as u8]); // LONG1
        output.extend_from_slice(&bytes);
    }
}

fn put_dimensions(output: &mut Vec<u8>, dimensions: &[usize]) -> Result<()> {
    output.push(b'('); // MARK
    for dimension in dimensions {
        put_integer(output, *dimension)?;
    }
    output.push(b't'); // TUPLE
    Ok(())
}

fn contiguous_strides(shape: &[usize]) -> Result<Vec<usize>> {
    let mut strides = vec![1usize; shape.len()];
    let mut stride = 1usize;
    for index in (0..shape.len()).rev() {
        strides[index] = stride;
        stride = stride
            .checked_mul(shape[index])
            .context("tensor stride overflow")?;
    }
    Ok(strides)
}
