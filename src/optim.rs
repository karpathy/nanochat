//! Single-device mixed AdamW/Muon optimizer matching nanochat's parameter groups.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor, Var, backprop::GradStore, pickle};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::torch_save::TorchValue;

const POLAR: [(f64, f64, f64); 5] = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
];

#[derive(Clone, Copy)]
pub struct MuonAdamWConfig {
    pub embedding_lr: f64,
    pub unembedding_lr: f64,
    pub matrix_lr: f64,
    pub scalar_lr: f64,
    pub weight_decay: f64,
}

struct AdamState {
    name: String,
    parameter: Var,
    first: Tensor,
    second: Tensor,
    step: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

struct MuonState {
    name: String,
    parameter: Var,
    momentum: Tensor,
    second: Tensor,
    lr: f64,
    weight_decay: f64,
}

pub struct MuonAdamW {
    adam: Vec<AdamState>,
    muon: Vec<MuonState>,
    config: MuonAdamWConfig,
    model_dim_scale: f64,
    shard_rank: usize,
    shard_world_size: usize,
}

#[derive(Serialize, Deserialize)]
struct OptimizerMetadata {
    global_step: usize,
    adam_steps: HashMap<String, usize>,
}

struct PythonParamGroup {
    kind: &'static str,
    names: Vec<String>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

impl PythonParamGroup {
    fn adam(
        names: Vec<String>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        Self {
            kind: "adamw",
            names,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        }
    }

    fn muon(names: Vec<String>, lr: f64, weight_decay: f64) -> Self {
        Self {
            kind: "muon",
            names,
            lr,
            beta1: 0.0,
            beta2: 0.9,
            eps: 0.0,
            weight_decay,
        }
    }
}

fn python_group_value(group: &PythonParamGroup, ids: Vec<i64>) -> TorchValue {
    let mut entries = vec![
        (
            TorchValue::String("kind".into()),
            TorchValue::String(group.kind.into()),
        ),
        (
            TorchValue::String("params".into()),
            TorchValue::List(ids.into_iter().map(TorchValue::Integer).collect()),
        ),
        (TorchValue::String("lr".into()), TorchValue::Float(group.lr)),
        (
            TorchValue::String("initial_lr".into()),
            TorchValue::Float(group.lr),
        ),
        (
            TorchValue::String("weight_decay".into()),
            TorchValue::Float(group.weight_decay),
        ),
    ];
    if group.kind == "adamw" {
        entries.push((
            TorchValue::String("betas".into()),
            TorchValue::Tuple(vec![
                TorchValue::Float(group.beta1),
                TorchValue::Float(group.beta2),
            ]),
        ));
        entries.push((
            TorchValue::String("eps".into()),
            TorchValue::Float(group.eps),
        ));
    } else {
        entries.extend([
            (
                TorchValue::String("momentum".into()),
                TorchValue::Float(0.95),
            ),
            (
                TorchValue::String("ns_steps".into()),
                TorchValue::Integer(5),
            ),
            (
                TorchValue::String("beta2".into()),
                TorchValue::Float(group.beta2),
            ),
        ]);
    }
    TorchValue::Dict(entries)
}

fn python_adam_shard(
    state: &Tensor,
    parameter_shape: &[usize],
    world_size: usize,
    rank: usize,
) -> Result<TorchValue> {
    let mut shape = state.dims().to_vec();
    let mut values = tensor_to_cpu_f32(state)?;
    if world_size > 1 && state.elem_count() >= 1024 {
        ensure!(
            !parameter_shape.is_empty() && parameter_shape[0] % world_size == 0,
            "Python AdamW shards require the first dimension to divide by world size"
        );
        let rows = parameter_shape[0] / world_size;
        ensure!(rank < world_size, "optimizer rank is outside world size");
        let row_elements = parameter_shape[1..].iter().product::<usize>();
        if shape.first() == Some(&parameter_shape[0]) {
            let start = rank * rows * row_elements;
            values = values[start..start + rows * row_elements].to_vec();
            shape[0] = rows;
        } else {
            ensure!(
                shape.first() == Some(&rows),
                "Adam state shape {:?} is neither full nor rank-local for parameter {:?}",
                shape,
                parameter_shape
            );
        }
    }
    Ok(TorchValue::Tensor { shape, values })
}

fn tensor_to_cpu_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

/// Update an Adam moment with the same operation order as PyTorch's
/// `Tensor.lerp_(target, weight)` used by nanochat's fused optimizer.
fn adam_lerp(previous: &Tensor, target: &Tensor, weight: f64) -> Result<Tensor> {
    Ok((previous + (target - previous)?.affine(weight, 0.0)?)?)
}

fn muon_step_batch(
    parameters: &Tensor,
    gradients: &Tensor,
    momentum_buffer: &Tensor,
    second_momentum_buffer: &Tensor,
    lr: f64,
    weight_decay: f64,
    momentum_coefficient: f64,
    compute_dtype: DType,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = parameters.dims();
    ensure!(shape.len() == 3, "batched Muon expects [batch, rows, cols]");
    let (rows, cols) = (shape[1], shape[2]);
    let momentum = adam_lerp(momentum_buffer, gradients, 1.0 - momentum_coefficient)?;
    let nesterov = adam_lerp(gradients, &momentum, momentum_coefficient)?;
    let mut x = if compute_dtype == DType::BF16 {
        nesterov.to_dtype(DType::BF16)?
    } else {
        nesterov
    };

    // Match nanochat's stacked 3D Muon kernel, including one norm per matrix.
    let x_f32 = x.to_dtype(DType::F32)?;
    let row_norm = x_f32.sqr()?.sum_keepdim(2)?.sqrt()?.clamp(1e-6, f64::MAX)?;
    let target = x_f32
        .sqr()?
        .sum_keepdim(2)?
        .sum_keepdim(1)?
        .sqrt()?
        .affine(1.0 / (rows as f64).sqrt(), 0.0)?;
    x = x.broadcast_mul(&target.broadcast_div(&row_norm)?.to_dtype(x.dtype())?)?;
    let global_norm = x
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_keepdim(2)?
        .sum_keepdim(1)?
        .sqrt()?
        .to_dtype(x.dtype())?;
    x = x.broadcast_div(&global_norm.affine(1.01, 1e-6)?)?;

    let tall = rows > cols;
    for (a, b, c) in POLAR {
        let xt = x.transpose(1, 2)?;
        let (gram, next) = if tall {
            let gram = xt.matmul(&x)?;
            let gram2 = gram.matmul(&gram)?;
            let update = (gram.affine(b, 0.0)? + gram2.affine(c, 0.0)?)?;
            (gram, x.affine(a, 0.0)? + x.matmul(&update)?)
        } else {
            let gram = x.matmul(&xt)?;
            let gram2 = gram.matmul(&gram)?;
            let update = (gram.affine(b, 0.0)? + gram2.affine(c, 0.0)?)?;
            (gram, x.affine(a, 0.0)? + update.matmul(&x)?)
        };
        let _ = gram;
        x = next?;
    }

    x = x.to_dtype(parameters.dtype())?;
    let target_norm = (rows.min(cols) as f64).sqrt();
    let current_norm = x
        .sqr()?
        .sum_keepdim(2)?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-6, f64::MAX)?;
    x = x.broadcast_mul(
        &Tensor::full(target_norm, x.shape(), x.device())?
            .to_dtype(x.dtype())?
            .broadcast_div(&current_norm)?,
    )?;

    let red_dim = if rows >= cols { 2 } else { 1 };
    let red_size = if red_dim == 1 { rows } else { cols };
    let v_mean = x.sqr()?.mean_keepdim(red_dim)?;
    let v_norm = v_mean
        .sum_keepdim(2)?
        .sum_keepdim(1)?
        .affine(red_size as f64, 0.0)?
        .sqrt()?;
    let second = adam_lerp(
        second_momentum_buffer,
        &v_mean.to_dtype(second_momentum_buffer.dtype())?,
        0.1,
    )?;
    let step_size = second.clamp(1e-10, f64::MAX)?.sqrt()?.recip()?;
    let scaled = v_mean
        .affine(red_size as f64, 0.0)?
        .broadcast_mul(&step_size.sqr()?)?;
    let v_norm_new = scaled.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
    let factor = v_norm.broadcast_div(&v_norm_new.clamp(1e-10, f64::MAX)?)?;
    let update = x.broadcast_mul(&step_size.broadcast_mul(&factor)?)?;
    let same_sign = update.broadcast_mul(parameters)?.ge(0.0)?;
    let decay_masked = parameters
        .broadcast_mul(&same_sign.to_dtype(parameters.dtype())?)?
        .affine(lr * weight_decay, 0.0)?;
    let next = ((parameters - update.affine(lr, 0.0)?)? - decay_masked)?;
    Ok((next, momentum, second))
}

#[cfg(test)]
mod tests {
    use super::{MuonAdamW, MuonAdamWConfig, adam_lerp};
    use candle_core::{DType, Device, Tensor, Var, backprop::GradStore};

    #[test]
    fn adam_moment_uses_pytorch_lerp_rounding() {
        let previous = Tensor::new(&[-0.009129825816118098f32], &Device::Cpu).unwrap();
        let target = Tensor::new(&[-0.10101787042252375f32], &Device::Cpu).unwrap();
        let actual = adam_lerp(&previous, &target, 0.005)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        // PyTorch CPU float32 `lerp_` output for these inputs and weight.
        assert_eq!(actual.to_bits(), 0xbc1d_1c4c);
    }

    #[test]
    fn adamw_scalar_step_matches_pytorch_fused_cpu_fixture() {
        let parameter = Var::from_vec(vec![0.12345679f32], (1,), &Device::Cpu).unwrap();
        let mut optimizer = MuonAdamW::new(
            vec![("x0_lambdas".into(), parameter.clone())],
            768,
            MuonAdamWConfig {
                embedding_lr: 0.2,
                unembedding_lr: 0.004,
                matrix_lr: 0.02,
                scalar_lr: 0.5,
                weight_decay: 0.0,
            },
        )
        .unwrap();
        let mut gradients = GradStore::default();
        gradients.insert(
            parameter.as_tensor(),
            Tensor::new(&[1e-9f32], &Device::Cpu).unwrap(),
        );
        optimizer
            .step(&gradients, 1.0, 0.95, 1.0, DType::F32)
            .unwrap();
        let actual = parameter.as_tensor().to_vec1::<f32>().unwrap()[0];
        // PyTorch nanochat AdamW's fused one-step output for this x0 group.
        assert_eq!(actual.to_bits(), 0xbea9_8474);
    }
}

impl MuonAdamW {
    pub fn new(
        parameters: Vec<(String, Var)>,
        model_dim: usize,
        config: MuonAdamWConfig,
    ) -> Result<Self> {
        let scale = (model_dim as f64 / 768.0).powf(-0.5);
        let mut adam = Vec::new();
        let mut muon = Vec::new();
        for (name, parameter) in parameters {
            let shape = parameter.dims().to_vec();
            let (lr, beta1, beta2, wd, eps) = if name.starts_with("lm_head.") {
                (config.unembedding_lr * scale, 0.8, 0.96, 0.01, 1e-10)
            } else if name.starts_with("transformer.wte.") {
                (config.embedding_lr * scale, 0.8, 0.995, 0.001, 1e-10)
            } else if name.starts_with("value_embeds.") {
                (config.embedding_lr * scale * 0.5, 0.8, 0.995, 0.01, 1e-10)
            } else if name == "resid_lambdas" {
                (config.scalar_lr * 0.01, 0.8, 0.95, 0.05, 1e-10)
            } else if name == "x0_lambdas" {
                (config.scalar_lr, 0.96, 0.95, 0.0, 1e-10)
            } else if name == "smear_gate.weight"
                || name == "smear_lambda"
                || name == "backout_lambda"
            {
                (0.2, 0.8, 0.95, 0.0, 1e-10)
            } else if name.starts_with("transformer.h.")
                && shape.len() == 2
                && name.ends_with(".weight")
            {
                let (rows, cols) = (shape[0], shape[1]);
                let reduced = if rows >= cols {
                    vec![rows, 1]
                } else {
                    vec![1, cols]
                };
                muon.push(MuonState {
                    name,
                    momentum: Tensor::zeros(shape.clone(), DType::F32, parameter.device())?,
                    second: Tensor::zeros(reduced, DType::F32, parameter.device())?,
                    parameter,
                    lr: config.matrix_lr,
                    weight_decay: config.weight_decay,
                });
                continue;
            } else {
                bail!(
                    "unclassified trainable parameter `{name}`; refusing to silently select an optimizer"
                )
            };
            adam.push(AdamState {
                name,
                first: Tensor::zeros(shape.clone(), DType::F32, parameter.device())?,
                second: Tensor::zeros(shape, DType::F32, parameter.device())?,
                parameter,
                step: 0,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay: wd,
            });
        }
        Ok(Self {
            adam,
            muon,
            config,
            model_dim_scale: scale,
            shard_rank: 0,
            shard_world_size: 1,
        })
    }

    /// Store only this rank's ZeRO-2 optimizer state. Gradients and model
    /// parameters remain replicated by the trainer's preceding all-reduce.
    pub fn shard_state(&mut self, rank: usize, world_size: usize) -> Result<()> {
        let world_size = world_size.max(1);
        ensure!(rank < world_size, "optimizer rank is outside world size");
        if world_size == 1 {
            self.shard_rank = rank;
            self.shard_world_size = world_size;
            return Ok(());
        }
        for state in &mut self.adam {
            if state.parameter.as_tensor().elem_count() < 1024 {
                continue;
            }
            let shape = state.parameter.dims();
            ensure!(
                !shape.is_empty() && shape[0] % world_size == 0,
                "AdamW state sharding requires parameter {} first dimension to divide by world size",
                state.name
            );
            let rows = shape[0] / world_size;
            let start = rank * rows;
            state.first = state.first.narrow(0, start, rows)?.detach();
            state.second = state.second.narrow(0, start, rows)?.detach();
        }
        let owners = self.muon_owners(world_size);
        for state in &mut self.muon {
            if owners[&state.name] != rank {
                state.momentum = Tensor::zeros((0,), DType::F32, state.parameter.device())?;
                state.second = Tensor::zeros((0,), DType::F32, state.parameter.device())?;
            }
        }
        self.shard_rank = rank;
        self.shard_world_size = world_size;
        Ok(())
    }

    fn muon_owners(&self, world_size: usize) -> HashMap<String, usize> {
        let mut owners = HashMap::new();
        for group in self.python_parameter_groups() {
            if group.kind != "muon" || group.names.is_empty() {
                continue;
            }
            let chunk_size = group.names.len().div_ceil(world_size);
            for (index, name) in group.names.into_iter().enumerate() {
                owners.insert(name, index / chunk_size);
            }
        }
        owners
    }

    pub fn save_state(
        &self,
        directory: impl AsRef<Path>,
        global_step: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<()> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        let state_map = VarMap::new();
        let mut adam_steps = HashMap::new();
        {
            let mut variables = state_map
                .data()
                .lock()
                .map_err(|_| anyhow::anyhow!("optimizer state lock is poisoned"))?;
            for state in &self.adam {
                variables.insert(
                    format!("adam_first.{}", state.name),
                    Var::from_tensor(&state.first.detach())?,
                );
                variables.insert(
                    format!("adam_second.{}", state.name),
                    Var::from_tensor(&state.second.detach())?,
                );
                adam_steps.insert(state.name.clone(), state.step);
            }
            for state in &self.muon {
                variables.insert(
                    format!("muon_momentum.{}", state.name),
                    Var::from_tensor(&state.momentum.detach())?,
                );
                variables.insert(
                    format!("muon_second.{}", state.name),
                    Var::from_tensor(&state.second.detach())?,
                );
            }
        }
        let optimizer_path = if world_size > 1 {
            directory.join(format!("optimizer_rank{rank}.safetensors"))
        } else {
            directory.join("optimizer.safetensors")
        };
        let numbered_optimizer_path = if world_size > 1 {
            directory.join(format!("optimizer_{global_step:06}_rank{rank}.safetensors"))
        } else {
            directory.join(format!("optimizer_{global_step:06}.safetensors"))
        };
        state_map.save(optimizer_path)?;
        state_map.save(numbered_optimizer_path)?;
        let metadata = OptimizerMetadata {
            global_step,
            adam_steps,
        };
        let metadata_bytes = serde_json::to_vec_pretty(&metadata)?;
        let metadata_path = if world_size > 1 {
            directory.join(format!("optimizer_rank{rank}.json"))
        } else {
            directory.join("optimizer.json")
        };
        let numbered_metadata_path = if world_size > 1 {
            directory.join(format!("optimizer_{global_step:06}_rank{rank}.json"))
        } else {
            directory.join(format!("optimizer_{global_step:06}.json"))
        };
        fs::write(metadata_path, &metadata_bytes)?;
        fs::write(numbered_metadata_path, metadata_bytes)?;
        let python_state = self.python_state_dict(rank, world_size.max(1))?;
        crate::torch_save::save_torch_value(
            python_state,
            directory.join(format!("optim_{global_step:06}_rank{rank}.pt")),
        )?;
        Ok(())
    }

    fn python_state_dict(&self, rank: usize, world_size: usize) -> Result<TorchValue> {
        let groups = self.python_parameter_groups();
        let mut parameter_groups = Vec::with_capacity(groups.len());
        let mut state_entries = Vec::new();
        let mut next_id = 0i64;

        for group in &groups {
            let ids = (0..group.names.len())
                .map(|offset| next_id + offset as i64)
                .collect::<Vec<_>>();
            next_id += group.names.len() as i64;

            if group.kind == "adamw" {
                for (name, id) in group.names.iter().zip(&ids) {
                    let state = self
                        .adam
                        .iter()
                        .find(|state| state.name == *name)
                        .with_context(|| format!("missing AdamW optimizer state for {name}"))?;
                    if state.step == 0 {
                        continue;
                    }
                    let first =
                        python_adam_shard(&state.first, state.parameter.dims(), world_size, rank)?;
                    let second =
                        python_adam_shard(&state.second, state.parameter.dims(), world_size, rank)?;
                    state_entries.push((
                        TorchValue::Integer(*id),
                        TorchValue::Dict(vec![
                            (
                                TorchValue::String("step".into()),
                                TorchValue::Integer(state.step as i64),
                            ),
                            (TorchValue::String("exp_avg".into()), first),
                            (TorchValue::String("exp_avg_sq".into()), second),
                        ]),
                    ));
                }
            } else if let Some(first_id) = ids.first() {
                let chunk_size = group.names.len().div_ceil(world_size);
                let first_name = group.names.first().expect("group has first name");
                let first_state = self
                    .muon
                    .iter()
                    .find(|state| state.name == *first_name)
                    .with_context(|| format!("missing Muon optimizer state for {first_name}"))?;
                let matrix_shape = first_state.parameter.dims();
                let variance_shape = if matrix_shape[0] >= matrix_shape[1] {
                    vec![matrix_shape[0], 1]
                } else {
                    vec![1, matrix_shape[1]]
                };
                let matrix_elements = matrix_shape.iter().product::<usize>();
                let variance_elements = variance_shape.iter().product::<usize>();
                let mut momentum = vec![0.0f32; chunk_size * matrix_elements];
                let mut second = vec![0.0f32; chunk_size * variance_elements];
                for local_index in 0..chunk_size {
                    let group_index = rank * chunk_size + local_index;
                    if group_index >= group.names.len() {
                        continue;
                    }
                    let name = &group.names[group_index];
                    let state = self
                        .muon
                        .iter()
                        .find(|state| state.name == *name)
                        .with_context(|| format!("missing Muon optimizer state for {name}"))?;
                    let momentum_values = tensor_to_cpu_f32(&state.momentum)?;
                    let second_values = tensor_to_cpu_f32(&state.second)?;
                    ensure!(
                        momentum_values.len() == matrix_elements
                            && second_values.len() == variance_elements,
                        "rank-local Muon state shape mismatch within Python parameter group"
                    );
                    let momentum_start = local_index * momentum_values.len();
                    momentum[momentum_start..momentum_start + momentum_values.len()]
                        .copy_from_slice(&momentum_values);
                    let second_start = local_index * second_values.len();
                    second[second_start..second_start + second_values.len()]
                        .copy_from_slice(&second_values);
                }
                let mut momentum_shape = vec![chunk_size];
                momentum_shape.extend_from_slice(matrix_shape);
                let mut second_shape = vec![chunk_size];
                second_shape.extend_from_slice(&variance_shape);
                state_entries.push((
                    TorchValue::Integer(*first_id),
                    TorchValue::Dict(vec![
                        (
                            TorchValue::String("momentum_buffer".into()),
                            TorchValue::Tensor {
                                shape: momentum_shape,
                                values: momentum,
                            },
                        ),
                        (
                            TorchValue::String("second_momentum_buffer".into()),
                            TorchValue::Tensor {
                                shape: second_shape,
                                values: second,
                            },
                        ),
                    ]),
                ));
            }

            parameter_groups.push(python_group_value(group, ids));
        }

        Ok(TorchValue::Dict(vec![
            (
                TorchValue::String("state".into()),
                TorchValue::Dict(state_entries),
            ),
            (
                TorchValue::String("param_groups".into()),
                TorchValue::List(parameter_groups),
            ),
        ]))
    }

    pub fn load_state(
        &mut self,
        directory: impl AsRef<Path>,
        checkpoint_step: Option<usize>,
        distributed: &mut crate::distributed::DistributedContext,
    ) -> Result<Option<usize>> {
        let directory = directory.as_ref();
        let rank = distributed.rank();
        let world_size = distributed.world_size();
        let (safetensors_path, metadata_path) = if let Some(step) = checkpoint_step {
            let (numbered_safetensors, numbered_metadata) = if world_size > 1 {
                (
                    directory.join(format!("optimizer_{step:06}_rank{rank}.safetensors")),
                    directory.join(format!("optimizer_{step:06}_rank{rank}.json")),
                )
            } else {
                (
                    directory.join(format!("optimizer_{step:06}.safetensors")),
                    directory.join(format!("optimizer_{step:06}.json")),
                )
            };
            if numbered_safetensors.is_file() && numbered_metadata.is_file() {
                (numbered_safetensors, numbered_metadata)
            } else {
                let latest_metadata = if world_size > 1 {
                    directory.join(format!("optimizer_rank{rank}.json"))
                } else {
                    directory.join("optimizer.json")
                };
                if !latest_metadata.is_file() {
                    let python_path = directory.join(format!("optim_{step:06}_rank{rank}.pt"));
                    if python_path.is_file() {
                        self.load_python_state(&python_path, step, distributed)?;
                        return Ok(Some(step));
                    }
                    return Ok(None);
                }
                let latest: OptimizerMetadata =
                    serde_json::from_slice(&fs::read(&latest_metadata)?)?;
                if latest.global_step != step {
                    let python_path = directory.join(format!("optim_{step:06}_rank{rank}.pt"));
                    if python_path.is_file() {
                        self.load_python_state(&python_path, step, distributed)?;
                        return Ok(Some(step));
                    }
                    return Ok(None);
                }
                let latest_safetensors = if world_size > 1 {
                    directory.join(format!("optimizer_rank{rank}.safetensors"))
                } else {
                    directory.join("optimizer.safetensors")
                };
                (latest_safetensors, latest_metadata)
            }
        } else {
            if world_size > 1 {
                (
                    directory.join(format!("optimizer_rank{rank}.safetensors")),
                    directory.join(format!("optimizer_rank{rank}.json")),
                )
            } else {
                (
                    directory.join("optimizer.safetensors"),
                    directory.join("optimizer.json"),
                )
            }
        };
        if !safetensors_path.is_file() || !metadata_path.is_file() {
            if let Some(step) = checkpoint_step {
                let python_path = directory.join(format!("optim_{step:06}_rank{rank}.pt"));
                if python_path.is_file() {
                    self.load_python_state(&python_path, step, distributed)?;
                    return Ok(Some(step));
                }
            }
            return Ok(None);
        }
        let metadata: OptimizerMetadata =
            serde_json::from_slice(&fs::read(&metadata_path).with_context(|| {
                format!("read optimizer metadata {}", metadata_path.display())
            })?)?;
        let state_map = VarMap::new();
        {
            let mut variables = state_map
                .data()
                .lock()
                .map_err(|_| anyhow::anyhow!("optimizer state lock is poisoned"))?;
            for state in &self.adam {
                variables.insert(
                    format!("adam_first.{}", state.name),
                    Var::from_tensor(&state.first.detach())?,
                );
                variables.insert(
                    format!("adam_second.{}", state.name),
                    Var::from_tensor(&state.second.detach())?,
                );
            }
            for state in &self.muon {
                variables.insert(
                    format!("muon_momentum.{}", state.name),
                    Var::from_tensor(&state.momentum.detach())?,
                );
                variables.insert(
                    format!("muon_second.{}", state.name),
                    Var::from_tensor(&state.second.detach())?,
                );
            }
        }
        let mut loaded_map = state_map.clone();
        loaded_map.load(&safetensors_path)?;
        let variables = loaded_map
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("optimizer state lock is poisoned"))?;
        for state in &mut self.adam {
            state.first = variables[&format!("adam_first.{}", state.name)]
                .as_tensor()
                .detach();
            state.second = variables[&format!("adam_second.{}", state.name)]
                .as_tensor()
                .detach();
            state.step = *metadata
                .adam_steps
                .get(&state.name)
                .with_context(|| format!("optimizer step count missing for {}", state.name))?;
        }
        for state in &mut self.muon {
            state.momentum = variables[&format!("muon_momentum.{}", state.name)]
                .as_tensor()
                .detach();
            state.second = variables[&format!("muon_second.{}", state.name)]
                .as_tensor()
                .detach();
        }
        Ok(Some(metadata.global_step))
    }

    fn load_python_state(
        &mut self,
        path: &Path,
        checkpoint_step: usize,
        distributed: &mut crate::distributed::DistributedContext,
    ) -> Result<()> {
        let (root, archive_root) = read_torch_pickle(path)?;
        let root_entries = object_dict(&root).context("Python optimizer state is not a dict")?;
        let state_map = dict_value(root_entries, "state")
            .and_then(object_dict)
            .context("Python optimizer state has no `state` dictionary")?;
        let groups = dict_value(root_entries, "param_groups")
            .and_then(object_list)
            .context("Python optimizer state has no `param_groups` list")?;
        let expected = self.python_parameter_groups();
        ensure!(
            groups.len() == expected.len(),
            "Python optimizer has {} parameter groups, Rust expects {}",
            groups.len(),
            expected.len()
        );

        let mut python_state_by_name = HashMap::new();
        let mut python_muon_groups = Vec::new();
        for (python_group, expected_group) in groups.iter().zip(expected.iter()) {
            let entries =
                object_dict(python_group).context("Python optimizer group is not a dict")?;
            let kind = dict_value(entries, "kind")
                .and_then(object_string)
                .context("Python optimizer group has no kind")?;
            ensure!(
                kind == expected_group.kind,
                "Python optimizer group kind `{kind}` does not match Rust group `{}`",
                expected_group.kind
            );
            let ids = dict_value(entries, "params")
                .and_then(object_list)
                .context("Python optimizer group has no params list")?;
            ensure!(
                ids.len() == expected_group.names.len(),
                "Python optimizer group contains {} parameters, Rust expects {}",
                ids.len(),
                expected_group.names.len()
            );
            let ids = ids
                .iter()
                .map(|id| object_integer(id).context("Python optimizer parameter id is invalid"))
                .collect::<Result<Vec<_>>>()?;
            if expected_group.kind == "adamw" {
                for (id, name) in ids.into_iter().zip(&expected_group.names) {
                    python_state_by_name.insert(name.clone(), id);
                }
            } else {
                python_muon_groups.push((ids, expected_group.names.clone()));
            }
        }

        // Validate and load AdamW moments by their stable parameter names.
        for state in &mut self.adam {
            let Some(id) = python_state_by_name.get(&state.name) else {
                bail!("Python optimizer has no AdamW parameter `{}`", state.name);
            };
            let Some(entry) = dict_value_by_integer(state_map, *id).and_then(object_dict) else {
                continue;
            };
            let parameter_step = dict_value(entry, "step")
                .and_then(object_integer)
                .unwrap_or(checkpoint_step as i64);
            ensure!(
                parameter_step >= 0,
                "Adam step cannot be negative for {}",
                state.name
            );
            if let Some(first) = load_tensor_field(
                entry,
                "exp_avg",
                path,
                &archive_root,
                state.parameter.device(),
            )? {
                let first = restore_adam_moment(
                    first,
                    &state.first,
                    state.parameter.as_tensor(),
                    distributed,
                )?;
                ensure!(
                    first.dims() == state.first.dims(),
                    "Adam first moment shape mismatch for {}",
                    state.name
                );
                state.first = first.to_dtype(DType::F32)?.detach();
                state.step = parameter_step as usize;
            }
            if let Some(second) = load_tensor_field(
                entry,
                "exp_avg_sq",
                path,
                &archive_root,
                state.parameter.device(),
            )? {
                let second = restore_adam_moment(
                    second,
                    &state.second,
                    state.parameter.as_tensor(),
                    distributed,
                )?;
                ensure!(
                    second.dims() == state.second.dims(),
                    "Adam second moment shape mismatch for {}",
                    state.name
                );
                state.second = second.to_dtype(DType::F32)?.detach();
            }
        }

        // nanochat stores each Muon shape group's moments as a stacked tensor
        // under the first parameter ID. ZeRO shards contain one rank's padded
        // chunk; gather those chunks into Rust's replicated state.
        for (ids, names) in python_muon_groups {
            let Some(first_id) = ids.first() else {
                continue;
            };
            let Some(entry) = dict_value_by_integer(state_map, *first_id).and_then(object_dict)
            else {
                continue;
            };
            let momentum = load_tensor_field(
                entry,
                "momentum_buffer",
                path,
                &archive_root,
                self.muon
                    .first()
                    .context("Rust model has no Muon parameters")?
                    .parameter
                    .device(),
            )?
            .context("Python Muon state has no momentum_buffer")?;
            let second = load_tensor_field(
                entry,
                "second_momentum_buffer",
                path,
                &archive_root,
                self.muon
                    .first()
                    .context("Rust model has no Muon parameters")?
                    .parameter
                    .device(),
            )?
            .context("Python Muon state has no second_momentum_buffer")?;
            let chunk_size = names.len().div_ceil(distributed.world_size());
            ensure!(
                momentum.dim(0)? == chunk_size,
                "Python Muon shard has {} matrices per rank; expected {chunk_size}",
                momentum.dim(0)?
            );
            let momentum_values =
                distributed.all_gather_f32(&momentum.flatten_all()?.to_vec1::<f32>()?)?;
            let momentum = Tensor::from_vec(
                momentum_values,
                (
                    chunk_size * distributed.world_size(),
                    momentum.dim(1)?,
                    momentum.dim(2)?,
                ),
                momentum.device(),
            )?;
            let second_values =
                distributed.all_gather_f32(&second.flatten_all()?.to_vec1::<f32>()?)?;
            let second = Tensor::from_vec(
                second_values,
                (
                    chunk_size * distributed.world_size(),
                    second.dim(1)?,
                    second.dim(2)?,
                ),
                second.device(),
            )?;
            for (index, name) in names.iter().enumerate() {
                if self.shard_world_size > 1 && index / chunk_size != self.shard_rank {
                    continue;
                }
                let state = self
                    .muon
                    .iter_mut()
                    .find(|state| state.name == *name)
                    .with_context(|| format!("Rust Muon parameter `{name}` is missing"))?;
                let shard_index = index / chunk_size;
                let local_index = index % chunk_size;
                let matrix = momentum
                    .narrow(0, shard_index * chunk_size + local_index, 1)?
                    .reshape(state.momentum.shape())?
                    .to_dtype(DType::F32)?;
                let variance = second
                    .narrow(0, shard_index * chunk_size + local_index, 1)?
                    .reshape(state.second.shape())?
                    .to_dtype(DType::F32)?;
                state.momentum = matrix.detach();
                state.second = variance.detach();
            }
        }
        Ok(())
    }

    fn python_parameter_groups(&self) -> Vec<PythonParamGroup> {
        let adam_names = |predicate: &dyn Fn(&str) -> bool| {
            self.adam
                .iter()
                .filter(|state| predicate(&state.name))
                .map(|state| state.name.clone())
                .collect::<Vec<_>>()
        };
        let mut value_embeds = adam_names(&|name| name.starts_with("value_embeds."));
        value_embeds.sort_by_key(|name| layer_index(name));
        let adam = |names: Vec<String>, lr, beta1, beta2, eps, weight_decay| {
            PythonParamGroup::adam(names, lr, beta1, beta2, eps, weight_decay)
        };
        let mut groups = vec![
            adam(
                adam_names(&|name| name == "lm_head.weight"),
                self.config.unembedding_lr * self.model_dim_scale,
                0.8,
                0.96,
                1e-10,
                0.01,
            ),
            adam(
                adam_names(&|name| name == "transformer.wte.weight"),
                self.config.embedding_lr * self.model_dim_scale,
                0.8,
                0.995,
                1e-10,
                0.001,
            ),
            adam(
                value_embeds,
                self.config.embedding_lr * self.model_dim_scale * 0.5,
                0.8,
                0.995,
                1e-10,
                0.01,
            ),
            adam(
                adam_names(&|name| name == "resid_lambdas"),
                self.config.scalar_lr * 0.01,
                0.8,
                0.95,
                1e-10,
                0.05,
            ),
            adam(
                adam_names(&|name| name == "x0_lambdas"),
                self.config.scalar_lr,
                0.96,
                0.95,
                1e-10,
                0.0,
            ),
            adam(
                vec![
                    "smear_gate.weight".to_owned(),
                    "smear_lambda".to_owned(),
                    "backout_lambda".to_owned(),
                ],
                0.2,
                0.8,
                0.95,
                1e-10,
                0.0,
            ),
        ];

        let mut by_shape = BTreeMap::<(usize, usize), Vec<String>>::new();
        for state in &self.muon {
            let shape = state.parameter.dims();
            by_shape
                .entry((shape[0], shape[1]))
                .or_default()
                .push(state.name.clone());
        }
        for names in by_shape.values_mut() {
            names.sort_by_key(|name| python_matrix_order(name));
            groups.push(PythonParamGroup {
                ..PythonParamGroup::muon(
                    names.clone(),
                    self.config.matrix_lr,
                    self.config.weight_decay,
                )
            });
        }
        groups
    }

    #[cfg(test)]
    pub fn step(
        &mut self,
        gradients: &GradStore,
        lr_multiplier: f64,
        muon_momentum: f64,
        muon_decay_multiplier: f64,
        compute_dtype: DType,
    ) -> Result<()> {
        self.step_internal(
            gradients,
            lr_multiplier,
            muon_momentum,
            muon_decay_multiplier,
            compute_dtype,
            None,
        )
    }

    pub fn step_distributed(
        &mut self,
        gradients: &GradStore,
        lr_multiplier: f64,
        muon_momentum: f64,
        muon_decay_multiplier: f64,
        compute_dtype: DType,
        distributed: &mut crate::distributed::DistributedContext,
    ) -> Result<()> {
        ensure!(
            distributed.rank() == self.shard_rank
                && distributed.world_size() == self.shard_world_size,
            "optimizer shard layout does not match distributed process group"
        );
        self.step_internal(
            gradients,
            lr_multiplier,
            muon_momentum,
            muon_decay_multiplier,
            compute_dtype,
            Some(distributed),
        )
    }

    fn step_internal(
        &mut self,
        gradients: &GradStore,
        lr_multiplier: f64,
        muon_momentum: f64,
        muon_decay_multiplier: f64,
        compute_dtype: DType,
        mut distributed: Option<&mut crate::distributed::DistributedContext>,
    ) -> Result<()> {
        for state in &mut self.adam {
            let Some(grad) = gradients.get(&state.parameter) else {
                continue;
            };
            state.step += 1;
            let sharded =
                self.shard_world_size > 1 && state.parameter.as_tensor().elem_count() >= 1024;
            let mut p = state.parameter.as_tensor().to_dtype(DType::F32)?;
            let mut g = grad.to_dtype(DType::F32)?;
            if sharded {
                let rows = state.parameter.dim(0)? / self.shard_world_size;
                let start = self.shard_rank * rows;
                p = p.narrow(0, start, rows)?;
                g = g.narrow(0, start, rows)?;
            }
            let first = adam_lerp(&state.first, &g, 1.0 - state.beta1)?;
            let second = adam_lerp(&state.second, &g.sqr()?, 1.0 - state.beta2)?;
            let bias1 = 1.0 - state.beta1.powi(state.step as i32);
            let bias2 = 1.0 - state.beta2.powi(state.step as i32);
            let denominator = (second
                .affine(1.0 / bias2, 0.0)?
                .sqrt()?
                .affine(1.0, state.eps))?;
            let update = first.broadcast_div(&denominator)?;
            let decayed = p.affine(1.0 - state.lr * lr_multiplier * state.weight_decay, 0.0)?;
            let step_size = state.lr * lr_multiplier / bias1;
            let next = (decayed + update.affine(-step_size, 0.0)?)?;
            state.first = first.detach();
            state.second = second.detach();
            if sharded {
                let gathered = distributed
                    .as_deref_mut()
                    .context("sharded AdamW step requires distributed collectives")?
                    .all_gather_tensor(&next)?;
                let full = gathered
                    .reshape(state.parameter.shape())?
                    .to_dtype(state.parameter.as_tensor().dtype())?;
                state.parameter.set(&full)?;
            } else {
                state
                    .parameter
                    .set(&next.to_dtype(state.parameter.as_tensor().dtype())?)?;
            }
        }
        let muon_owners = self.muon_owners(self.shard_world_size);
        let mut muon_groups: BTreeMap<(Vec<usize>, usize), Vec<usize>> = BTreeMap::new();
        for (index, state) in self.muon.iter().enumerate() {
            muon_groups
                .entry((state.parameter.dims().to_vec(), muon_owners[&state.name]))
                .or_default()
                .push(index);
        }
        for ((parameter_shape, owner), indices) in muon_groups {
            let is_owner = self.shard_world_size == 1 || owner == self.shard_rank;
            let matrix_lr = self.muon[indices[0]].lr;
            let matrix_weight_decay = self.muon[indices[0]].weight_decay;
            let batched_update = if is_owner {
                let states = indices
                    .iter()
                    .map(|&index| &self.muon[index])
                    .collect::<Vec<_>>();
                let mut gradients_batch = Vec::with_capacity(states.len());
                let mut parameters_batch = Vec::with_capacity(states.len());
                let mut momentum_batch = Vec::with_capacity(states.len());
                let mut second_batch = Vec::with_capacity(states.len());
                for state in &states {
                    let grad = gradients
                        .get(&state.parameter)
                        .map(|grad| grad.to_dtype(DType::F32))
                        .unwrap_or_else(|| {
                            Tensor::zeros(
                                state.parameter.shape(),
                                DType::F32,
                                state.parameter.device(),
                            )
                        })?;
                    gradients_batch.push(grad);
                    parameters_batch.push(state.parameter.as_tensor().to_dtype(DType::F32)?);
                    momentum_batch.push(state.momentum.clone());
                    second_batch.push(state.second.clone());
                }
                let gradients_batch = Tensor::stack(&gradients_batch, 0)?;
                let parameters_batch = Tensor::stack(&parameters_batch, 0)?;
                let momentum_batch = Tensor::stack(&momentum_batch, 0)?;
                let second_batch = Tensor::stack(&second_batch, 0)?;
                let rows = parameter_shape[0];
                let cols = parameter_shape[1];
                let aspect_scale = (rows as f64 / cols as f64).max(1.0).sqrt();
                Some(muon_step_batch(
                    &parameters_batch,
                    &gradients_batch,
                    &momentum_batch,
                    &second_batch,
                    matrix_lr * aspect_scale * lr_multiplier,
                    matrix_weight_decay * muon_decay_multiplier,
                    muon_momentum,
                    compute_dtype,
                )?)
            } else {
                None
            };
            for (batch_index, state_index) in indices.into_iter().enumerate() {
                let state = &mut self.muon[state_index];
                let next = if let Some((next, momentum, second)) = &batched_update {
                    state.momentum = momentum
                        .narrow(0, batch_index, 1)?
                        .reshape(parameter_shape.as_slice())?
                        .detach();
                    let second_shape = state.second.dims().to_vec();
                    state.second = second
                        .narrow(0, batch_index, 1)?
                        .reshape(second_shape.as_slice())?
                        .detach();
                    Some(
                        next.narrow(0, batch_index, 1)?
                            .reshape(parameter_shape.as_slice())?,
                    )
                } else {
                    None
                };
                if self.shard_world_size > 1 {
                    let tensor = if let Some(next) = next {
                        next
                    } else {
                        Tensor::zeros(
                            state.parameter.shape(),
                            DType::F32,
                            state.parameter.device(),
                        )?
                    };
                    let gathered = distributed
                        .as_deref_mut()
                        .context("sharded Muon step requires distributed collectives")?
                        .all_gather_tensor(&tensor)?;
                    let full = gathered
                        .narrow(0, owner, 1)?
                        .reshape(state.parameter.shape())?
                        .to_dtype(state.parameter.as_tensor().dtype())?;
                    state.parameter.set(&full)?;
                } else if let Some(next) = next {
                    state
                        .parameter
                        .set(&next.to_dtype(state.parameter.as_tensor().dtype())?)?;
                }
            }
        }
        Ok(())
    }
}

fn read_torch_pickle(path: &Path) -> Result<(pickle::Object, PathBuf)> {
    let file = fs::File::open(path)
        .with_context(|| format!("open PyTorch optimizer shard {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(BufReader::new(file))
        .with_context(|| format!("open PyTorch optimizer archive {}", path.display()))?;
    let mut pickle_name = None;
    for index in 0..archive.len() {
        let entry = archive.by_index(index)?;
        let name = entry.name().to_owned();
        if name.ends_with("data.pkl") {
            pickle_name = Some(name);
            break;
        }
    }
    let pickle_name = pickle_name.context("PyTorch optimizer archive has no data.pkl")?;
    let root = PathBuf::from(
        pickle_name
            .strip_suffix("data.pkl")
            .unwrap_or(&pickle_name)
            .trim_end_matches('/'),
    );
    let mut stack = pickle::Stack::empty();
    {
        let entry = archive.by_name(&pickle_name)?;
        let mut reader = BufReader::new(entry);
        stack.read_loop(&mut reader)?;
    }
    Ok((stack.finalize()?, root))
}

fn restore_adam_moment(
    shard: Tensor,
    template: &Tensor,
    parameter: &Tensor,
    distributed: &mut crate::distributed::DistributedContext,
) -> Result<Tensor> {
    if shard.dims() == template.dims() {
        return Ok(shard.to_dtype(DType::F32)?.detach());
    }
    ensure!(
        distributed.world_size() > 1
            && shard.rank() == template.rank()
            && shard.rank() > 0
            && shard.dim(0)? * distributed.world_size() == template.dim(0)?
            && shard.dims()[1..] == template.dims()[1..]
            && parameter.elem_count() >= 1024,
        "Python Adam moment shape {:?} does not match parameter shape {:?}",
        shard.dims(),
        template.dims()
    );
    let gathered = distributed.all_gather_f32(&shard.flatten_all()?.to_vec1::<f32>()?)?;
    Ok(
        Tensor::from_vec(gathered, template.shape().clone(), template.device())?
            .to_dtype(DType::F32)?
            .detach(),
    )
}

fn load_tensor_field(
    entries: &[(pickle::Object, pickle::Object)],
    field: &str,
    archive_path: &Path,
    archive_root: &Path,
    device: &Device,
) -> Result<Option<Tensor>> {
    let Some(value) = dict_value(entries, field) else {
        return Ok(None);
    };
    let Some(info) = value
        .clone()
        .into_tensor_info(pickle::Object::Unicode(field.to_owned()), archive_root)?
    else {
        return Ok(None);
    };
    ensure!(
        info.layout.is_contiguous() || info.layout.is_fortran_contiguous(),
        "PyTorch optimizer tensor `{field}` has an unsupported strided layout"
    );
    let file = fs::File::open(archive_path)?;
    let mut archive = zip::ZipArchive::new(BufReader::new(file))?;
    let storage_path = if archive.file_names().any(|name| name == info.path) {
        info.path.clone()
    } else {
        let storage_id = info
            .path
            .rsplit('/')
            .next()
            .context("PyTorch tensor storage has no identifier")?;
        archive_root
            .join("data")
            .join(storage_id)
            .to_string_lossy()
            .replace('\\', "/")
    };
    let mut reader = archive
        .by_name(&storage_path)
        .with_context(|| format!("optimizer tensor storage `{storage_path}` is missing"))?;
    let start_offset = info.layout.start_offset();
    if start_offset > 0 {
        std::io::copy(
            &mut reader.by_ref().take(start_offset as u64),
            &mut std::io::sink(),
        )?;
    }
    let byte_count = info
        .layout
        .shape()
        .elem_count()
        .checked_mul(info.dtype.size_in_bytes())
        .context("PyTorch optimizer tensor byte size overflow")?;
    let mut bytes = vec![0u8; byte_count];
    reader.read_exact(&mut bytes)?;
    let mut tensor = Tensor::from_raw_buffer(&bytes, info.dtype, info.layout.dims(), device)?;
    if info.layout.is_fortran_contiguous() && info.layout.shape().rank() > 1 {
        let rank = info.layout.shape().rank();
        let reversed_shape = info.layout.dims().iter().rev().copied().collect::<Vec<_>>();
        let reversed = tensor.reshape(reversed_shape)?;
        let axes = (0..rank).rev().collect::<Vec<_>>();
        tensor = reversed.permute(axes)?;
    }
    Ok(Some(tensor.to_device(device)?))
}

fn object_dict(value: &pickle::Object) -> Option<&[(pickle::Object, pickle::Object)]> {
    match value {
        pickle::Object::Dict(entries) => Some(entries),
        _ => None,
    }
}

fn object_list(value: &pickle::Object) -> Option<&[pickle::Object]> {
    match value {
        pickle::Object::List(items) | pickle::Object::Tuple(items) => Some(items),
        _ => None,
    }
}

fn dict_value<'a>(
    entries: &'a [(pickle::Object, pickle::Object)],
    key: &str,
) -> Option<&'a pickle::Object> {
    entries
        .iter()
        .find_map(|(entry_key, value)| match entry_key {
            pickle::Object::Unicode(name) if name == key => Some(value),
            _ => None,
        })
}

fn dict_value_by_integer<'a>(
    entries: &'a [(pickle::Object, pickle::Object)],
    key: i64,
) -> Option<&'a pickle::Object> {
    entries.iter().find_map(|(entry_key, value)| {
        let candidate = match entry_key {
            pickle::Object::Int(value) => i64::from(*value),
            pickle::Object::Long(value) => *value,
            _ => return None,
        };
        (candidate == key).then_some(value)
    })
}

fn object_string(value: &pickle::Object) -> Option<&str> {
    match value {
        pickle::Object::Unicode(text) => Some(text),
        _ => None,
    }
}

fn object_integer(value: &pickle::Object) -> Option<i64> {
    match value {
        pickle::Object::Int(number) => Some(i64::from(*number)),
        pickle::Object::Long(number) => Some(*number),
        _ => None,
    }
}

fn layer_index(name: &str) -> usize {
    name.strip_prefix("value_embeds.")
        .and_then(|tail| tail.split('.').next())
        .and_then(|layer| layer.parse().ok())
        .unwrap_or(usize::MAX)
}

fn python_matrix_order(name: &str) -> (usize, usize) {
    let layer = name
        .strip_prefix("transformer.h.")
        .and_then(|tail| tail.split('.').next())
        .and_then(|layer| layer.parse().ok())
        .unwrap_or(usize::MAX);
    let order = if name.contains(".attn.c_q.") {
        0
    } else if name.contains(".attn.c_k.") {
        1
    } else if name.contains(".attn.c_v.") {
        2
    } else if name.contains(".attn.c_proj.") {
        3
    } else if name.contains(".attn.ve_gate.") {
        4
    } else if name.contains(".mlp.c_fc.") {
        5
    } else if name.contains(".mlp.c_proj.") {
        6
    } else {
        usize::MAX
    };
    (layer, order)
}
