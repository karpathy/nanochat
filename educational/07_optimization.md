# Advanced Optimization Techniques

nanochat uses a **hybrid optimization** strategy: combining **Muon** for matrix parameters and **AdamW** for embeddings. This is more sophisticated than standard approaches.

## Why Different Optimizers?

Different parameter types have different optimization needs:

| Parameter Type | Examples | Characteristics | Best Optimizer |
|----------------|----------|-----------------|----------------|
| **Matrices** | Attention, MLP | Dense, high-dimensional | Muon |
| **Embeddings** | Token embeddings | Sparse updates, embedding-specific | AdamW |
| **Vectors** | LM head | Output layer, sparse | AdamW |

**Traditional approach:** Use AdamW for everything
**nanochat approach:** Use Muon for matrices, AdamW for embeddings/head

**Result:** Faster training, better convergence

## 1. Muon Optimizer

Muon is a novel optimizer designed specifically for **matrix parameters** in neural networks.

### Core Idea

Standard optimizers (SGD, Adam) treat matrices as flat vectors:
```
Matrix [3×4] → Flatten to vector [12] → Update
```

Muon exploits **matrix structure**:
```
Matrix [3×4] → Update using matrix operations → Keep matrix shape
```

### Mathematical Formulation

For weight matrix $W \in \mathbb{R}^{m \times n}$:

**Standard momentum:**
$$v_t = \beta v_{t-1} + (1-\beta) g_t$$
$$W_t = W_{t-1} - \eta v_t$$

**Muon:**
1. Compute gradient $G_t = \nabla_W \mathcal{L}$
2. Orthogonalize using Newton-Schulz iteration
3. Apply momentum in tangent space
4. Update with adaptive step size

### Implementation: `nanochat/muon.py:53`

```python
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super(Muon, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad  # Gradient

                # Get state
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']

                # Handle matrix vs non-matrix parameters
                if g.ndim == 2 and g.size(0) >= 16 and g.size(1) >= 16:
                    # Matrix parameter: use Muon update
                    g = newton_schulz_orthogonalize(g, steps=5)

                # Momentum update
                buf.mul_(momentum).add_(g)

                # Parameter update
                p.data.add_(buf, alpha=-lr)
```

### Newton-Schulz Orthogonalization: `nanochat/muon.py:16`

```python
def newton_schulz_orthogonalize(G, steps=5, eps=1e-7):
    """
    Orthogonalize gradient matrix using Newton-Schulz iteration
    """
    # Make square by padding or cropping
    a, b = G.size()
    if a > b:
        G = G[:b, :]
    elif a < b:
        G = G[:, :a]

    # Initialize
    # Normalization factor
    t = G.size(0)

    # X_0 = G / ||G||_F
    A = G / (G.norm() + eps)

    # Newton-Schulz iteration: X_{k+1} = X_k * (3I - X_k^T X_k) / 2
    for _ in range(steps):
        A_T_A = A.t() @ A
        A = A @ (1.5 * torch.eye(t, device=A.device, dtype=A.dtype) - 0.5 * A_T_A)

    # Restore original shape
    if a > b:
        A = torch.cat([A, torch.zeros(a - b, b, device=A.device, dtype=A.dtype)], dim=0)
    elif a < b:
        A = torch.cat([A, torch.zeros(a, b - a, device=A.device, dtype=A.dtype)], dim=1)

    return A
```

**What does this do?**

For a matrix $G$, find orthogonal matrix $Q$ closest to $G$:
$$Q = \arg\min_{\tilde{Q}^T\tilde{Q}=I} \|G - \tilde{Q}\|_F$$

Uses iterative formula:
$$X_{k+1} = X_k \left(\frac{3I - X_k^TX_k}{2}\right)$$

Converges to $Q = G(G^TG)^{-1/2}$ (the orthogonal component of $G$).

**Why orthogonalize?**
- Keeps gradients on Stiefel manifold
- Better geometry for optimization
- Prevents gradient explosion/vanishing
- Faster convergence

### Distributed Muon: `nanochat/muon.py:155`

For multi-GPU training:

```python
class DistMuon(Muon):
    def step(self):
        # First, average gradients across all GPUs
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

        # Then apply standard Muon update
        super().step()
```

**Key:** All-reduce gradients before Muon update ensures synchronization.

### Muon Learning Rate Scaling

```python
# From scripts/base_train.py:238
dmodel_lr_scale = (model_dim / 768) ** -0.5
lr_scaled = matrix_lr  # No scaling for Muon (handles it internally)
```

Muon is **scale-invariant**, so no need to scale LR by model dimension!

### Momentum Schedule for Muon: `scripts/base_train.py:160`

```python
def get_muon_momentum(it):
    """Warmup momentum from 0.85 to 0.95"""
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum
```

Start with lower momentum (more responsive), increase to higher momentum (more stable).

## 2. AdamW Optimizer

AdamW is used for embedding and language model head parameters.

### Standard Adam

Combines **momentum** and **adaptive learning rates**:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(first moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(second moment)}$$

Bias correction:
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

Update:
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### AdamW: Decoupled Weight Decay

**Adam with L2 regularization:**
$$\mathcal{L}' = \mathcal{L} + \frac{\lambda}{2}\|\theta\|^2$$

**Problem:** Weight decay interacts with adaptive learning rate in weird ways.

**AdamW solution:** Decouple weight decay from gradient:
$$\theta_t = (1 - \lambda \eta) \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Benefits:**
- Cleaner regularization
- Better generalization
- Less hyperparameter interaction

### Implementation: `nanochat/adamw.py:53`

```python
class DistAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        # First, all-reduce gradients across GPUs
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

        # Then apply AdamW update
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)       # m_t
                    state['exp_avg_sq'] = torch.zeros_like(p)    # v_t

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                step = state['step']

                # Update biased first and second moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1

                # Compute denominator (with bias correction)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Weight decay (decoupled)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
```

### AdamW Hyperparameters in nanochat

```python
# From scripts/base_train.py:228
adam_groups = [
    dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),  # 0.004
    dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),   # 0.2
]

adamw_kwargs = dict(
    betas=(0.8, 0.95),  # Instead of default (0.9, 0.999)
    eps=1e-10,
    weight_decay=weight_decay  # Usually 0.0 for small models
)
```

**Why different betas?**
- $\beta_1 = 0.8$: Slightly less momentum (more responsive)
- $\beta_2 = 0.95$: Much less variance accumulation (adapts faster)

This is better tuned for LLM training than defaults.

### Learning Rate Scaling by Model Dimension

```python
dmodel_lr_scale = (model_dim / 768) ** -0.5

# Example:
# model_dim = 1280 → scale = (1280/768)^{-0.5} ≈ 0.77
# model_dim = 384  → scale = (384/768)^{-0.5} ≈ 1.41
```

**Why $\propto 1/\sqrt{d_{model}}$?**

Larger models have larger gradients (sum over more dimensions). Scaling LR prevents instability.

## 3. Hybrid Optimizer Setup: `nanochat/gpt.py:228`

```python
def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
    model_dim = self.config.n_embd
    ddp, rank, _, _ = get_dist_info()

    # Separate parameters into groups
    matrix_params = list(self.transformer.h.parameters())        # All transformer blocks
    embedding_params = list(self.transformer.wte.parameters())   # Token embeddings
    lm_head_params = list(self.lm_head.parameters())            # Output layer

    # Scale learning rates
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    # AdamW for embeddings and LM head
    adam_groups = [
        dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
    ]
    AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
    adamw_optimizer = AdamWFactory(adam_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)

    # Muon for transformer matrices
    MuonFactory = DistMuon if ddp else Muon
    muon_optimizer = MuonFactory(matrix_params, lr=matrix_lr, momentum=0.95)

    # Return both optimizers
    optimizers = [adamw_optimizer, muon_optimizer]
    return optimizers
```

**Why different learning rates?**

| Parameter | LR | Reasoning |
|-----------|-----|-----------|
| Embeddings | 0.2 | Sparse updates, can handle high LR |
| LM head | 0.004 | Dense gradients, needs lower LR |
| Matrices | 0.02 | Muon handles geometry, moderate LR |

### Stepping Multiple Optimizers: `scripts/base_train.py:269`

```python
# Update learning rates for all optimizers
lrm = get_lr_multiplier(step)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["initial_lr"] * lrm

# Update Muon momentum
muon_momentum = get_muon_momentum(step)
for group in muon_optimizer.param_groups:
    group["momentum"] = muon_momentum

# Step all optimizers
for opt in optimizers:
    opt.step()

# Clear gradients
model.zero_grad(set_to_none=True)
```

**Important:** `set_to_none=True` saves memory compared to zeroing.

## 4. Gradient Clipping

Prevents exploding gradients during training.

### Global Norm Clipping: `scripts/base_train.py:265`

```python
if grad_clip > 0.0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

**How it works:**

1. Compute global gradient norm:
$$\|\mathbf{g}\|_{global} = \sqrt{\sum_{\theta \in \Theta} \|\nabla_\theta \mathcal{L}\|^2}$$

2. If too large, scale all gradients:
$$\mathbf{g}_\theta \leftarrow \frac{\text{max\_norm}}{\|\mathbf{g}\|_{global}} \mathbf{g}_\theta$$

**Effect:** Limits maximum gradient magnitude without changing direction.

### Implementation Details

```python
def clip_grad_norm_(parameters, max_norm, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    # Compute total norm
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type
    )

    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)

    # Clip if necessary
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

    return total_norm
```

## 5. Warmup and Decay Schedules

### Why Warmup?

At initialization:
- Weights are random
- Gradients can be very large
- Adam's second moment estimate is inaccurate

**Solution:** Start with low LR, gradually increase.

### Why Decay?

Near end of training:
- Model is close to optimum
- Small steps refine solution
- Prevents oscillation

**Solution:** Gradually decrease LR to 0.

### Schedule Implementation: `scripts/base_train.py:148`

```python
warmup_ratio = 0.0      # Skip warmup for simplicity
warmdown_ratio = 0.2    # Last 20% of training
final_lr_frac = 0.0     # Decay to 0

def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)

    if it < warmup_iters:
        # Linear warmup
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        # Constant LR
        return 1.0
    else:
        # Linear warmdown
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac
```

**Alternative schedules:**
- Cosine decay: Smoother than linear
- Exponential decay: Aggressive reduction
- Step decay: Discrete jumps

## 6. Optimization Best Practices

### Learning Rate Tuning

**Too high:**
- Training unstable
- Loss oscillates or diverges
- NaN loss

**Too low:**
- Training very slow
- Gets stuck in local minima
- Underfits

**Good LR:**
- Steady loss decrease
- Occasional small oscillations
- Converges smoothly

### Finding Good LR: Learning Rate Range Test

```python
# Start with very low LR, gradually increase
lrs = []
losses = []

lr = 1e-8
for step in range(1000):
    loss = train_step(lr)
    lrs.append(lr)
    losses.append(loss)
    lr *= 1.01  # Increase by 1%

# Plot losses vs LRs
# Good LR is where loss decreases fastest
```

### Batch Size Effects

**Larger batch size:**
- More stable gradients
- Better GPU utilization
- Can use higher LR
- Slower wall-clock time per iteration
- May generalize worse

**Smaller batch size:**
- Noisier gradients (implicit regularization)
- Less GPU efficient
- Lower LR needed
- Faster iterations

**nanochat choice:** 524K tokens/batch (very large for stability)

## 7. Comparison: Different Optimization Strategies

| Strategy | Training Speed | Final Loss | Complexity |
|----------|----------------|------------|------------|
| SGD | Slow | Good | Simple |
| Adam | Fast | Good | Medium |
| AdamW | Fast | Better | Medium |
| Muon (matrices only) | Very Fast | Best | High |
| **Hybrid (AdamW + Muon)** | **Very Fast** | **Best** | **High** |

nanochat's hybrid approach is cutting-edge!

## 8. Memory Optimization

### Gradient Checkpointing (Not used in nanochat)

Trade compute for memory:
- Don't store intermediate activations
- Recompute during backward pass
- 2× slower, but 10× less memory

### Optimizer State Management

AdamW stores:
- First moment (m): same size as parameters
- Second moment (v): same size as parameters

**Memory:** ~2× parameter size

For 270M param model:
- Parameters: 270M × 2 bytes (BF16) = 540 MB
- AdamW states: 270M × 8 bytes (FP32) = 2.16 GB
- Total: ~2.7 GB

### Fused Optimizers

```python
AdamW(..., fused=True)  # Uses fused CUDA kernel
```

**Benefits:**
- Faster updates (single kernel launch)
- Less memory traffic
- ~10-20% speedup

## Next Steps

We've covered optimization! Next, we'll explore **Implementation Details** - practical coding techniques used throughout nanochat.
