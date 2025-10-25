# Mathematical Foundations

This section covers all the mathematical concepts you need to understand LLMs. We'll start from basics and build up to the complex operations used in modern Transformers.

## 1. Linear Algebra Essentials

### 1.1 Vectors and Matrices

**Vectors** are lists of numbers. In deep learning, we use vectors to represent:
- Word embeddings: `[0.2, -0.5, 0.8, ...]`
- Hidden states: representations of tokens at each layer

**Matrices** are 2D arrays of numbers. We use them for:
- Linear transformations: $y = Wx + b$
- Attention scores
- Weight parameters

**Notation:**
- Vectors: lowercase bold $\mathbf{x} \in \mathbb{R}^{d}$
- Matrices: uppercase bold $\mathbf{W} \in \mathbb{R}^{m \times n}$
- Scalars: regular letters $a, b, c$

### 1.2 Matrix Multiplication

The fundamental operation in neural networks.

Given $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$:

$$\mathbf{C} = \mathbf{A}\mathbf{B} \quad \text{where} \quad C_{ij} = \sum_{k=1}^{K} A_{ik} B_{kj}$$

**Example in Python:**
```python
import torch

A = torch.randn(3, 4)  # 3×4 matrix
B = torch.randn(4, 5)  # 4×5 matrix
C = A @ B              # 3×5 matrix (@ is matrix multiplication)
```

**Computational Cost:** $O(m \times n \times k)$ operations

### 1.3 Dot Product

The dot product of two vectors measures their similarity:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{d} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_d b_d$$

**Geometric Interpretation:**
$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)$$

where $\theta$ is the angle between vectors.

**Key Properties:**
- If $\mathbf{a} \cdot \mathbf{b} > 0$: vectors point in similar directions
- If $\mathbf{a} \cdot \mathbf{b} = 0$: vectors are orthogonal (perpendicular)
- If $\mathbf{a} \cdot \mathbf{b} < 0$: vectors point in opposite directions

### 1.4 Norms

The **L2 norm** (Euclidean norm) measures vector magnitude:

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{d} x_i^2}$$

**Normalization** scales a vector to unit length:

$$\text{normalize}(\mathbf{x}) = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$

This is used in RMSNorm and QK normalization.

## 2. Probability and Information Theory

### 2.1 Probability Distributions

A **probability distribution** $p(x)$ assigns probabilities to outcomes:
- $p(x) \geq 0$ for all $x$
- $\sum_x p(x) = 1$ (discrete) or $\int p(x)dx = 1$ (continuous)

**Language Modeling** is about learning $p(\text{next word} | \text{previous words})$.

### 2.2 Conditional Probability

Given events $A$ and $B$:

$$p(A|B) = \frac{p(A \cap B)}{p(B)}$$

In language models, we compute:

$$p(\text{sentence}) = p(w_1) \cdot p(w_2|w_1) \cdot p(w_3|w_1, w_2) \cdots$$

### 2.3 Cross-Entropy Loss

Cross-entropy measures the difference between two probability distributions.

For a true distribution $q$ and predicted distribution $p$:

$$H(q, p) = -\sum_{x} q(x) \log p(x)$$

**In language modeling:**
- $q$ is the true distribution (1 for correct token, 0 for others)
- $p$ is our model's predicted probability distribution

This simplifies to:

$$\mathcal{L} = -\log p(\text{correct token})$$

**Example:**
```python
# Suppose vocabulary size = 4, correct token = 2
logits = torch.tensor([2.0, 1.0, 3.0, 0.5])  # Model outputs
target = 2  # Correct token

# Compute cross-entropy
loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]))
# This is: -log(softmax(logits)[2])
```

### 2.4 KL Divergence

Kullback-Leibler divergence measures how one distribution differs from another:

$$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

Properties:
- $D_{KL}(p \| q) \geq 0$ always
- $D_{KL}(p \| q) = 0$ if and only if $p = q$
- Not symmetric: $D_{KL}(p \| q) \neq D_{KL}(q \| p)$

Used in some advanced training techniques like KL-regularized RL.

## 3. Calculus and Optimization

### 3.1 Derivatives

The derivative measures how a function changes:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Partial derivatives** for functions of multiple variables:

$$\frac{\partial f}{\partial x_i}$$

measures change with respect to $x_i$ while holding other variables constant.

### 3.2 Gradient

The **gradient** is the vector of all partial derivatives:

$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]$$

The gradient points in the direction of steepest increase.

### 3.3 Chain Rule

For composite functions $f(g(x))$:

$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

**Backpropagation** is just repeated application of the chain rule!

### 3.4 Gradient Descent

To minimize a function $\mathcal{L}(\theta)$:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

where:
- $\theta$: parameters
- $\eta$: learning rate
- $\nabla_\theta \mathcal{L}$: gradient of loss with respect to parameters

**Stochastic Gradient Descent (SGD)**: Use a small batch of data to estimate gradient.

## 4. Neural Network Operations

### 4.1 Linear Transformation

$$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

where:
- $\mathbf{x} \in \mathbb{R}^{d_{in}}$: input
- $\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}$: weight matrix
- $\mathbf{b} \in \mathbb{R}^{d_{out}}$: bias vector (often omitted in modern architectures)
- $\mathbf{y} \in \mathbb{R}^{d_{out}}$: output

**In PyTorch:**
```python
linear = nn.Linear(d_in, d_out, bias=False)
y = linear(x)
```

### 4.2 Activation Functions

Activation functions introduce non-linearity.

**ReLU (Rectified Linear Unit):**
$$\text{ReLU}(x) = \max(0, x)$$

**Squared ReLU (used in nanochat):**
$$\text{ReLU}^2(x) = \max(0, x)^2$$

**GELU (Gaussian Error Linear Unit):**
$$\text{GELU}(x) = x \cdot \Phi(x)$$
where $\Phi$ is the Gaussian CDF.

**Tanh:**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

### 4.3 Softmax

Converts logits to a probability distribution:

$$\text{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

Properties:
- Outputs sum to 1: $\sum_i \text{softmax}(\mathbf{x})_i = 1$
- All outputs in $(0, 1)$
- Higher input values get higher probabilities

**Temperature scaling:**
$$\text{softmax}(\mathbf{x}/T)_i = \frac{e^{x_i/T}}{\sum_{j=1}^{n} e^{x_j/T}}$$

- Higher $T$: more uniform distribution (more random)
- Lower $T$: more peaked distribution (more deterministic)

### 4.4 Layer Normalization

**LayerNorm** normalizes activations:

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where:
- $\mu = \frac{1}{d}\sum_i x_i$: mean
- $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$: variance
- $\gamma, \beta$: learnable parameters
- $\epsilon$: small constant for numerical stability

**RMSNorm (used in nanochat)** is simpler:

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}$$

No learnable parameters, just normalization!

**Implementation:**
```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

## 5. Attention Mechanism Mathematics

### 5.1 Scaled Dot-Product Attention

The core of the Transformer:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $Q \in \mathbb{R}^{T \times d_k}$: Queries
- $K \in \mathbb{R}^{T \times d_k}$: Keys
- $V \in \mathbb{R}^{T \times d_v}$: Values
- $d_k$: dimension of keys/queries

**Step by step:**

1. **Compute similarity scores:** $S = QK^T \in \mathbb{R}^{T \times T}$
   - $S_{ij}$ = how much query $i$ attends to key $j$

2. **Scale:** $S' = S / \sqrt{d_k}$
   - Prevents gradients from vanishing/exploding

3. **Softmax:** $A = \text{softmax}(S')$
   - Convert to probabilities (each row sums to 1)

4. **Weighted sum:** $\text{Output} = AV$
   - Aggregate values weighted by attention

**Why scaling by $\sqrt{d_k}$?**

For random vectors, $QK^T$ has variance $\propto d_k$. Scaling keeps variance stable, preventing softmax saturation.

### 5.2 Multi-Head Attention

Split into multiple "heads" for different representation subspaces:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where each head is:

$$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

**Parameters:**
- $W^Q_i, W^K_i, W^V_i \in \mathbb{R}^{d_{model} \times d_k}$: projection matrices
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$: output projection

### 5.3 Causal Masking

For autoregressive language models, we must prevent attending to future tokens:

$$\text{mask}_{ij} = \begin{cases}
0 & \text{if } i < j \\
-\infty & \text{if } i \geq j
\end{cases}$$

Add mask before softmax:
$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)$$

The $-\infty$ values become 0 after softmax.

## 6. Positional Encodings

Transformers have no inherent notion of position. We add positional information.

### 6.1 Sinusoidal Positional Encoding (Original Transformer)

$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

### 6.2 Rotary Position Embeddings (RoPE)

**Used in nanochat!** Encode position by rotating key/query vectors:

$$\mathbf{q}_m = R_m \mathbf{q}, \quad \mathbf{k}_n = R_n \mathbf{k}$$

where $R_\theta$ is a rotation matrix. The dot product $\mathbf{q}_m^T \mathbf{k}_n$ depends only on relative position $m-n$.

**For 2D case:**
$$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**Implementation:**
```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

## 7. Optimization Algorithms

### 7.1 Momentum

Accumulates past gradients for smoother updates:

$$v_t = \beta v_{t-1} + (1-\beta) g_t$$
$$\theta_t = \theta_{t-1} - \eta v_t$$

where:
- $g_t = \nabla \mathcal{L}(\theta_{t-1})$: current gradient
- $v_t$: velocity (exponential moving average of gradients)
- $\beta$: momentum coefficient (typically 0.9)

### 7.2 Adam/AdamW

Adaptive learning rates for each parameter:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**AdamW** adds weight decay:
$$\theta_t = (1-\lambda)\theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Typical values:
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$ (sometimes 0.95 for LLMs)
- $\epsilon = 10^{-8}$
- $\lambda = 0.01$ (weight decay)

### 7.3 Learning Rate Schedules

**Warmup:** Gradually increase LR at the start
$$\eta_t = \eta_{max} \cdot \min(1, t/T_{warmup})$$

**Cosine decay:**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t\pi}{T_{max}}\right)\right)$$

**Linear warmup + cosine decay** (common for LLMs)

## 8. Information Theory for LLMs

### 8.1 Entropy

Measures uncertainty in a distribution:

$$H(p) = -\sum_x p(x) \log_2 p(x)$$

Units: **bits** (with $\log_2$) or **nats** (with $\ln$)

### 8.2 Perplexity

Perplexity is the exponentiated cross-entropy:

$$\text{PPL} = 2^{H(q,p)} = \exp(H(q,p))$$

Interpretation: "effective vocabulary size" - how many choices the model is uncertain between.

Lower perplexity = better model.

### 8.3 Bits Per Byte (BPB)

For byte-level tokenization:

$$\text{BPB} = \frac{H(q,p)}{\log_2(256)}$$

Measures how many bits needed to encode each byte. Used in nanochat for evaluation.

## Summary: Key Equations

| Concept | Equation |
|---------|----------|
| **Linear layer** | $y = Wx + b$ |
| **Softmax** | $\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$ |
| **Cross-entropy** | $\mathcal{L} = -\sum_i y_i \log(\hat{y}_i)$ |
| **Attention** | $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ |
| **RMSNorm** | $\text{RMS}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2}}$ |
| **Gradient descent** | $\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}$ |

## Next Steps

Now that we have the mathematical foundations, we'll dive into **Tokenization** - how we convert text into numbers that the model can process.
