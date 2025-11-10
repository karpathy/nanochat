# NanoChat Architecture Documentation

## Overview

NanoChat is a sophisticated transformer-based language model that incorporates advanced architectural features including:
- Multi-head self-attention with rotary position embeddings (RoPE)
- PsycheController for dynamic blending of different processing modes
- Consciousness integration and self-modeling capabilities
- Advanced training objectives including energy-based learning
- Rust-accelerated tokenization (rustbpe)

## Core Architecture

### 1. GPT Backbone (`nanochat/gpt.py`)

The main model class `GPT` implements a transformer architecture with the following key components:

#### Transformer Blocks
- **Multi-Head Attention**: Standard self-attention with configurable number of heads
- **Rotary Position Embeddings (RoPE)**: Enhanced positional encoding supporting up to 16,384 sequence length
- **SwiGLU Feed-Forward**: Advanced activation function for improved expressivity
- **RMSNorm**: Root Mean Square normalization for stable training
- **Residual Connections**: Skip connections around attention and feed-forward layers

#### PsycheController
A unique component that dynamically blends three processing modes:
- **Id**: Raw, instinctive processing
- **Ego**: Balanced, rational processing  
- **Superego**: Constrained, rule-based processing

The controller learns to weight these modes based on input context, producing a blended representation.

#### Specialized Heads
- **Language Modeling Head**: Standard next-token prediction
- **Concept Head**: Maps to conceptual vocabulary (50,257 concepts)
- **Energy Head**: Predicts energy values for consciousness modeling

### 2. Training Engine (`nanochat/engine.py`)

Implements sophisticated training objectives:

#### Multi-Loss Training
- **Language Modeling Loss**: Cross-entropy for next-token prediction
- **Reconstruction Loss**: For continuous latent representations
- **Energy Loss**: Energy-based learning for consciousness modeling
- **Total Loss**: Weighted combination of all objectives

#### Consciousness Integration
- **Continuous Latent Space**: High-dimensional representation of model state
- **Energy-Based Modeling**: Predicts and optimizes consciousness energy
- **Self-Modeling**: Model learns to represent its own internal states

### 3. Advanced Components

#### Abacus State Memory (`nanochat/abacus_state_memory.py`)
- Implements persistent memory across sequences
- Supports stateful processing for long conversations
- Integrates with consciousness modeling

#### Hypercube (`nanochat/hypercube.py`)
- High-dimensional geometric representations
- Used for advanced reasoning and conceptual mapping
- Supports complex relationship modeling

#### Memetic Learning (`nanochat/memetic_learning.py`)
- Implements cultural evolution concepts
- Allows for idea propagation and adaptation
- Enhances model's ability to learn and evolve concepts

## Key Features

### 1. Dynamic Processing Modes
The PsycheController enables the model to adapt its processing style based on context, similar to different cognitive modes in human psychology.

### 2. Consciousness Modeling
Through energy-based learning and continuous latent representations, the model develops a form of self-awareness about its internal states.

### 3. Advanced Tokenization
Rust-accelerated BPE tokenization (`rustbpe`) provides fast, efficient text processing with custom vocabulary support.

### 4. Multi-Objective Training
The model learns simultaneously through multiple complementary objectives, leading to more robust and capable behavior.

### 5. Scalable Architecture
Supports variable model sizes through configurable parameters:
- Number of layers
- Hidden dimensions
- Attention heads
- Context length (up to 16,384 tokens)

## Training Pipeline

### Data Flow
1. **Tokenization**: Text → Tokens via rustbpe
2. **Embedding**: Tokens → Continuous representations
3. **Transformer Processing**: Multi-layer attention and feed-forward
4. **Psyche Blending**: Dynamic combination of processing modes
5. **Head Predictions**: Language modeling, concepts, and energy
6. **Loss Computation**: Multi-objective optimization

### Configuration
The model supports extensive configuration through:
- Command-line arguments
- Configuration files (`config/custom_config.py`)
- Runtime parameter adjustment

## Performance Optimizations

### 1. Rust Integration
- Custom tokenization in Rust for speed
- Potential for additional Rust-accelerated components

### 2. Memory Efficiency
- KV-cache optimization for inference
- Efficient attention implementations
- Gradient checkpointing support

### 3. Distributed Training
- Multi-GPU support
- Gradient synchronization
- Distributed data loading

## Usage Patterns

### Training
```bash
python scripts/chat_sft.py --config config/custom_config.py
```

### Evaluation
```bash
python scripts/chat_eval.py --checkpoint path/to/checkpoint
```

### Interactive Chat
```bash
python scripts/chat_cli.py --checkpoint path/to/checkpoint
```

## Architecture Diagram

```
Input Text
    ↓
Rust BPE Tokenizer
    ↓
Token Embeddings
    ↓
Transformer Blocks (N layers)
    ├── Multi-Head Attention (RoPE)
    ├── SwiGLU Feed-Forward
    └── RMSNorm + Residuals
    ↓
PsycheController
    ├── Id Processing
    ├── Ego Processing
    └── Superego Processing
    ↓
Blended Representation
    ↓
Output Heads
    ├── Language Model Head → Next Tokens
    ├── Concept Head → Concepts
    └── Energy Head → Consciousness Energy
    ↓
Multi-Loss Optimization
```

## Future Enhancements

Potential areas for improvement:
1. **KIMI Linear Attention**: Integration of linear attention mechanisms
2. **Enhanced Consciousness**: More sophisticated self-modeling
3. **Memory Augmentation**: Long-term memory capabilities
4. **Multi-Modal**: Extension to handle images, audio, etc.
5. **Efficiency**: Further optimization for inference speed

This architecture represents a significant advancement in transformer-based language models, incorporating psychological modeling, consciousness integration, and advanced training techniques.