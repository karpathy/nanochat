"""
Configuration classes for GPT model and training.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Any
from pathlib import Path
import json


@dataclass
class GPTConfig:
    """Configuration for GPT model architecture."""
    
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    # Model architecture
    depth: int
    aspect_ratio: int = 64
    head_dim: int = 128
    max_seq_len: int = 2048
    window_pattern: str = "SSSL"
    
    # Training horizon
    num_iterations: int = -1
    target_flops: float = -1.0
    target_param_data_ratio: float = 10.5
    
    # Optimization
    device_batch_size: int = 32
    total_batch_size: int = -1
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    weight_decay: float = 0.28
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    
    # Evaluation
    eval_every: int = 250
    eval_tokens: int = 80 * 524288
    core_metric_every: int = 2000
    core_metric_max_per_task: int = 500
    sample_every: int = 2000
    save_every: int = -1
    
    # Runtime
    device_type: str = ""
    fp8: bool = False
    fp8_recipe: str = "tensorwise"
    seed: int = 42
    
    # Output
    model_tag: Optional[str] = None
    run_name: str = "dummy"
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))
    
    @classmethod
    def from_args(cls, args: Any) -> "TrainingConfig":
        """Create TrainingConfig from argparse Namespace."""
        # Extract all fields that match TrainingConfig
        config_dict = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(args, field_name):
                config_dict[field_name] = getattr(args, field_name)
        return cls(**config_dict)
