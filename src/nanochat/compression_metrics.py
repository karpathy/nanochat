"""
Compression-based metrics for training optimization.

This module implements information compression metrics to track training
efficiency, predict overfitting, and evaluate dataset quality.

Based on the principle that intelligence emerges from information compression
through pattern discovery and unification.
"""

import torch
import numpy as np
from collections import Counter
import gzip
from typing import Dict, List, Optional, Tuple


class CompressionMetrics:
    """Track information compression during training."""
    
    def __init__(self, vocab_size: int):
        """
        Initialize compression metrics tracker.
        
        Args:
            vocab_size: Size of the vocabulary
        """
        self.vocab_size = vocab_size
        self.history: List[Dict[str, float]] = []
    
    def compute_entropy(self, tokens: torch.Tensor) -> float:
        """
        Compute Shannon entropy of token distribution.
        
        H(X) = -Σ p(x) log₂ p(x)
        
        Args:
            tokens: Token tensor of shape (B, T)
            
        Returns:
            Entropy in bits
        """
        flat_tokens = tokens.flatten().cpu().numpy()
        counts = Counter(flat_tokens)
        total = len(flat_tokens)
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def compute_conditional_entropy(
        self, 
        tokens: torch.Tensor, 
        logits: torch.Tensor
    ) -> float:
        """
        Compute conditional entropy H(X|Model).
        
        Uses cross-entropy as upper bound on conditional entropy.
        
        Args:
            tokens: Target tokens of shape (B, T)
            logits: Model predictions of shape (B, T, V)
            
        Returns:
            Conditional entropy in bits (base 2)
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Gather log probabilities of actual tokens
        B, T = tokens.shape
        token_log_probs = log_probs.gather(
            dim=-1, 
            index=tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # Convert from nats to bits and return average
        conditional_entropy = -token_log_probs.mean().item() / np.log(2)
        
        return conditional_entropy
    
    def compute_compression_ratio(
        self, 
        tokens: torch.Tensor, 
        logits: torch.Tensor
    ) -> float:
        """
        Compute compression ratio: H(X) / H(X|Model).
        
        Higher ratio = better compression = better model.
        
        Args:
            tokens: Target tokens of shape (B, T)
            logits: Model predictions of shape (B, T, V)
            
        Returns:
            Compression ratio (dimensionless)
        """
        original_entropy = self.compute_entropy(tokens)
        conditional_entropy = self.compute_conditional_entropy(tokens, logits)
        
        if conditional_entropy < 1e-6:
            return float('inf')
        
        compression_ratio = original_entropy / conditional_entropy
        return compression_ratio
    
    def compute_gzip_compression(self, tokens: torch.Tensor) -> float:
        """
        Compute actual gzip compression ratio.
        
        Args:
            tokens: Token tensor of shape (B, T)
            
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        token_bytes = tokens.cpu().numpy().tobytes()
        compressed = gzip.compress(token_bytes, compresslevel=9)
        
        ratio = len(token_bytes) / len(compressed)
        return ratio
    
    def compute_pattern_diversity(
        self, 
        activations: torch.Tensor,
        window_size: int = 5
    ) -> float:
        """
        Count unique activation patterns (n-grams).
        
        Args:
            activations: Activation tensor of shape (B, T, C)
            window_size: Size of n-gram window
            
        Returns:
            Diversity score (unique patterns / total patterns)
        """
        # Quantize activations to discrete values
        quantized = (activations * 100).long()
        
        B, T, C = quantized.shape
        ngrams = set()
        
        # Sample first 10 channels to keep computation tractable
        sample_channels = min(10, C)
        
        for b in range(B):
            for t in range(T - window_size + 1):
                ngram = tuple(
                    quantized[b, t:t+window_size, :sample_channels]
                    .flatten()
                    .tolist()
                )
                ngrams.add(ngram)
        
        total_patterns = B * (T - window_size + 1)
        diversity = len(ngrams) / max(total_patterns, 1)
        
        return diversity
    
    def log_metrics(
        self,
        step: int,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        loss: float,
        activations: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Compute and log all compression metrics.
        
        Args:
            step: Current training step
            tokens: Target tokens
            logits: Model predictions
            loss: Training loss
            activations: Optional dict of layer activations
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {
            'step': step,
            'loss': loss,
            'entropy': self.compute_entropy(tokens),
            'conditional_entropy': self.compute_conditional_entropy(tokens, logits),
            'compression_ratio': self.compute_compression_ratio(tokens, logits),
            'gzip_compression': self.compute_gzip_compression(tokens),
        }
        
        # Add pattern diversity if activations provided
        if activations:
            for layer_name, acts in activations.items():
                if acts.dim() == 3:  # (B, T, C)
                    metrics[f'{layer_name}_diversity'] = self.compute_pattern_diversity(acts)
        
        # Compression efficiency: compression per unit loss
        metrics['compression_efficiency'] = metrics['compression_ratio'] / max(loss, 1e-6)
        
        self.history.append(metrics)
        return metrics
    
    def detect_overfitting(self, window: int = 100) -> bool:
        """
        Detect overfitting via compression plateau.
        
        Args:
            window: Number of steps to compare
            
        Returns:
            True if compression has plateaued (possible overfitting)
        """
        if len(self.history) < window * 2:
            return False
        
        recent = [h['compression_ratio'] for h in self.history[-window:]]
        previous = [h['compression_ratio'] for h in self.history[-2*window:-window]]
        
        recent_mean = np.mean(recent)
        previous_mean = np.mean(previous)
        
        improvement = (recent_mean - previous_mean) / previous_mean
        
        # Overfitting if compression improvement < 1%
        return improvement < 0.01
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of compression metrics.
        
        Returns:
            Dictionary with mean, std, min, max for key metrics
        """
        if not self.history:
            return {}
        
        compression_ratios = [h['compression_ratio'] for h in self.history]
        efficiencies = [h['compression_efficiency'] for h in self.history]
        
        return {
            'compression_ratio_mean': np.mean(compression_ratios),
            'compression_ratio_std': np.std(compression_ratios),
            'compression_ratio_min': np.min(compression_ratios),
            'compression_ratio_max': np.max(compression_ratios),
            'compression_efficiency_mean': np.mean(efficiencies),
            'compression_efficiency_std': np.std(efficiencies),
        }
