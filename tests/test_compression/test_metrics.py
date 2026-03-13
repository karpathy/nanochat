"""Tests for compression metrics."""

import torch
import pytest
import numpy as np
from nanochat.compression_metrics import CompressionMetrics


@pytest.fixture
def compression_tracker():
    """Create a compression metrics tracker."""
    return CompressionMetrics(vocab_size=1000)


def test_compute_entropy(compression_tracker):
    """Test entropy computation."""
    # Uniform distribution should have high entropy
    tokens_uniform = torch.randint(0, 100, (4, 128))
    entropy_uniform = compression_tracker.compute_entropy(tokens_uniform)
    
    # Concentrated distribution should have low entropy
    tokens_concentrated = torch.zeros(4, 128, dtype=torch.long)
    tokens_concentrated[:, :] = 42  # All same token
    entropy_concentrated = compression_tracker.compute_entropy(tokens_concentrated)
    
    assert entropy_uniform > entropy_concentrated
    assert entropy_concentrated == 0.0  # Single token = zero entropy
    assert entropy_uniform > 0.0


def test_compute_conditional_entropy(compression_tracker):
    """Test conditional entropy computation."""
    tokens = torch.randint(0, 100, (2, 64))
    
    # Perfect predictions (high confidence)
    logits_perfect = torch.zeros(2, 64, 1000)
    for b in range(2):
        for t in range(64):
            logits_perfect[b, t, tokens[b, t]] = 10.0  # High logit for correct token
    
    # Random predictions (low confidence)
    logits_random = torch.randn(2, 64, 1000) * 0.1
    
    cond_entropy_perfect = compression_tracker.compute_conditional_entropy(tokens, logits_perfect)
    cond_entropy_random = compression_tracker.compute_conditional_entropy(tokens, logits_random)
    
    # Perfect predictions should have lower conditional entropy
    assert cond_entropy_perfect < cond_entropy_random


def test_compute_compression_ratio(compression_tracker):
    """Test compression ratio computation."""
    tokens = torch.randint(0, 100, (2, 64))
    
    # Good model (high confidence on correct tokens)
    logits_good = torch.zeros(2, 64, 1000)
    for b in range(2):
        for t in range(64):
            logits_good[b, t, tokens[b, t]] = 5.0
    
    # Bad model (random predictions)
    logits_bad = torch.randn(2, 64, 1000) * 0.1
    
    ratio_good = compression_tracker.compute_compression_ratio(tokens, logits_good)
    ratio_bad = compression_tracker.compute_compression_ratio(tokens, logits_bad)
    
    # Good model should have higher compression ratio
    assert ratio_good > ratio_bad
    assert ratio_good > 1.0  # Should compress better than raw entropy


def test_compute_gzip_compression(compression_tracker):
    """Test gzip compression ratio."""
    # Repetitive data should compress well
    tokens_repetitive = torch.zeros(4, 128, dtype=torch.long)
    tokens_repetitive[:, :] = 42
    
    # Random data should compress poorly
    tokens_random = torch.randint(0, 1000, (4, 128))
    
    ratio_repetitive = compression_tracker.compute_gzip_compression(tokens_repetitive)
    ratio_random = compression_tracker.compute_gzip_compression(tokens_random)
    
    # Repetitive data should have higher compression ratio
    assert ratio_repetitive > ratio_random
    assert ratio_repetitive > 1.0


def test_compute_pattern_diversity(compression_tracker):
    """Test pattern diversity computation."""
    # All same activations (low diversity)
    activations_uniform = torch.ones(2, 64, 128)
    
    # Random activations (high diversity)
    activations_diverse = torch.randn(2, 64, 128)
    
    diversity_uniform = compression_tracker.compute_pattern_diversity(activations_uniform)
    diversity_diverse = compression_tracker.compute_pattern_diversity(activations_diverse)
    
    # Diverse activations should have higher diversity score
    assert diversity_diverse > diversity_uniform
    assert 0.0 <= diversity_uniform <= 1.0
    assert 0.0 <= diversity_diverse <= 1.0


def test_log_metrics(compression_tracker):
    """Test metrics logging."""
    tokens = torch.randint(0, 100, (2, 64))
    logits = torch.randn(2, 64, 1000)
    loss = 2.5
    
    metrics = compression_tracker.log_metrics(
        step=100,
        tokens=tokens,
        logits=logits,
        loss=loss
    )
    
    # Check all expected metrics are present
    assert 'step' in metrics
    assert 'loss' in metrics
    assert 'entropy' in metrics
    assert 'conditional_entropy' in metrics
    assert 'compression_ratio' in metrics
    assert 'gzip_compression' in metrics
    assert 'compression_efficiency' in metrics
    
    # Check values are reasonable
    assert metrics['step'] == 100
    assert metrics['loss'] == 2.5
    assert metrics['entropy'] > 0
    assert metrics['compression_ratio'] > 0
    
    # Check history is updated
    assert len(compression_tracker.history) == 1
    assert compression_tracker.history[0] == metrics


def test_log_metrics_with_activations(compression_tracker):
    """Test metrics logging with layer activations."""
    tokens = torch.randint(0, 100, (2, 64))
    logits = torch.randn(2, 64, 1000)
    loss = 2.5
    activations = {
        'layer_0': torch.randn(2, 64, 128),
        'layer_1': torch.randn(2, 64, 128),
    }
    
    metrics = compression_tracker.log_metrics(
        step=100,
        tokens=tokens,
        logits=logits,
        loss=loss,
        activations=activations
    )
    
    # Check layer diversity metrics are present
    assert 'layer_0_diversity' in metrics
    assert 'layer_1_diversity' in metrics
    assert 0.0 <= metrics['layer_0_diversity'] <= 1.0


def test_detect_overfitting(compression_tracker):
    """Test overfitting detection."""
    tokens = torch.randint(0, 100, (2, 64))
    
    # Simulate improving compression (no overfitting)
    for step in range(250):
        logits = torch.randn(2, 64, 1000)
        # Gradually improve predictions
        for b in range(2):
            for t in range(64):
                logits[b, t, tokens[b, t]] += step * 0.01
        
        compression_tracker.log_metrics(
            step=step,
            tokens=tokens,
            logits=logits,
            loss=2.5 - step * 0.01
        )
    
    # Should not detect overfitting when improving
    assert not compression_tracker.detect_overfitting(window=100)
    
    # Simulate plateau (overfitting)
    for step in range(250, 350):
        logits = torch.randn(2, 64, 1000)
        # No improvement
        for b in range(2):
            for t in range(64):
                logits[b, t, tokens[b, t]] += 2.5
        
        compression_tracker.log_metrics(
            step=step,
            tokens=tokens,
            logits=logits,
            loss=2.0
        )
    
    # Should detect overfitting when plateaued
    assert compression_tracker.detect_overfitting(window=50)


def test_get_summary(compression_tracker):
    """Test summary statistics."""
    tokens = torch.randint(0, 100, (2, 64))
    
    # Log some metrics
    for step in range(10):
        logits = torch.randn(2, 64, 1000)
        compression_tracker.log_metrics(
            step=step,
            tokens=tokens,
            logits=logits,
            loss=2.5
        )
    
    summary = compression_tracker.get_summary()
    
    # Check summary contains expected keys
    assert 'compression_ratio_mean' in summary
    assert 'compression_ratio_std' in summary
    assert 'compression_ratio_min' in summary
    assert 'compression_ratio_max' in summary
    assert 'compression_efficiency_mean' in summary
    assert 'compression_efficiency_std' in summary
    
    # Check values are reasonable
    assert summary['compression_ratio_mean'] > 0
    assert summary['compression_ratio_std'] >= 0
    assert summary['compression_ratio_min'] <= summary['compression_ratio_mean']
    assert summary['compression_ratio_max'] >= summary['compression_ratio_mean']


def test_empty_summary(compression_tracker):
    """Test summary with no history."""
    summary = compression_tracker.get_summary()
    assert summary == {}
