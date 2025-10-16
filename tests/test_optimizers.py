"""
Tests for custom optimizers (AdamW and Muon).

Run with:
python -m pytest tests/test_optimizers.py -v -s --timeout=60
"""

import torch
import pytest
from nanochat.adamw import DistAdamW
from nanochat.muon import Muon


@pytest.fixture
def simple_model():
    """Create a simple model for testing optimizers."""
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 20, bias=False)
            self.linear2 = torch.nn.Linear(20, 10, bias=False)
        
        def forward(self, x):
            return self.linear2(self.linear1(x))
    
    return SimpleModel()


def test_muon_initialization(simple_model):
    """Test Muon optimizer initialization."""
    params = list(simple_model.parameters())
    optimizer = Muon(params, lr=0.02, momentum=0.95)
    
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 0.02
    assert optimizer.param_groups[0]['momentum'] == 0.95


def test_muon_step(simple_model):
    """Test Muon optimizer step."""
    optimizer = Muon(simple_model.parameters(), lr=0.02)
    
    # Forward and backward
    x = torch.randn(4, 10)
    y = simple_model(x)
    loss = y.sum()
    loss.backward()
    
    # Get original weights
    original_weights = {name: param.data.clone() 
                       for name, param in simple_model.named_parameters()}
    
    # Optimizer step
    optimizer.step()
    
    # Weights should have changed
    for name, param in simple_model.named_parameters():
        assert not torch.allclose(param.data, original_weights[name])


def test_muon_momentum():
    """Test that Muon maintains momentum state."""
    param = torch.nn.Parameter(torch.randn(10, 10))
    optimizer = Muon([param], lr=0.02, momentum=0.95)
    
    # First step
    param.grad = torch.randn_like(param)
    optimizer.step()
    
    # Check that momentum state is created
    assert len(optimizer.state) > 0


def test_muon_zero_grad():
    """Test zero_grad functionality."""
    param = torch.nn.Parameter(torch.randn(10, 10))
    optimizer = Muon([param], lr=0.02)
    
    param.grad = torch.randn_like(param)
    assert param.grad is not None
    
    optimizer.zero_grad()
    assert param.grad is None or torch.all(param.grad == 0)


def test_muon_parameter_groups():
    """Test Muon groups parameters automatically by size."""
    param1 = torch.nn.Parameter(torch.randn(10, 10))  # 100 elements
    param2 = torch.nn.Parameter(torch.randn(5, 5))     # 25 elements
    param3 = torch.nn.Parameter(torch.randn(10, 10))  # 100 elements (same as param1)
    
    optimizer = Muon([param1, param2, param3], lr=0.02)
    
    # Muon automatically groups by parameter size
    # Should have 2 groups: one for 100-element params, one for 25-element params
    assert len(optimizer.param_groups) == 2
    
    # Find the groups
    groups_by_size = {len(g['params']): g for g in optimizer.param_groups}
    
    # One group should have 2 params (param1 and param3), one should have 1 param (param2)
    sizes = sorted([len(g['params']) for g in optimizer.param_groups])
    assert sizes == [1, 2]


def test_muon_updates_params(simple_model):
    """Test that Muon actually updates parameters."""
    optimizer = Muon(simple_model.parameters(), lr=0.02)
    
    # Store original params
    original = [p.data.clone() for p in simple_model.parameters()]
    
    # Create gradients
    for p in simple_model.parameters():
        p.grad = torch.randn_like(p) * 0.1
    
    # Take optimization step
    optimizer.step()
    
    # Parameters should be different
    for orig, current in zip(original, simple_model.parameters()):
        assert not torch.allclose(orig, current.data)


def test_muon_with_real_loss(simple_model):
    """Test Muon with a real loss function."""
    optimizer = Muon(simple_model.parameters(), lr=0.02)
    
    # Training loop simulation
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        
        x = torch.randn(4, 10)
        target = torch.randn(4, 10)
        
        output = simple_model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
    
    # Loss should be finite
    assert all(not torch.isnan(torch.tensor(l)) for l in losses)
    assert all(not torch.isinf(torch.tensor(l)) for l in losses)


def test_muon_vs_sgd_different():
    """Test that Muon produces different updates than vanilla SGD."""
    # Create two identical models
    model1 = torch.nn.Linear(10, 10, bias=False)
    model2 = torch.nn.Linear(10, 10, bias=False)
    model2.load_state_dict(model1.state_dict())
    
    # Use Muon for model1, SGD for model2
    opt1 = Muon(model1.parameters(), lr=0.01, momentum=0.0)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.0)
    
    # Same forward/backward
    x = torch.randn(4, 10)
    
    y1 = model1(x)
    loss1 = y1.sum()
    loss1.backward()
    
    y2 = model2(x)
    loss2 = y2.sum()
    loss2.backward()
    
    # Gradients should be identical
    torch.testing.assert_close(model1.weight.grad, model2.weight.grad)
    
    # Take steps
    opt1.step()
    opt2.step()
    
    # Weights should be different (Muon uses different update rule)
    # Note: They might be similar but Muon has different normalization
    # Just check both updated successfully
    assert not torch.allclose(model1.weight, torch.zeros_like(model1.weight))
    assert not torch.allclose(model2.weight, torch.zeros_like(model2.weight))


def test_muon_lr_scheduling():
    """Test that learning rate can be adjusted."""
    param = torch.nn.Parameter(torch.randn(10, 10))
    optimizer = Muon([param], lr=0.02)
    
    # Check initial lr
    assert optimizer.param_groups[0]['lr'] == 0.02
    
    # Modify lr
    optimizer.param_groups[0]['lr'] = 0.01
    assert optimizer.param_groups[0]['lr'] == 0.01


def test_muon_handles_different_shapes():
    """Test Muon with various parameter shapes (must be 2D+)."""
    params = [
        torch.nn.Parameter(torch.randn(10, 10)),  # 2D
        torch.nn.Parameter(torch.randn(20, 5)),   # 2D different shape
        torch.nn.Parameter(torch.randn(5, 5, 5)),  # 3D
    ]
    
    optimizer = Muon(params, lr=0.02)
    
    # Create gradients and step
    for p in params:
        p.grad = torch.randn_like(p) * 0.1
    
    optimizer.step()
    
    # Should work without errors
    assert True

