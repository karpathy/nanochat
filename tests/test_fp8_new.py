import torch
import torch.nn as nn
import pytest

import nanochat.fp8 as fp8


def test_to_fp8_and_col_major():
    x = torch.tensor([[0.0, 1.0], [2.0, -3.0]], dtype=torch.float32)
    x_fp8, inv = fp8._to_fp8(x, torch.float8_e4m3fn)
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert inv.ndim == 0

    cm = fp8._to_col_major(torch.arange(6, dtype=torch.float32).view(2, 3))
    assert cm.shape == (2, 3)
    assert torch.equal(cm, torch.arange(6, dtype=torch.float32).view(2, 3))


def test_float8_matmul_forward_backward(monkeypatch):
    def fake_scaled_mm(a, b, scale_a=None, scale_b=None, out_dtype=None, use_fast_accum=None):
        del scale_a, scale_b, use_fast_accum
        out = a.float() @ b.float()
        return out.to(out_dtype if out_dtype is not None else out.dtype)

    monkeypatch.setattr(fp8.torch, "_scaled_mm", fake_scaled_mm)

    x = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    w = torch.randn(5, 4, dtype=torch.float32, requires_grad=True)
    y = fp8._Float8Matmul.apply(x, w)
    assert y.shape == (3, 5)
    y.sum().backward()
    assert x.grad is not None
    assert w.grad is not None


def test_float8_linear_and_config(monkeypatch):
    monkeypatch.setattr(fp8._Float8Matmul, "apply", lambda input_2d, weight: input_2d.float() @ weight.float().t())
    layer = fp8.Float8Linear(4, 3, bias=True)
    inp = torch.randn(2, 5, 4)
    out = layer(inp)
    assert out.shape == (2, 5, 3)

    # Autocast branch.
    monkeypatch.setattr(fp8.torch, "is_autocast_enabled", lambda: True)
    monkeypatch.setattr(fp8.torch, "get_autocast_gpu_dtype", lambda: torch.float16)
    out2 = layer(inp)
    assert out2.shape == (2, 5, 3)

    layer2 = fp8.Float8Linear(4, 3, bias=False)
    out3 = layer2(inp)
    assert out3.shape == (2, 5, 3)

    src = nn.Linear(4, 3, bias=True)
    converted = fp8.Float8Linear.from_float(src)
    assert isinstance(converted, fp8.Float8Linear)
    assert converted.weight is src.weight
    assert converted.bias is src.bias

    assert isinstance(fp8.Float8LinearConfig.from_recipe_name("tensorwise"), fp8.Float8LinearConfig)
    with pytest.raises(ValueError):
        fp8.Float8LinearConfig.from_recipe_name("rowwise")


def test_convert_to_float8_training(monkeypatch):
    monkeypatch.setattr(fp8.Float8Linear, "from_float", classmethod(lambda cls, mod: cls(mod.in_features, mod.out_features, bias=(mod.bias is not None))))

    m = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1)),
    )
    out = fp8.convert_to_float8_training(m)
    assert isinstance(out[0], fp8.Float8Linear)
    assert isinstance(out[2][0], fp8.Float8Linear)
    assert isinstance(out[2][1], fp8.Float8Linear)

    m2 = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    out2 = fp8.convert_to_float8_training(
        m2,
        module_filter_fn=lambda _mod, fqn: fqn.endswith("0"),
    )
    assert isinstance(out2[0], fp8.Float8Linear)
    assert isinstance(out2[1], nn.Linear)
