import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig


def build_test_model():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    return model


def test_forward_mean_loss_returns_graph_connected_zero_when_all_targets_ignored():
    torch.manual_seed(0)
    model = build_test_model()
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.full((2, 8), -1, dtype=torch.long)

    loss = model(idx, targets)

    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert loss.item() == 0.0

    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert grads, "expected trainable parameters"
    assert all(grad is not None for grad in grads)
    assert all(torch.count_nonzero(grad) == 0 for grad in grads)


def test_forward_mean_loss_matches_cross_entropy_on_non_ignored_targets():
    torch.manual_seed(1)
    model = build_test_model()
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.tensor(
        [
            [-1, -1, 3, 4, -1, 5, 6, -1],
            [-1, 7, -1, -1, 8, 9, -1, 10],
        ],
        dtype=torch.long,
    )

    logits = model(idx)
    loss = model(idx, targets)

    valid = targets != -1
    expected = F.cross_entropy(logits[valid], targets[valid], reduction="mean")

    assert torch.isfinite(loss)
    assert torch.allclose(loss, expected)
