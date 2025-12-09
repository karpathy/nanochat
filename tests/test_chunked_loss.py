
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from nanochat.gpt import chunked_cross_entropy

def standard_loss(x, targets, lm_head, softcap=15.0, ignore_index=-1):
    # Standard implementation
    logits = lm_head(x)
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_index)
    return loss

def test_chunked_vs_standard_equivalence():
    torch.manual_seed(42)

    B, T, C = 2, 64, 32
    vocab_size = 100

    x = torch.randn(B, T, C)
    # Generate targets with some ignore_index
    targets = torch.randint(0, vocab_size, (B, T))
    targets[0, :5] = -1 # Set some ignore indices

    lm_head = nn.Linear(C, vocab_size, bias=False)

    # Standard Loss
    loss_std = standard_loss(x, targets, lm_head)

    # Chunked Loss (using the imported one now)
    loss_chunk = chunked_cross_entropy(x, targets, lm_head, chunk_size=16)

    print(f"Standard: {loss_std.item()}, Chunked: {loss_chunk.item()}")

    assert torch.allclose(loss_std, loss_chunk, atol=1e-5), f"Mismatch: {loss_std.item()} vs {loss_chunk.item()}"

def test_chunked_loss_grad():
    torch.manual_seed(1337)
    B, T, C = 2, 32, 16
    vocab_size = 50

    x = torch.randn(B, T, C, requires_grad=True)
    targets = torch.randint(0, vocab_size, (B, T))
    lm_head = nn.Linear(C, vocab_size, bias=False)

    # Standard Backward
    x_std = x.clone().detach().requires_grad_(True)
    loss_std = standard_loss(x_std, targets, lm_head)
    loss_std.backward()

    # Chunked Backward
    x_chunk = x.clone().detach().requires_grad_(True)
    loss_chunk = chunked_cross_entropy(x_chunk, targets, lm_head, chunk_size=10)
    loss_chunk.backward()

    # Check Gradients
    assert torch.allclose(x_std.grad, x_chunk.grad, atol=1e-5)

    # Rerun with fresh lm_head copies to be cleaner
    lm_head_std = nn.Linear(C, vocab_size, bias=False)
    lm_head_std.weight.data.copy_(lm_head.weight.data)

    lm_head_chunk = nn.Linear(C, vocab_size, bias=False)
    lm_head_chunk.weight.data.copy_(lm_head.weight.data)

    x_std = x.clone().detach().requires_grad_(True)
    loss_std = standard_loss(x_std, targets, lm_head_std)
    loss_std.backward()

    x_chunk = x.clone().detach().requires_grad_(True)
    loss_chunk = chunked_cross_entropy(x_chunk, targets, lm_head_chunk, chunk_size=10)
    loss_chunk.backward()

    assert torch.allclose(x_std.grad, x_chunk.grad, atol=1e-5), "Input gradients mismatch"
    assert torch.allclose(lm_head_std.weight.grad, lm_head_chunk.weight.grad, atol=1e-5), "Weight gradients mismatch"

if __name__ == "__main__":
    test_chunked_vs_standard_equivalence()
    test_chunked_loss_grad()
    print("All tests passed!")
