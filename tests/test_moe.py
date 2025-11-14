import torch
import torch.nn.functional as F

from nanochat.gpt import GPTConfig, MoEFeedForward, GPT


def _make_moe(num_experts: int, top_k: int, shared: int = 0) -> MoEFeedForward:
    config = GPTConfig(
        sequence_len=8,
        vocab_size=16,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=4,
        moe_num_experts=num_experts,
        moe_num_shared_experts=shared,
        moe_experts_per_token=top_k,
        moe_expert_ffn_mult=1.0,
        dense_layers_before_moe=0,
    )
    moe = MoEFeedForward(config)
    moe.eval()
    with torch.no_grad():
        moe.router.weight.zero_()
        bias = torch.linspace(float(num_experts), 1.0, steps=num_experts)
        moe.router_bias.copy_(bias)
        eye = torch.eye(moe.model_dim)
        for idx in range(num_experts):
            moe.routed_w1[idx].copy_(eye)
            moe.routed_w2[idx].copy_(eye * (idx + 1))
        if shared:
            for idx in range(shared):
                moe.shared_w1[idx].copy_(eye)
                moe.shared_w2[idx].copy_(eye * (idx + 1))
    return moe


def _reference_forward(moe: MoEFeedForward, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    x_flat = x.view(B * T, C)
    router_logits = moe.router(x_flat) + moe.router_bias
    probs = torch.softmax(router_logits, dim=-1)
    topk_scores, topk_idx = torch.topk(probs, moe.top_k, dim=-1)
    topk_scores = topk_scores / torch.clamp(topk_scores.sum(dim=-1, keepdim=True), min=1e-9)

    routed = torch.zeros_like(x_flat)
    for token in range(x_flat.size(0)):
        for slot in range(moe.top_k):
            expert = topk_idx[token, slot].item()
            weight = topk_scores[token, slot]
            hidden = torch.matmul(moe.routed_w1[expert], x_flat[token])
            hidden = F.relu(hidden).square()
            routed[token] = routed[token] + weight * torch.matmul(moe.routed_w2[expert], hidden)

    if moe.num_shared_experts > 0:
        shared = torch.zeros_like(x_flat)
        for idx in range(moe.num_shared_experts):
            hidden = torch.matmul(moe.shared_w1[idx], x_flat.t()).t()
            hidden = F.relu(hidden).square()
            shared = shared + torch.matmul(moe.shared_w2[idx], hidden.t()).t()
        routed = routed + shared / moe.num_shared_experts

    return routed.view(B, T, C)


def test_moe_topk_changes_output():
    torch.manual_seed(0)
    moe_full = _make_moe(num_experts=4, top_k=4)
    moe_single = _make_moe(num_experts=4, top_k=1)
    with torch.no_grad():
        moe_single.router.weight.copy_(moe_full.router.weight)
        moe_single.router_bias.copy_(moe_full.router_bias)
        moe_single.routed_w1.copy_(moe_full.routed_w1)
        moe_single.routed_w2.copy_(moe_full.routed_w2)
    x = torch.ones(2, 3, moe_full.model_dim)
    out_full = moe_full(x)
    out_single = moe_single(x)
    assert not torch.allclose(out_full, out_single)


def test_router_bias_pushes_toward_uniform_load():
    moe = _make_moe(num_experts=4, top_k=1)
    moe.train()
    assignments = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    probs = torch.full((assignments.shape[0], moe.num_routed_experts), 1.0 / moe.num_routed_experts)
    before = moe.router_bias.clone()
    moe._balance_router(assignments, probs)
    after = moe.router_bias
    assert after[0].item() < before[0].item()
    assert torch.all(after[1:] > before[1:])


def test_shared_expert_matches_reference():
    torch.manual_seed(0)
    moe = _make_moe(num_experts=2, top_k=1, shared=2)
    x = torch.randn(2, 3, moe.model_dim)
    out = moe(x)
    ref = _reference_forward(moe, x)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)


class LoopMoE(MoEFeedForward):
    def _dispatch_batched(self, flat_assignments, x_flat, topk_scores):
        num_tokens, model_dim = x_flat.shape
        routed = torch.zeros(num_tokens, model_dim, device=x_flat.device, dtype=x_flat.dtype)
        assignments = flat_assignments.view(num_tokens, self.top_k)
        for token in range(num_tokens):
            for slot in range(self.top_k):
                expert = assignments[token, slot].item()
                weight = topk_scores[token, slot]
                hidden = torch.matmul(self.routed_w1[expert], x_flat[token])
                hidden = F.relu(hidden).square()
                routed[token] = routed[token] + weight * torch.matmul(self.routed_w2[expert], hidden)
        return routed


def test_moe_gradients_match_loop_reference():
    torch.manual_seed(0)
    config_moe = GPTConfig(
        sequence_len=8,
        vocab_size=16,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=4,
        moe_num_experts=4,
        moe_num_shared_experts=1,
        moe_experts_per_token=2,
        moe_expert_ffn_mult=1.5,
        dense_layers_before_moe=0,
    )
    moe_fast = MoEFeedForward(config_moe)
    moe_loop = LoopMoE(config_moe)
    moe_loop.load_state_dict(moe_fast.state_dict())

    x_fast = torch.randn(2, 3, config_moe.n_embd, requires_grad=True)
    x_loop = x_fast.detach().clone().requires_grad_(True)
    target = torch.randn_like(x_fast)

    out_fast = moe_fast(x_fast)
    out_loop = moe_loop(x_loop)
    assert torch.allclose(out_fast, out_loop, atol=1e-6, rtol=1e-5)

    loss_fast = (out_fast * target).sum()
    loss_loop = (out_loop * target).sum()

    moe_fast.zero_grad(set_to_none=True)
    moe_loop.zero_grad(set_to_none=True)
    x_fast.grad = None
    x_loop.grad = None

    loss_fast.backward(retain_graph=True)
    loss_loop.backward()

    for (name_fast, p_fast), (name_loop, p_loop) in zip(moe_fast.named_parameters(), moe_loop.named_parameters()):
        assert torch.allclose(p_fast.grad, p_loop.grad, atol=1e-6, rtol=1e-5), f"gradient mismatch for {name_fast}"

    assert torch.allclose(x_fast.grad, x_loop.grad, atol=1e-6, rtol=1e-5)


def test_moe_experts_get_gradients_and_update():
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        moe_num_experts=4,
        moe_num_shared_experts=1,
        moe_experts_per_token=2,
        moe_expert_ffn_mult=2.0,
        dense_layers_before_moe=0,
    )
    model = GPT(config)
    model.init_weights()
    adamw_opt, muon_opt = model.setup_optimizers()

    def train_step():
        x = torch.randint(0, config.vocab_size, (2, config.sequence_len))
        y = torch.randint(0, config.vocab_size, (2, config.sequence_len))
        loss = model(x, y)
        loss.backward()

    # Warmup step: lm_head starts at zero, so the first backward pass
    # leaves trunk gradients at zero. Step once to propagate signal.
    train_step()
    for opt in (adamw_opt, muon_opt):
        opt.step()
    model.zero_grad(set_to_none=True)

    # Second step should yield non-zero gradients for the experts.
    train_step()
    routed_grad_norms = []
    shared_grad_norms = []
    routed_param_norms = []
    for block in model.transformer.h:
        mlp = getattr(block, "mlp", None)
        if hasattr(mlp, "routed_w2"):
            routed_grad = mlp.routed_w2.grad
            assert routed_grad is not None
            routed_grad_norms.append(routed_grad.abs().mean().item())
            routed_param_norms.append(mlp.routed_w2.abs().mean().item())
            if mlp.shared_w2 is not None:
                shared_grad = mlp.shared_w2.grad
                assert shared_grad is not None
                shared_grad_norms.append(shared_grad.abs().mean().item())
    assert all(norm > 0.0 for norm in routed_grad_norms)
    assert all(norm > 0.0 for norm in shared_grad_norms)

    for opt in (adamw_opt, muon_opt):
        opt.step()
    model.zero_grad(set_to_none=True)
    routed_param_norms_after = []
    for block in model.transformer.h:
        mlp = getattr(block, "mlp", None)
        if hasattr(mlp, "routed_w2"):
            routed_param_norms_after.append(mlp.routed_w2.abs().mean().item())
    assert all(after > before for before, after in zip(routed_param_norms, routed_param_norms_after))
