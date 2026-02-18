#!/usr/bin/env python3
"""
Instantiate the local `GPT` model and generate visualization images for key weight matrices
and sample activations. If the `visualtorch` package is installed, the script will try to use it;
otherwise it falls back to saving PNG heatmaps using matplotlib.

Example:
  python visual/run_visualtorch_viz.py --n-embd 256 --n-head 4 --vocab 4096 --seq-len 32
"""
from __future__ import annotations
import os
import argparse
import math
import numpy as np
import torch

try:
    # attempt to import visualtorch (user-installed)
    import visualtorch as vt
    HAS_VT = True
except Exception:
    vt = None
    HAS_VT = False

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_heatmap(arr, path, title=None, dpi=180):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        # if matplotlib missing, save raw npy
        np.save(path + '.npy', arr)
        print(f"matplotlib not available — wrote raw array to {path}.npy")
        return
    plt.figure(figsize=(6,6))
    plt.imshow(arr, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def try_visualtorch_save(arr, path, name=None):
    # Try a few plausible VisualTorch APIs, but fall back to heatmap
    if not HAS_VT:
        return False
    try:
        # Common pattern: visualtorch may have a `save` or `save_tensor` API
        if hasattr(vt, 'save'):
            vt.save(arr, path)
            return True
        if hasattr(vt, 'save_tensor'):
            vt.save_tensor(arr, path)
            return True
        if hasattr(vt, 'visualize') and hasattr(vt.visualize, 'save'):
            vt.visualize.save(arr, path)
            return True
        # as a last resort, try constructing a VisualTorch image via repr
        if hasattr(vt, 'to_image'):
            img = vt.to_image(arr)
            with open(path, 'wb') as f:
                f.write(img)
            return True
    except Exception as e:
        print('visualtorch save attempt failed:', e)
    return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-embd', type=int, default=768)
    p.add_argument('--n-head', type=int, default=6)
    p.add_argument('--n-kv-head', type=int, default=6)
    p.add_argument('--vocab', type=int, default=32768)
    p.add_argument('--n-layer', type=int, default=12)
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--out-dir', type=str, default='visual/vis_output')
    args = p.parse_args()

    ensure_dir(args.out_dir)

    # lazy import model from repo
    from nanochat.gpt import GPT, GPTConfig

    cfg = GPTConfig(sequence_len=args.seq_len, vocab_size=args.vocab, n_layer=args.n_layer, n_head=args.n_head, n_kv_head=args.n_kv_head, n_embd=args.n_embd)
    model = GPT(cfg)
    # initialize weights
    model.init_weights()

    device = torch.device('cpu')
    model.to(device)

    # prepare a random input batch and run forward to produce activations
    idx = torch.randint(0, args.vocab, (1, args.seq_len), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model.forward(idx)

    # collect tensors of interest
    items = []
    # embedding
    wte = model.transformer.wte.weight.detach().cpu().numpy()
    items.append(('wte', wte))
    # lm_head
    lm = model.lm_head.weight.detach().cpu().numpy()
    items.append(('lm_head', lm))
    # per-block matrices (first block as representative)
    for i, block in enumerate(model.transformer.h):
        attn = block.attn
        # linear weights - transpose to (in, out) for visual consistency
        try:
            cq = attn.c_q.weight.detach().cpu().numpy()
            ck = attn.c_k.weight.detach().cpu().numpy()
            cv = attn.c_v.weight.detach().cpu().numpy()
            cproj = attn.c_proj.weight.detach().cpu().numpy()
            items.append((f'block{i}_c_q', cq))
            items.append((f'block{i}_c_k', ck))
            items.append((f'block{i}_c_v', cv))
            items.append((f'block{i}_c_proj', cproj))
        except Exception:
            pass
        # mlp
        try:
            mlp_fc = block.mlp.c_fc.weight.detach().cpu().numpy()
            mlp_proj = block.mlp.c_proj.weight.detach().cpu().numpy()
            items.append((f'block{i}_mlp_fc', mlp_fc))
            items.append((f'block{i}_mlp_proj', mlp_proj))
        except Exception:
            pass
        # only visualize first few blocks to limit output
        if i >= 2:
            break

    # sample activations
    try:
        emb = model.transformer.wte(idx).detach().cpu().numpy()[0]
        items.append(('sample_emb', emb))
    except Exception:
        pass

    # save images
    for name, arr in items:
        # try to reduce to 2D for heatmap: if >2D, collapse leading dims
        darr = np.array(arr)
        if darr.ndim > 2:
            darr2 = darr.reshape(darr.shape[0], -1) if darr.shape[0] >= darr.shape[-1] else darr.reshape(-1, darr.shape[-1])
        else:
            darr2 = darr
        out_path = os.path.join(args.out_dir, f'{name}.png')
        saved = False
        if HAS_VT:
            saved = try_visualtorch_save(darr2, out_path)
        if not saved:
            save_heatmap(darr2, out_path, title=name)
        print('wrote', out_path)

    print('Done. Visualizations written to', args.out_dir)

if __name__ == '__main__':
    main()
