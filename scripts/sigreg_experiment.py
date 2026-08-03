"""
A/B test: does SIGReg change what a tiny nanochat learns?

Trains a tiny character-level nanochat GPT on TinyShakespeare twice - once with plain
cross-entropy, once with cross-entropy + lambda * SIGReg on the final hidden states - and
compares them on three axes:

  1. Language modelling:  val bits-per-character (the task metric; lower is better)
  2. Representation geometry: effective rank, top-PC dominance, mean |cosine|, and the
     held-out SIGReg statistic itself
  3. Downstream transfer: linear probe accuracy for speaker attribution from *frozen*
     mean-pooled features - the LM equivalent of LeJEPA's linear-probe protocol

Everything runs on CPU in a few minutes per arm. Each (lambda, seed) pair is a separate
run; results are aggregated as mean +/- std across seeds, because at this scale the
seed-to-seed spread is the thing any claimed effect has to beat.

Usage:
    python -m scripts.sigreg_experiment                       # default A/B, 3 seeds
    python -m scripts.sigreg_experiment --lams 0,0.03,0.3 --seeds 0,1,2,3
    python -m scripts.sigreg_experiment --center sequence     # temporally-centered variant
"""

import argparse
import json
import math
import os
import re
import time

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.sigreg import SIGReg

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# -----------------------------------------------------------------------------
# Data: character-level TinyShakespeare

def load_text(cache_dir):
    path = os.path.join(cache_dir, "tinyshakespeare.txt")
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        # bytes per token, so that bits-per-byte is comparable to nanochat's BPE runs.
        # TinyShakespeare is ASCII, so this is 1 everywhere and bpb == bits per character.
        self.token_bytes = torch.tensor([len(c.encode("utf-8")) for c in self.chars], dtype=torch.int64)

    def encode(self, s):
        return torch.tensor([self.stoi[c] for c in s if c in self.stoi], dtype=torch.long)


def make_batch(data, batch_size, seq_len, generator):
    """Random contiguous windows. Same generator + same seed => identical stream for every arm."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + seq_len] for i in ix])
    return x, y


# -----------------------------------------------------------------------------
# Probe data: speaker attribution, labels come free from the "NAME:" markup

BLOCK_RE = re.compile(r"^([A-Z][A-Za-z ]{1,20}):\n((?:.+\n)+)", re.M)


def build_probe_set(text, tokenizer, num_speakers, min_chars, seq_len):
    """(utterance, speaker) pairs for the N most prolific speakers, as fixed-length token windows."""
    blocks = [(s, u) for s, u in BLOCK_RE.findall(text) if len(u) >= min_chars]
    counts = {}
    for s, _ in blocks:
        counts[s] = counts.get(s, 0) + 1
    top = sorted(counts, key=counts.get, reverse=True)[:num_speakers]
    label_of = {s: i for i, s in enumerate(top)}

    xs, ys, lens = [], [], []
    for s, u in blocks:
        if s not in label_of:
            continue
        ids = tokenizer.encode(u)[:seq_len]
        if len(ids) < 2:
            continue
        pad = torch.zeros(seq_len, dtype=torch.long)
        pad[:len(ids)] = ids
        xs.append(pad)
        ys.append(label_of[s])
        lens.append(len(ids))
    return torch.stack(xs), torch.tensor(ys), torch.tensor(lens), top


# -----------------------------------------------------------------------------
# Metrics

@torch.no_grad()
def eval_bpb(model, data, tokenizer, batch_size, seq_len, steps, generator):
    """Bits per byte: sum of nats over the batch divided by ln(2) * total target bytes."""
    total_nats, total_bytes = 0.0, 0
    for _ in range(steps):
        x, y = make_batch(data, batch_size, seq_len, generator)
        loss = model(x, y, loss_reduction="none").view(-1)
        nb = tokenizer.token_bytes[y.view(-1)]
        total_nats += loss.sum().item()
        total_bytes += nb.sum().item()
    return total_nats / (math.log(2) * total_bytes)


@torch.no_grad()
def collect_hidden(model, x_all, batch_size):
    out = []
    for i in range(0, len(x_all), batch_size):
        _, h = model(x_all[i:i + batch_size], return_hidden=True)
        out.append(h.float())
    return torch.cat(out)


@torch.no_grad()
def geometry_stats(h):
    """h: (B, T, C). Isotropy diagnostics on the flattened token cloud."""
    z = h.reshape(-1, h.size(-1))
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.T @ zc) / (zc.size(0) - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    p = eig / eig.sum()
    eff_rank = torch.exp(-(p * p.log()).sum()).item()   # entropy-based effective rank
    top_pc = p.max().item()                             # variance share of the dominant direction
    # mean |cosine| between random token pairs: 0 for isotropic, ->1 for a narrow cone
    zn = F.normalize(z, dim=-1)
    idx = torch.randperm(zn.size(0))[:4096]
    a, b = zn[idx[:2048]], zn[idx[2048:4096]]
    mean_abs_cos = (a * b).sum(-1).abs().mean().item()
    return dict(eff_rank=eff_rank, eff_rank_frac=eff_rank / z.size(-1),
                top_pc_var=top_pc, mean_abs_cos=mean_abs_cos)


def linear_probe(feats, labels, num_classes, seed=1234, steps=400, train_frac=0.7):
    """Logistic regression on frozen, standardized features. Returns test accuracy."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(feats), generator=g)
    feats, labels = feats[perm], labels[perm]
    n_tr = int(train_frac * len(feats))
    xtr, ytr, xte, yte = feats[:n_tr], labels[:n_tr], feats[n_tr:], labels[n_tr:]
    # standardize using train statistics only
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd

    torch.manual_seed(seed)
    clf = torch.nn.Linear(feats.size(-1), num_classes)
    opt = torch.optim.AdamW(clf.parameters(), lr=0.05, weight_decay=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(clf(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return (clf(xte).argmax(-1) == yte).float().mean().item()


# -----------------------------------------------------------------------------
# One run

def run_one(args, lam, seed, data, tokenizer, probe):
    t_start = time.time()
    train_data, val_data = data
    px, py, plens, speakers = probe

    # Model. Identical init for every arm at a given seed.
    torch.manual_seed(seed)
    cfg = GPTConfig(sequence_len=args.seq_len, vocab_size=tokenizer.vocab_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_kv_head=args.n_head,
                    n_embd=args.n_embd, window_pattern="L")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()

    optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr,
                                      matrix_lr=args.matrix_lr, weight_decay=args.weight_decay)

    sigreg = SIGReg(num_slices=args.num_slices, token_subsample=args.token_subsample, center=args.center)
    # Dedicated RNG streams: the data order and the init must not depend on whether
    # SIGReg is switched on, otherwise the arms differ by more than lambda.
    data_gen = torch.Generator().manual_seed(seed)
    sigreg_gen = torch.Generator().manual_seed(seed + 10_000)

    ce_hist, sr_hist = [], []
    for step in range(args.steps):
        # linear warmup then linear decay to 10% of peak
        if step < args.warmup:
            lrm = (step + 1) / args.warmup
        else:
            frac = (step - args.warmup) / max(1, args.steps - args.warmup)
            lrm = 1.0 - 0.9 * frac
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm

        x, y = make_batch(train_data, args.batch_size, args.seq_len, data_gen)
        ce, hidden = model(x, y, return_hidden=True)
        sr = sigreg(hidden, generator=sigreg_gen)
        loss = ce + lam * sr

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        ce_hist.append(ce.item())
        sr_hist.append(sr.item())
        if args.verbose and (step % args.log_every == 0 or step == args.steps - 1):
            print(f"    step {step:4d}/{args.steps}  ce {ce.item():.4f}  sigreg {sr.item():9.2f}  lrm {lrm:.3f}")

    # ---- evaluation
    model.eval()
    eval_gen = torch.Generator().manual_seed(999)  # same eval batches for every arm
    val_bpb = eval_bpb(model, val_data, tokenizer, args.batch_size, args.seq_len, args.eval_steps, eval_gen)
    train_bpb = eval_bpb(model, train_data, tokenizer, args.batch_size, args.seq_len, args.eval_steps,
                         torch.Generator().manual_seed(998))

    # geometry + held-out SIGReg statistic, measured on val text
    x_val, _ = make_batch(val_data, 32, args.seq_len, torch.Generator().manual_seed(997))
    with torch.no_grad():
        _, h_val = model(x_val, return_hidden=True)
    geo = geometry_stats(h_val)
    diag_sigreg = SIGReg(num_slices=1024, token_subsample=args.token_subsample, center=args.center)
    with torch.no_grad():
        heldout_sigreg = diag_sigreg(h_val, generator=torch.Generator().manual_seed(996)).item()

    # linear probe on frozen mean-pooled features (mask out padding)
    h_probe = collect_hidden(model, px, args.batch_size)
    mask = (torch.arange(args.seq_len)[None, :] < plens[:, None]).float().unsqueeze(-1)
    feats = (h_probe * mask).sum(1) / mask.sum(1).clamp_min(1)
    probe_acc = linear_probe(feats, py, len(speakers))

    return dict(
        lam=lam, seed=seed, val_bpb=val_bpb, train_bpb=train_bpb,
        final_ce=sum(ce_hist[-50:]) / 50, final_sigreg=sum(sr_hist[-50:]) / 50,
        heldout_sigreg=heldout_sigreg, probe_acc=probe_acc,
        wall_s=time.time() - t_start, **geo,
    )


# -----------------------------------------------------------------------------

def summarize(rows, keys):
    """Group by lambda, report mean +/- std over seeds."""
    by_lam = {}
    for r in rows:
        by_lam.setdefault(r["lam"], []).append(r)
    out = {}
    for lam, rs in sorted(by_lam.items()):
        out[lam] = {}
        for k in keys:
            vals = [r[k] for r in rs]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
            out[lam][k] = (mean, var ** 0.5)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lams", type=str, default="0.0,0.03")
    p.add_argument("--seeds", type=str, default="0,1,2")
    # model / data
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=20)
    # optimizer (nanochat defaults)
    p.add_argument("--unembedding-lr", type=float, default=0.004)
    p.add_argument("--embedding-lr", type=float, default=0.2)
    p.add_argument("--matrix-lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.0)
    # sigreg
    p.add_argument("--num-slices", type=int, default=256)
    p.add_argument("--token-subsample", type=int, default=1024)
    p.add_argument("--center", type=str, default="none", choices=["none", "sequence", "batch"])
    # probe
    p.add_argument("--num-speakers", type=int, default=12)
    p.add_argument("--min-chars", type=int, default=48)
    # misc
    p.add_argument("--cache-dir", type=str, default=os.path.expanduser("~/.cache/sigreg"))
    p.add_argument("--out", type=str, default="")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    lams = [float(v) for v in args.lams.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    text = load_text(args.cache_dir)
    tokenizer = CharTokenizer(text)
    n_train = int(0.9 * len(text))
    train_text, val_text = text[:n_train], text[n_train:]
    data = (tokenizer.encode(train_text), tokenizer.encode(val_text))
    # probe examples come from the LM-train region; val text is reserved for bpb
    probe = build_probe_set(train_text, tokenizer, args.num_speakers, args.min_chars, args.seq_len)
    print(f"vocab {tokenizer.vocab_size} | train chars {len(train_text):,} | val chars {len(val_text):,}")
    print(f"probe: {len(probe[0])} utterances over {len(probe[3])} speakers "
          f"(majority-class baseline {torch.bincount(probe[1]).max().item() / len(probe[1]):.3f})")

    rows = []
    for lam in lams:
        for seed in seeds:
            print(f"\n=== lambda={lam} seed={seed} ===")
            r = run_one(args, lam, seed, data, tokenizer, probe)
            rows.append(r)
            print(f"    val_bpb {r['val_bpb']:.4f} | probe {r['probe_acc']:.3f} | "
                  f"eff_rank {r['eff_rank']:.1f} | heldout_sigreg {r['heldout_sigreg']:.1f} | "
                  f"{r['wall_s']:.0f}s")

    keys = ["val_bpb", "train_bpb", "probe_acc", "eff_rank", "eff_rank_frac",
            "top_pc_var", "mean_abs_cos", "heldout_sigreg"]
    summary = summarize(rows, keys)

    print("\n" + "=" * 78)
    print(f"RESULTS over {len(seeds)} seeds (mean +/- std)   center={args.center}")
    print("=" * 78)
    header = f"{'metric':<16}" + "".join(f"{'lam=' + str(l):>20}" for l in lams)
    print(header)
    print("-" * len(header))
    for k in keys:
        line = f"{k:<16}"
        for l in lams:
            m, s = summary[l][k]
            line += f"{m:>13.4f} +/-{s:<5.3f}"
        print(line)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(args=vars(args), rows=rows), f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
