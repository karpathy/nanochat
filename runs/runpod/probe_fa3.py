#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Comprehensive FA3 / kernels diagnostic probe.

nanochat/flash_attention.py:_load_flash_attention_3 swallows ALL exceptions silently
and falls back to SDPA. This script runs the same code path with full tracebacks
so we can see why FA3 isn't loading on the pod.

Run inside the pod (after uv sync, with venv active):
  python runs/runpod/probe_fa3.py

Exits 0 if FA3 is fully usable, 1 if any check fails.

Note: the pyright pragma above is intentional — torch/huggingface_hub/kernels
are only present at pod runtime; the local IDE will flag them as unresolved.
"""
import os
import sys
import traceback
import platform
import subprocess

OK = "\033[32mOK\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def section(n, name):
    print()
    print("=" * 80)
    print(f"[{n}] {name}")
    print("=" * 80)


def fmt_token(tok):
    if not tok:
        return "NOT SET"
    return f"SET (len={len(tok)}, prefix={tok[:7]}…)"


passed_all = True


def fail(msg):
    global passed_all
    passed_all = False
    print(f"  {FAIL} {msg}")


def warn(msg):
    print(f"  {WARN} {msg}")


def ok(msg):
    print(f"  {OK} {msg}")


# ---------------------------------------------------------------------------
section(1, "Environment")
print(f"  python    : {sys.version.split()[0]}")
print(f"  platform  : {platform.platform()}")
print(f"  cwd       : {os.getcwd()}")
hf_tok = os.environ.get("HF_TOKEN", "")
hf_hub_tok = os.environ.get("HF_HUB_TOKEN", "")
print(f"  HF_TOKEN     : {fmt_token(hf_tok)}")
print(f"  HF_HUB_TOKEN : {fmt_token(hf_hub_tok)}")
print(f"  HF_HOME      : {os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}")
print(f"  HUGGINGFACE_HUB_CACHE : {os.environ.get('HUGGINGFACE_HUB_CACHE', '(unset)')}")
print(f"  WANDB_API_KEY: {fmt_token(os.environ.get('WANDB_API_KEY',''))}")

if not hf_tok:
    fail("HF_TOKEN env var is empty — kernels lib will fall back to anonymous and may rate-limit")

# ---------------------------------------------------------------------------
section(2, "Network connectivity")
for url, label in [
    ("https://huggingface.co", "huggingface.co"),
    ("https://cdn-lfs.huggingface.co", "cdn-lfs.huggingface.co"),
    ("https://github.com", "github.com"),
]:
    try:
        rc = subprocess.run(
            ["curl", "-sfI", "--max-time", "10", url],
            capture_output=True, text=True, timeout=15,
        ).returncode
        if rc == 0:
            ok(f"{label} reachable")
        else:
            fail(f"{label} unreachable (curl rc={rc})")
    except Exception as e:
        fail(f"{label}: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
section(3, "huggingface_hub auth (does the token actually work?)")
try:
    from huggingface_hub import whoami
    info = whoami(token=hf_tok or None)
    ok(f"authenticated as: {info.get('name','?')} (type={info.get('type','?')})")
    print(f"     orgs: {[o.get('name') for o in info.get('orgs', [])]}")
    print(f"     access token role: {info.get('auth',{}).get('accessToken',{}).get('role','?')}")
except Exception as e:
    fail(f"whoami failed: {type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section(4, "PyTorch / CUDA / GPU")
try:
    import torch
    print(f"  torch          : {torch.__version__}")
    print(f"  cuda available : {torch.cuda.is_available()}")
    print(f"  cuda version   : {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  device count   : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            name = torch.cuda.get_device_name(i)
            mark = OK if major == 9 else WARN
            print(f"  device {i}       : {name}  sm{major}{minor}  [{mark}]")
        major, _ = torch.cuda.get_device_capability(0)
        if major != 9:
            fail(f"FA3 requires sm90 (Hopper); device 0 is sm{major}{_}")
    else:
        fail("CUDA not available")
except Exception as e:
    fail(f"torch import failed: {type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section(5, "kernels library")
try:
    import kernels
    ver = getattr(kernels, "__version__", "?")
    print(f"  kernels       : {ver}")
    if ver != "?":
        major_minor = tuple(int(x) for x in ver.split(".")[:2])
        if major_minor < (0, 13):
            warn(f"kernels {ver} < 0.13 — older versions have known kernel-resolution bugs; consider 'uv pip install --upgrade kernels'")
        else:
            ok(f"kernels {ver} is recent")
    print(f"  kernels path  : {kernels.__file__}")
except Exception as e:
    fail(f"kernels not importable: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
section(6, "Fetch varunneal/flash-attention-3 (THE actual nanochat code path)")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
try:
    from kernels import get_kernel
    print("  calling get_kernel('varunneal/flash-attention-3') …")
    k = get_kernel("varunneal/flash-attention-3")
    ok(f"get_kernel returned: {type(k).__name__}")
    print(f"     module path: {getattr(k, '__file__', '(no __file__)')}")
    iface = k.flash_attn_interface
    ok(f"flash_attn_interface: {type(iface).__name__}")
    fn = iface.flash_attn_func
    ok(f"flash_attn_func: callable={callable(fn)}")
    print("\n  >>> FA3 binary is usable on this pod. <<<")
except Exception as e:
    fail(f"FA3 fetch failed: {type(e).__name__}: {e}")
    print()
    traceback.print_exc()
    print()
    print("  Likely causes:")
    print("    1. Network/DNS issue (HF Hub unreachable from this DC)")
    print("    2. Old kernels version with resolver bugs (try kernels>=0.13)")
    print("    3. HF token not flowing — try `export HF_HUB_TOKEN=$HF_TOKEN`")
    print("    4. No prebuilt binary for this torch/cuda combo (we have torch 2.9 + cu128 — should be supported)")

# ---------------------------------------------------------------------------
section(7, "HF Hub cache state")
import pathlib
cache_root = pathlib.Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
print(f"  cache root    : {cache_root}")
if cache_root.exists():
    try:
        size = sum(p.stat().st_size for p in cache_root.rglob("*") if p.is_file())
        print(f"  size on disk  : {size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"  (could not size cache: {e})")
    fa3_marker = list(cache_root.rglob("*flash*attention*3*"))
    if fa3_marker:
        ok(f"found FA3-related cache entries: {len(fa3_marker)}")
        for p in fa3_marker[:5]:
            print(f"     {p}")
    else:
        warn("no flash-attention-3 entries in cache yet")
else:
    print("  (cache directory does not exist)")

# ---------------------------------------------------------------------------
section(8, "Replicate nanochat.flash_attention detection")
try:
    from nanochat.flash_attention import HAS_FA3, USE_FA3, _fa3
    if HAS_FA3:
        ok("nanochat.flash_attention.HAS_FA3 = True")
    else:
        fail("nanochat.flash_attention.HAS_FA3 = False (despite section 6 results)")
    print(f"  USE_FA3 = {USE_FA3}")
    print(f"  _fa3 object: {_fa3}")
except Exception as e:
    fail(f"import nanochat.flash_attention failed: {type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section(9, "Verdict")
if passed_all:
    print(f"  {OK} all checks passed — FA3 is wired up and base_train should use it")
    sys.exit(0)
else:
    print(f"  {FAIL} one or more checks failed — see above. Run will fall back to SDPA (slower, possibly much).")
    print()
    print("  Continuing the training run anyway is safe; FA3 fallback to SDPA is automatic.")
    sys.exit(1)
