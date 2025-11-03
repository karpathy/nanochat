#!/usr/bin/env uv run --with=huggingface_hub

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# directories
HF_TEMP = Path("./hf_temp")
TOKEN_DIR = Path.home() / ".cache/nanochat/tokenizer"
MODEL_DIR = Path.home() / ".cache/nanochat/chatsft_checkpoints/d32"

# ensure dirs exist
for d in [HF_TEMP, TOKEN_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# files mapping -> destination dir
files = {
    "token_bytes.pt": TOKEN_DIR,
    "tokenizer.pkl": TOKEN_DIR,
    "meta_000650.json": MODEL_DIR,
    "model_000650.pt": MODEL_DIR,
}

repo_id = "karpathy/nanochat-d32"

for fname, target_dir in files.items():
    target_path = target_dir / fname
    if target_path.exists():
        print(f"{fname} already exists in {target_dir}, skipping.")
        continue

    print(f"downloading {fname} to {HF_TEMP}...")
    temp_file = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=HF_TEMP)
    print(f"moving {temp_file} to {target_dir}")
    os.rename(temp_file, target_path)

print("all files placed correctly.")
