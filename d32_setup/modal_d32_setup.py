"""
Setup script to organize the uploaded d32 model from HuggingFace.

This script moves the d32 model files into the proper nanochat directory structure
so it can be used alongside your trained d20 model.

The d32 model comes with its own tokenizer, so we need to set up a separate
environment variable or parameter to switch between d20 and d32.

Directory structure after setup:
  .cache/nanochat/
    ├── chatsft_checkpoints/
    │   ├── d20/              # Your trained model
    │   │   ├── model_*.pt
    │   │   └── meta_*.json
    │   └── d32/              # Karpathy's d32 model
    │       ├── model_000650.pt
    │       └── meta_000650.json
    └── tokenizer_d32/        # d32's tokenizer (separate from your d20 tokenizer)
        ├── tokenizer.pkl
        └── token_bytes.pt
"""

import modal
from pathlib import Path
import shutil

APP_NAME = "nanochat-d32-setup"
VOLUME_NAME = "nanochat-data"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11")

@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=60 * 10,  # 10 minutes
    max_inputs=1,
)
def setup_d32_model():
    """
    Organize the uploaded d32 model files into the proper nanochat directory structure.

    This moves:
    - model_000650.pt -> chatsft_checkpoints/d32/model_000650.pt
    - meta_000650.json -> chatsft_checkpoints/d32/meta_000650.json
    - tokenizer.pkl -> tokenizer_d32/tokenizer.pkl
    - token_bytes.pt -> tokenizer_d32/token_bytes.pt
    """
    DATA = Path("/data")
    BASE_DIR = DATA / ".cache" / "nanochat"

    # Source: where the d32 files were uploaded
    UPLOADED_DIR = BASE_DIR / "chatsft_uploaded"

    # Destinations
    D32_CHECKPOINT_DIR = BASE_DIR / "chatsft_checkpoints" / "d32"
    D32_TOKENIZER_DIR = BASE_DIR / "tokenizer_d32"

    print(f"\n{'='*80}")
    print(f"🔧 Setting up d32 model in nanochat directory structure")
    print(f"{'='*80}\n")

    # Check if uploaded files exist
    if not UPLOADED_DIR.exists():
        print(f"❌ Error: Upload directory not found: {UPLOADED_DIR}")
        print(f"   Please run modal_d32_upload.py first!")
        return

    uploaded_files = list(UPLOADED_DIR.iterdir())
    print(f"📦 Found {len(uploaded_files)} files in {UPLOADED_DIR}:")
    for f in uploaded_files:
        print(f"   - {f.name}")

    # Create destination directories
    D32_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    D32_TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    # Move model checkpoint and metadata
    print(f"\n📁 Setting up model checkpoint in {D32_CHECKPOINT_DIR}...")
    model_file = UPLOADED_DIR / "model_000650.pt"
    meta_file = UPLOADED_DIR / "meta_000650.json"

    if model_file.exists():
        dest = D32_CHECKPOINT_DIR / "model_000650.pt"
        print(f"   Copying {model_file.name} -> {dest}")
        shutil.copy2(model_file, dest)
    else:
        print(f"   ⚠️  Warning: {model_file.name} not found")

    if meta_file.exists():
        dest = D32_CHECKPOINT_DIR / "meta_000650.json"
        print(f"   Copying {meta_file.name} -> {dest}")
        shutil.copy2(meta_file, dest)
    else:
        print(f"   ⚠️  Warning: {meta_file.name} not found")

    # Move tokenizer files
    print(f"\n🔤 Setting up d32 tokenizer in {D32_TOKENIZER_DIR}...")
    tokenizer_file = UPLOADED_DIR / "tokenizer.pkl"
    token_bytes_file = UPLOADED_DIR / "token_bytes.pt"

    if tokenizer_file.exists():
        dest = D32_TOKENIZER_DIR / "tokenizer.pkl"
        print(f"   Copying {tokenizer_file.name} -> {dest}")
        shutil.copy2(tokenizer_file, dest)
    else:
        print(f"   ⚠️  Warning: {tokenizer_file.name} not found")

    if token_bytes_file.exists():
        dest = D32_TOKENIZER_DIR / "token_bytes.pt"
        print(f"   Copying {token_bytes_file.name} -> {dest}")
        shutil.copy2(token_bytes_file, dest)
    else:
        print(f"   ⚠️  Warning: {token_bytes_file.name} not found")

    # Commit changes to volume
    vol.commit()

    print(f"\n{'='*80}")
    print(f"✅ d32 model setup complete!")
    print(f"{'='*80}\n")

    print("\n📋 Directory structure:")
    print(f"\nYour d20 model:")
    print(f"   Model: .cache/nanochat/chatsft_checkpoints/d20/")
    print(f"   Tokenizer: .cache/nanochat/tokenizer/")

    print(f"\nKarpathy's d32 model:")
    print(f"   Model: .cache/nanochat/chatsft_checkpoints/d32/")
    print(f"   Tokenizer: .cache/nanochat/tokenizer_d32/")

    print("\n⚠️  IMPORTANT:")
    print("   Each model MUST use its own tokenizer!")
    print("   You cannot mix d20 model with d32 tokenizer or vice versa.")
    print("\n   To use d32, you'll need to modify scripts to:")
    print("   1. Set model_tag='d32' to load the d32 checkpoint")
    print("   2. Modify get_tokenizer() to load from tokenizer_d32/")
    print("   OR use a separate script that handles d32 specifically.")


@app.local_entrypoint()
def main():
    """
    Organize uploaded d32 model files into proper nanochat directory structure.

    Usage:
        modal run modal_d32_setup.py

    This should be run AFTER modal_d32_upload.py completes.
    """
    print("\n🚀 Setting up d32 model directory structure...")
    setup_d32_model.remote()
    print("\n✅ Setup complete!")
