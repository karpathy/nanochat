import modal
from pathlib import Path

APP_NAME = "nanochat-d32-upload"
VOLUME_NAME = "nanochat-data"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Image with huggingface-hub for downloading files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface-hub")
)

@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=60 * 60 * 2,  # 2 hours for downloading ~7.25GB
    max_inputs=1,
)
def upload_d32_checkpoint():
    """
    Download nanochat-d32 model files from Hugging Face and sync them into Modal volume.

    Downloads from: https://huggingface.co/karpathy/nanochat-d32/tree/main
    Target path in volume: .cache/nanochat/chatsft_uploaded

    Files to download:
    - model_000650.pt (7.25 GB) - PyTorch model checkpoint
    - meta_000650.json (263 B) - Metadata for checkpoint
    - tokenizer.pkl (846 kB) - Tokenizer pickle file
    - token_bytes.pt (264 kB) - Token bytes
    - README.md (526 B) - Documentation
    """
    from huggingface_hub import hf_hub_download
    import shutil

    DATA = Path("/data")
    TARGET_DIR = DATA / ".cache" / "nanochat" / "chatsft_uploaded"
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    REPO_ID = "karpathy/nanochat-d32"
    FILES_TO_DOWNLOAD = [
        "model_000650.pt",      # 7.25 GB - main model checkpoint
        "meta_000650.json",     # 263 B - metadata
        "tokenizer.pkl",        # 846 kB - tokenizer
        "token_bytes.pt",       # 264 kB - token bytes
        "README.md",            # 526 B - documentation
    ]

    print(f"\n{'='*80}")
    print(f"📥 Downloading nanochat-d32 files from Hugging Face")
    print(f"   Repository: {REPO_ID}")
    print(f"   Target: {TARGET_DIR}")
    print(f"{'='*80}\n")

    for filename in FILES_TO_DOWNLOAD:
        print(f"\n📦 Downloading {filename}...")

        # Download file to HF cache
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="model",
        )

        # Copy to our target directory in the volume
        target_path = TARGET_DIR / filename
        print(f"   Copying to {target_path}...")
        shutil.copy2(cached_path, target_path)
        print(f"   ✓ {filename} synced")

    # Commit changes to volume
    vol.commit()

    print(f"\n{'='*80}")
    print(f"✅ All files downloaded and synced to volume!")
    print(f"{'='*80}\n")

    # List final directory contents
    print("\n📋 Final directory contents:")
    for file in sorted(TARGET_DIR.iterdir()):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   {file.name:25s} ({size_mb:8.2f} MB)")

    print(f"\nFiles available at: /data/.cache/nanochat/chatsft_uploaded")


# Local entrypoint
@app.local_entrypoint()
def main():
    """
    Download nanochat-d32 checkpoint from Hugging Face to Modal volume.

    Usage:
        modal run modal_d32_upload.py

    This will download ~7.25GB of model files and sync them to:
        {VOLUME_NAME}/.cache/nanochat/chatsft_uploaded/
    """
    print("\n🚀 Starting nanochat-d32 upload to Modal volume...")
    upload_d32_checkpoint.remote()
    print("\n✅ Upload complete! Files are now in the Modal volume.")
