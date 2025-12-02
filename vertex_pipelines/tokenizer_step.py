import os
import sys
import subprocess
import argparse
import shutil
from google.cloud import storage

def upload_directory_to_gcs(local_path, bucket_name, gcs_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for root, _, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            blob_path = os.path.join(gcs_path, relative_path)
            blob = bucket.blob(blob_path)
            blob.upload_from_file(open(local_file, 'rb'))
            print(f"Uploaded {local_file} to gs://{bucket_name}/{blob_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket for artifacts")
    args = parser.parse_args()

    # Parse bucket name and prefix from args.gcs_bucket
    if args.gcs_bucket.startswith("gs://"):
        bucket_name = args.gcs_bucket.replace("gs://", "").split("/")[0]
        # Handle cases where there might be a prefix
        prefix_parts = args.gcs_bucket.replace("gs://", "").split("/")[1:]
        prefix = "/".join(prefix_parts) if prefix_parts else ""
    else:
        bucket_name = args.gcs_bucket
        prefix = ""

    # Check if tokenizer artifacts already exist (checkpoint detection)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    gcs_tokenizer_path = os.path.join(prefix, "tokenizer") if prefix else "tokenizer"
    
    # Check for key tokenizer files
    tokenizer_files = ["model.json", "vocab.json", "merges.txt"]
    all_exist = all(bucket.blob(os.path.join(gcs_tokenizer_path, f)).exists() for f in tokenizer_files)
    
    if all_exist:
        print(f"✓ Tokenizer artifacts already exist in gs://{bucket_name}/{gcs_tokenizer_path}")
        print("Skipping tokenizer training (already completed)")
        return

    print(f"Tokenizer artifacts not found. Running tokenizer training...")

    # Set the base directory to a local temporary directory.
    # We cannot use GCS directly because the tokenizer training script (Rust) expects local files.
    local_base_dir = "/tmp/nanochat"
    os.environ["NANOCHAT_BASE_DIR"] = local_base_dir
    os.makedirs(local_base_dir, exist_ok=True)

    try:
        # Download the dataset.
        # nanochat.dataset supports GCS, so we can point NANOCHAT_DATA_DIR to GCS if we wanted,
        # but for simplicity let's just let it download to local temp.
        # If you have data in GCS, you could set NANOCHAT_DATA_DIR to gs://...
        # For now, we assume we download from HF to local.
        print("Downloading dataset (n=8)...")
        subprocess.run([sys.executable, "-m", "nanochat.dataset", "-n", "8"], check=True)
        print("Downloading dataset (n=240)...")
        subprocess.run([sys.executable, "-m", "nanochat.dataset", "-n", "240"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        raise

    try:
        # Train the tokenizer.
        print("Training tokenizer...")
        subprocess.run([sys.executable, "scripts/tok_train.py", "--max_chars=2000000000"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error training tokenizer: {e}")
        raise

    try:
        # Evaluate the tokenizer.
        print("Evaluating tokenizer...")
        subprocess.run([sys.executable, "scripts/tok_eval.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating tokenizer: {e}")
        raise

    # Upload artifacts to GCS
    print("Uploading artifacts to GCS...")
    
    # Upload tokenizer
    local_tokenizer_dir = os.path.join(local_base_dir, "tokenizer")
    gcs_tokenizer_path = os.path.join(prefix, "tokenizer") if prefix else "tokenizer"
    upload_directory_to_gcs(local_tokenizer_dir, bucket_name, gcs_tokenizer_path)

    # Upload tokenized data if needed? 
    # Usually we don't upload the raw data here, but tok_train might produce token_bytes.pt which is in tokenizer dir.

if __name__ == "__main__":
    main()
