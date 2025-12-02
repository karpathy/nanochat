import os
import subprocess
import argparse
import shutil
from google.cloud import storage

def download_directory_from_gcs(bucket_name, gcs_path, local_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_path)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative_path = os.path.relpath(blob.name, gcs_path)
        local_file = os.path.join(local_path, relative_path)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        blob.download_to_filename(local_file)
        print(f"Downloaded gs://{bucket_name}/{blob.name} to {local_file}")

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
    parser.add_argument("--wandb-run", type=str, default="dummy", help="Wandb run name")
    parser.add_argument("--vertex-experiment", type=str, default="", help="Vertex AI experiment name")
    parser.add_argument("--vertex-tensorboard", type=str, default="", help="Vertex AI TensorBoard resource name")
    parser.add_argument("--device-batch-size", type=int, default=8, help="Batch size per device")
    
    args = parser.parse_args()

    # Parse bucket name and prefix
    if args.gcs_bucket.startswith("gs://"):
        bucket_name = args.gcs_bucket.replace("gs://", "").split("/")[0]
        prefix_parts = args.gcs_bucket.replace("gs://", "").split("/")[1:]
        prefix = "/".join(prefix_parts) if prefix_parts else ""
    else:
        bucket_name = args.gcs_bucket
        prefix = ""

    # Check if pretraining checkpoint already exists (checkpoint detection)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    gcs_base_ckpt_path = os.path.join(prefix, "base_checkpoints") if prefix else "base_checkpoints"
    
    # Check for model.pt (the key checkpoint file)
    # Note: base_train.py saves to f"d{depth}" where depth defaults to 20
    depth = 20
    gcs_base_ckpt_path = os.path.join(gcs_base_ckpt_path, f"d{depth}")
    checkpoint_exists = bucket.blob(os.path.join(gcs_base_ckpt_path, "model.pt")).exists()
    
    if checkpoint_exists:
        print(f"✓ Pretraining checkpoint already exists in gs://{bucket_name}/{gcs_base_ckpt_path}")
        print("Skipping pretraining (already completed)")
        return

    print(f"Pretraining checkpoint not found. Running pretraining...")

    # Set local base dir
    local_base_dir = "/tmp/nanochat"
    os.environ["NANOCHAT_BASE_DIR"] = local_base_dir
    os.makedirs(local_base_dir, exist_ok=True)

    # Set data dir to GCS so we stream/cache data there
    gcs_data_path = f"gs://{bucket_name}/{prefix}/base_data" if prefix else f"gs://{bucket_name}/base_data"
    # Clean up double slashes if any
    gcs_data_path = gcs_data_path.replace("//base_data", "/base_data")
    os.environ["NANOCHAT_DATA_DIR"] = gcs_data_path
    print(f"Set NANOCHAT_DATA_DIR to {gcs_data_path}")
    
    # Download tokenizer from GCS to local disk
    print("Downloading tokenizer from GCS...")
    gcs_tokenizer_path = os.path.join(prefix, "tokenizer") if prefix else "tokenizer"
    local_tokenizer_dir = os.path.join(local_base_dir, "tokenizer")
    download_directory_from_gcs(bucket_name, gcs_tokenizer_path, local_tokenizer_dir)

    try:
        # Diagnostic: Check if PyTorch can see CUDA
        import torch
        print(f"PRE-TRAINING DIAGNOSTICS:")
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"  torch.__version__: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  torch.version.cuda: {torch.version.cuda}")
            print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Print environment variables
        env_vars = ["LD_LIBRARY_PATH", "PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"]
        for var in env_vars:
            print(f"  env {var}: {os.environ.get(var, 'NOT SET')}")
        
        # We use a smaller batch size to be safe on standard GPUs, or rely on auto-config.
        # speedrun.sh uses d20.
        # A100 80GB: Use batch_size=32 for optimal MFU (uses ~38-40GB)
        # A100 40GB (Distributed): Use batch_size=8 per GPU.
        
        # Dynamic GPU detection
        import torch
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPUs. Configuring distributed training...")
        
        # Adjust batch size based on GPU type (heuristic)
        # If we are on A100 40GB, we need batch_size=8.
        # If we are on A100 80GB, we can use 32.
        # Since we are likely switching back to 40GB for distributed, let's be safe with 8.
        # The user can override this if needed, but 8 is safe for 40GB.
        # If we are on 80GB, 8 is also fine, just less efficient per GPU, but with multiple GPUs it's okay.
        # Let's stick to 8 to be safe for the 40GB distributed case.
        device_batch_size = "8" 
        
        print("Starting pretraining...")
        subprocess.run([
            "torchrun", "--standalone", f"--nproc_per_node={gpu_count}",
            "-m", "scripts.base_train",
            "--depth=20", f"--device_batch_size={args.device_batch_size}",
            f"--wandb_run_name={args.wandb_run}",
            f"--vertex_experiment={args.vertex_experiment}",
            f"--vertex_tensorboard={args.vertex_tensorboard}"
        ], check=True)

        # Evaluate the model on a larger chunk of train/val data and draw some samples.
        print("Running base_loss evaluation...")
        subprocess.run([
            "torchrun", "--standalone", "--nproc_per_node=1",
            "-m", "scripts.base_loss",
            "--device_batch_size=8"
        ], check=True)

        # Evaluate the model on CORE tasks.
        print("Running base_eval...")
        subprocess.run([
            "torchrun", "--standalone", "--nproc_per_node=1",
            "-m", "scripts.base_eval"
        ], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error during pretraining steps: {e}")
        raise

    # Upload checkpoints and report to GCS
    print("Uploading artifacts to GCS...")
    
    # Upload base_checkpoints
    local_checkpoints_dir = os.path.join(local_base_dir, "base_checkpoints")
    gcs_checkpoints_path = os.path.join(prefix, "base_checkpoints") if prefix else "base_checkpoints"
    if os.path.exists(local_checkpoints_dir):
        upload_directory_to_gcs(local_checkpoints_dir, bucket_name, gcs_checkpoints_path)
    else:
        print(f"Warning: {local_checkpoints_dir} does not exist.")

    # Upload report (it might be in base_dir or somewhere else, let's check report.py behavior or just upload everything in base_dir except data/tokenizer?)
    # report.py likely writes to a file.
    # For now, let's just upload the whole base_dir excluding data and tokenizer which we handled/don't need.
    # Actually, let's just look for report.md or similar.
    # But we don't know exactly where report.py writes.
    # Assuming it writes to base_dir/report.md or similar.
    
    # Let's just upload everything in local_base_dir that is NOT tokenizer or base_checkpoints (already uploaded) or tokenized_data.
    for root, dirs, files in os.walk(local_base_dir):
        # Skip directories we don't want to re-upload or are empty
        if "tokenizer" in dirs:
            dirs.remove("tokenizer")
        if "base_checkpoints" in dirs:
            dirs.remove("base_checkpoints")
        if "tokenized_data" in dirs:
            dirs.remove("tokenized_data")
            
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_base_dir)
            blob_path = os.path.join(prefix, relative_path) if prefix else relative_path
            blob = bucket.blob(blob_path)
            blob.upload_from_file(open(local_file, 'rb'))
            print(f"Uploaded {local_file} to gs://{bucket_name}/{blob_path}")

if __name__ == "__main__":
    main()
