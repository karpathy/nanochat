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
    args = parser.parse_args()

    # Parse bucket name and prefix
    if args.gcs_bucket.startswith("gs://"):
        bucket_name = args.gcs_bucket.replace("gs://", "").split("/")[0]
        prefix_parts = args.gcs_bucket.replace("gs://", "").split("/")[1:]
        prefix = "/".join(prefix_parts) if prefix_parts else ""
    else:
        bucket_name = args.gcs_bucket
        prefix = ""

    # Check if SFT checkpoint already exists (checkpoint detection)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    gcs_sft_ckpt_path = os.path.join(prefix, "chatsft_checkpoints") if prefix else "chatsft_checkpoints"
    
    # Check for model.pt (the key checkpoint file)
    checkpoint_exists = bucket.blob(os.path.join(gcs_sft_ckpt_path, "model.pt")).exists()
    
    if checkpoint_exists:
        print(f"✓ SFT checkpoint already exists in gs://{bucket_name}/{gcs_sft_ckpt_path}")
        print("Skipping SFT training (already completed)")
        return

    print(f"SFT checkpoint not found. Running SFT training...")

    # Set local tmp dir for temporary files
    local_base_dir = "/tmp/nanochat"
    os.makedirs(local_base_dir, exist_ok=True)

    # Download tokenizer from GCS
    print("Downloading tokenizer from GCS...")
    gcs_tokenizer_path = os.path.join(prefix, "tokenizer") if prefix else "tokenizer"
    local_tokenizer_dir = os.path.join(local_base_dir, "tokenizer")
    download_directory_from_gcs(bucket_name, gcs_tokenizer_path, local_tokenizer_dir)

    # Download mid checkpoints from GCS
    print("Downloading mid checkpoints from GCS...")
    gcs_mid_checkpoints_path = os.path.join(prefix, "mid_checkpoints") if prefix else "mid_checkpoints"
    local_mid_checkpoints_dir = os.path.join(local_base_dir, "mid_checkpoints")
    download_directory_from_gcs(bucket_name, gcs_mid_checkpoints_path, local_mid_checkpoints_dir)

    # Download report dir from GCS
    print("Downloading report dir from GCS...")
    gcs_report_path = os.path.join(prefix, "report") if prefix else "report"
    local_report_dir = os.path.join(local_base_dir, "report")
    download_directory_from_gcs(bucket_name, gcs_report_path, local_report_dir)
    # Ensure report directory exists even if nothing was downloaded
    os.makedirs(local_report_dir, exist_ok=True)

    try:
        # Download the identity conversations dataset.
        print("Downloading identity conversations...")
        subprocess.run([
            "curl", "-L", "-o",
            f"{local_base_dir}/identity_conversations.jsonl",
            "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
        ], check=True)

        # Run supervised finetuning.
        print("Starting SFT...")
        env = os.environ.copy()
        env["NANOCHAT_BASE_DIR"] = local_base_dir
        subprocess.run([
            "torchrun", "--standalone", "--nproc_per_node=1",
            "-m", "scripts.chat_sft",
            f"--wandb_run_name={args.wandb_run}",
            f"--vertex_experiment={args.vertex_experiment}",
            f"--vertex_tensorboard={args.vertex_tensorboard}"
        ], check=True, env=env)

        # Evaluate the model.
        print("Running chat_eval (sft)...")
        subprocess.run([
            "torchrun", "--standalone", "--nproc_per_node=1",
            "-m", "scripts.chat_eval", "--",
            "-i", "sft"
        ], check=True, env=env)

    except subprocess.CalledProcessError as e:
        print(f"Error during SFT steps: {e}")
        raise

    # Upload checkpoints to GCS
    print("Uploading artifacts to GCS...")
    
    # Upload chatsft_checkpoints
    local_checkpoints_dir = os.path.join(local_base_dir, "chatsft_checkpoints")
    gcs_checkpoints_path = os.path.join(prefix, "chatsft_checkpoints") if prefix else "chatsft_checkpoints"
    if os.path.exists(local_checkpoints_dir):
        upload_directory_to_gcs(local_checkpoints_dir, bucket_name, gcs_checkpoints_path)
    else:
        print(f"Warning: {local_checkpoints_dir} does not exist.")

    # Upload report dir
    if os.path.exists(local_report_dir):
        upload_directory_to_gcs(local_report_dir, bucket_name, gcs_report_path)

if __name__ == "__main__":
    main()
