import os
import subprocess
import argparse
from nanochat.common import get_base_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket for artifacts")
    parser.add_argument("--wandb-run", type=str, default="dummy", help="Wandb run name")
    args = parser.parse_args()

    # Set the base directory to the GCS bucket.
    os.environ["NANOCHAT_BASE_DIR"] = args.gcs_bucket

    # Download the identity conversations dataset.
    subprocess.run([
        "curl", "-L", "-o",
        f"{get_base_dir()}/identity_conversations.jsonl",
        "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
    ], check=True)

    # Run mid-training.
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.mid_train", "--",
        f"--run={args.wandb_run}"
    ], check=True)

    # Evaluate the model.
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.chat_eval", "--",
        "-i", "mid"
    ], check=True)

if __name__ == "__main__":
    main()
