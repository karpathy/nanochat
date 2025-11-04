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

    # Pre-train the d20 model.
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.base_train", "--",
        "--depth=20", f"--run={args.wandb_run}"
    ], check=True)

    # Evaluate the model on a larger chunk of train/val data and draw some samples.
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.base_loss"
    ], check=True)

    # Evaluate the model on CORE tasks.
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.base_eval"
    ], check=True)

if __name__ == "__main__":
    main()
