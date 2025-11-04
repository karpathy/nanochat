import os
import subprocess
import argparse
from nanochat.common import get_base_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket for artifacts")
    args = parser.parse_args()

    # Set the base directory to the GCS bucket.
    os.environ["NANOCHAT_BASE_DIR"] = args.gcs_bucket

    # Download the dataset.
    subprocess.run(["python", "-m", "nanochat.dataset", "-n", "8"], check=True)
    subprocess.run(["python", "-m", "nanochat.dataset", "-n", "240"], check=True)

    # Train the tokenizer.
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars=2000000000"], check=True)

    # Evaluate the tokenizer.
    subprocess.run(["python", "-m", "scripts.tok_eval"], check=True)

if __name__ == "__main__":
    main()
