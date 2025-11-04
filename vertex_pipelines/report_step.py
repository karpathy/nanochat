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

    # Generate the full report.
    subprocess.run(["python", "-m", "nanochat.report", "generate"], check=True)

if __name__ == "__main__":
    main()
