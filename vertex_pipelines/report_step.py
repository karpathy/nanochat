import os
import sys
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
    args = parser.parse_args()

    # Parse bucket name and prefix
    if args.gcs_bucket.startswith("gs://"):
        bucket_name = args.gcs_bucket.replace("gs://", "").split("/")[0]
        prefix_parts = args.gcs_bucket.replace("gs://", "").split("/")[1:]
        prefix = "/".join(prefix_parts) if prefix_parts else ""
    else:
        bucket_name = args.gcs_bucket
        prefix = ""

    # Check if report already exists (checkpoint detection)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    gcs_report_file = os.path.join(prefix, "report.md") if prefix else "report.md"
    
    report_exists = bucket.blob(gcs_report_file).exists()
    
    if report_exists:
        print(f"✓ Report already exists at gs://{bucket_name}/{gcs_report_file}")
        print("Skipping report generation (already completed)")
        return

    print(f"Report not found. Generating report...")

    # Set local base dir
    local_base_dir = "/tmp/nanochat"
    os.environ["NANOCHAT_BASE_DIR"] = local_base_dir
    os.makedirs(local_base_dir, exist_ok=True)

    # Download report dir from GCS
    print("Downloading report dir from GCS...")
    gcs_report_path = os.path.join(prefix, "report") if prefix else "report"
    local_report_dir = os.path.join(local_base_dir, "report")
    download_directory_from_gcs(bucket_name, gcs_report_path, local_report_dir)

    try:
        # Generate the full report.
        print("Generating report...")
        subprocess.run([sys.executable, "-m", "nanochat.report", "generate"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating report: {e}")
        raise

    # Upload report.md to GCS
    print("Uploading report to GCS...")
    # report.py generates report.md in local_base_dir/report/report.md AND copies it to current dir.
    # We want to upload it to the bucket root or prefix root.
    
    local_report_file = "report.md"
    if os.path.exists(local_report_file):
        blob_path = os.path.join(prefix, "report.md") if prefix else "report.md"
        bucket = storage.Client().bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_file(open(local_report_file, 'rb'))
        print(f"Uploaded {local_report_file} to gs://{bucket_name}/{blob_path}")
    else:
        print("Warning: report.md not found in current directory.")

    # Also upload the report dir just in case
    if os.path.exists(local_report_dir):
        upload_directory_to_gcs(local_report_dir, bucket_name, gcs_report_path)

if __name__ == "__main__":
    main()
