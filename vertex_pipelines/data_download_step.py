#!/usr/bin/env python3
"""
Data download step for Vertex AI Pipeline.
Downloads training data shards from HuggingFace and uploads to GCS.
"""
import argparse
import os
import subprocess
import tempfile
from google.cloud import storage


def download_and_upload_data(gcs_bucket: str, num_shards: int = 50):
    """
    Download training data shards and upload to GCS.
    
    Args:
        gcs_bucket: GCS bucket path (e.g., 'gs://nzp-nanochat')
        num_shards: Number of parquet shards to download (default: 50 for testing)
    """
    # Extract bucket name from gs:// path
    bucket_name = gcs_bucket.replace("gs://", "").split("/")[0]
    prefix = "/".join(gcs_bucket.replace("gs://", "").split("/")[1:]) if "/" in gcs_bucket.replace("gs://", "") else ""
    
    # Check if data already exists
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    gcs_data_path = f"{prefix}/base_data" if prefix else "base_data"
    
    blobs = list(bucket.list_blobs(prefix=gcs_data_path))
    parquet_blobs = [b for b in blobs if b.name.endswith('.parquet')]
    
    if len(parquet_blobs) >= num_shards:
        print(f"Found {len(parquet_blobs)} parquet files in gs://{bucket_name}/{gcs_data_path}")
        print(f"Skipping download as {num_shards} shards were requested and sufficient data exists.")
        return

    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Downloading {num_shards} data shards to {temp_dir}...")
        local_data_dir = os.path.join(temp_dir, "base_data")
        os.makedirs(local_data_dir, exist_ok=True)
        
        # Set environment variable for nanochat dataset module
        os.environ["NANOCHAT_DATA_DIR"] = local_data_dir
        
        # Download data using nanochat's dataset module
        print(f"Running: python -m nanochat.dataset -n {num_shards}")
        subprocess.run([
            "python", "-m", "nanochat.dataset", "-n", str(num_shards)
        ], check=True)
        
        # Upload to GCS
        print(f"Uploading data to gs://{bucket_name}/{prefix}/base_data/...")
        
        # Upload all parquet files
        parquet_files = [f for f in os.listdir(local_data_dir) if f.endswith('.parquet')]
        print(f"Found {len(parquet_files)} parquet files to upload")
        
        for i, filename in enumerate(parquet_files):
            local_path = os.path.join(local_data_dir, filename)
            gcs_path = f"{prefix}/base_data/{filename}" if prefix else f"base_data/{filename}"
            blob = bucket.blob(gcs_path)
            
            print(f"Uploading {i+1}/{len(parquet_files)}: {filename}")
            blob.upload_from_filename(local_path)
        
        print(f"Successfully uploaded {len(parquet_files)} data shards to GCS")
        
        # Verify upload
        gcs_data_path = f"gs://{bucket_name}/{prefix}/base_data" if prefix else f"gs://{bucket_name}/base_data"
        print(f"Data is now available at: {gcs_data_path}")
        
        # Download and upload eval bundle
        print("Downloading eval bundle from Karpathy's S3...")
        import urllib.request
        import zipfile
        eval_bundle_url = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
        eval_bundle_path = "/tmp/eval_bundle.zip"
        eval_bundle_extracted = "/tmp/eval_bundle"
        
        urllib.request.urlretrieve(eval_bundle_url, eval_bundle_path)
        print(f"Downloaded eval_bundle.zip to {eval_bundle_path}")
        
        # Extract and upload to GCS
        with zipfile.ZipFile(eval_bundle_path, 'r') as zip_ref:
            zip_ref.extractall("/tmp")
        
        # Upload eval_bundle directory to GCS
        print("Uploading eval bundle to GCS...")
        eval_bundle_files = []
        for root, dirs, files in os.walk(eval_bundle_extracted):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, "/tmp")
                eval_bundle_files.append((local_file_path, relative_path))
        
        for local_file_path, relative_path in eval_bundle_files:
            gcs_path = f"{prefix}/{relative_path}" if prefix else relative_path
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_file_path)
        
        print(f"Uploaded {len(eval_bundle_files)} eval bundle files to GCS")


def main():
    parser = argparse.ArgumentParser(description="Download and upload training data to GCS")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket path")
    parser.add_argument("--num-shards", type=int, default=50, help="Number of data shards to download")
    args = parser.parse_args()
    
    print("=" * 80)
    print("DATA DOWNLOAD STEP")
    print("=" * 80)
    print(f"GCS Bucket: {args.gcs_bucket}")
    print(f"Number of shards: {args.num_shards}")
    print("=" * 80)
    
    download_and_upload_data(args.gcs_bucket, args.num_shards)
    
    print("=" * 80)
    print("DATA DOWNLOAD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
