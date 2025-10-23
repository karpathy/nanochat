from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/ClimbMix",
    repo_type="dataset",
    allow_patterns=["climbmix_small/*.parquet"],
    local_dir="./",
    max_workers=16
)