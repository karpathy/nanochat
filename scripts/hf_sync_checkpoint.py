"""
Upload native nanochat checkpoints and related artifacts to a Hugging Face repo.

Examples:
python -m scripts.hf_sync_checkpoint --repo-id ManmohanSharma/nanochat-d24 --source base --model-tag d24_hf_import
python -m scripts.hf_sync_checkpoint --repo-id ManmohanSharma/nanochat-d24 --source base --model-tag d24_hf_import --step 0
"""

import argparse
import os

from huggingface_hub import HfApi

from nanochat.common import get_base_dir


def resolve_checkpoint_dir(source, model_tag):
    phase_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    return os.path.join(get_base_dir(), phase_dir, model_tag)


def main():
    parser = argparse.ArgumentParser(description="Upload native nanochat checkpoints to Hugging Face")
    parser.add_argument("--repo-id", required=True, help="Destination HF model repo")
    parser.add_argument("--source", choices=["base", "sft", "rl"], required=True, help="Checkpoint phase")
    parser.add_argument("--model-tag", required=True, help="Local nanochat model tag")
    parser.add_argument("--step", type=int, default=None, help="Optional specific step to upload")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing the HF token")
    parser.add_argument("--private", type=int, default=0, help="Create the repo as private if it does not exist")
    parser.add_argument("--repo-subdir", default="native_checkpoints", help="Subdirectory inside the repo")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise ValueError(f"Missing Hugging Face token in {args.token_env}")

    checkpoint_dir = resolve_checkpoint_dir(args.source, args.model_tag)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=bool(args.private), exist_ok=True)

    if args.step is None:
        path_in_repo = f"{args.repo_subdir}/{args.source}/{args.model_tag}"
        api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=args.repo_id,
            repo_type="model",
            path_in_repo=path_in_repo,
            commit_message=f"Upload native {args.source} checkpoint folder for {args.model_tag}",
        )
        print(f"Uploaded {checkpoint_dir} to {args.repo_id}:{path_in_repo}")
        return

    step_str = f"{args.step:06d}"
    files = [
        f"model_{step_str}.pt",
        f"meta_{step_str}.json",
    ]
    optimizer_pattern = f"optim_{step_str}_"
    for filename in sorted(os.listdir(checkpoint_dir)):
        if filename.startswith(optimizer_pattern) and filename.endswith(".pt"):
            files.append(filename)

    for filename in files:
        local_path = os.path.join(checkpoint_dir, filename)
        if not os.path.exists(local_path):
            continue
        path_in_repo = f"{args.repo_subdir}/{args.source}/{args.model_tag}/{filename}"
        api.upload_file(
            path_or_fileobj=local_path,
            repo_id=args.repo_id,
            repo_type="model",
            path_in_repo=path_in_repo,
            commit_message=f"Upload {args.source} checkpoint {args.model_tag} step {step_str}",
        )
    print(f"Uploaded step {step_str} for {args.model_tag} to {args.repo_id}")


if __name__ == "__main__":
    main()
