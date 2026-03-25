"""
Verify external service access from environment variables.

Expected env vars:
- HF_TOKEN
- CLOUDFLARE_API_TOKEN
- CLOUDFLARE_ACCOUNT_ID
"""

import argparse
import os

import requests
from huggingface_hub import HfApi


def verify_hf(repo_id, token_env):
    token = os.environ.get(token_env)
    if not token:
        return False, f"Missing {token_env}"
    api = HfApi(token=token)
    info = api.model_info(repo_id)
    return True, f"HF access OK: {info.id}"


def verify_cloudflare(token_env):
    token = os.environ.get(token_env)
    if not token:
        return False, f"Missing {token_env}"
    response = requests.get(
        "https://api.cloudflare.com/client/v4/user/tokens/verify",
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success", False):
        return False, f"Cloudflare verify failed: {payload.get('errors')}"
    status = payload.get("result", {}).get("status")
    return True, f"Cloudflare token OK: status={status}"


def main():
    parser = argparse.ArgumentParser(description="Verify Hugging Face and Cloudflare access from env vars")
    parser.add_argument("--hf-repo-id", default="ManmohanSharma/nanochat-d24", help="HF repo id to verify")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="HF token env var")
    parser.add_argument("--cloudflare-token-env", default="CLOUDFLARE_API_TOKEN", help="Cloudflare token env var")
    args = parser.parse_args()

    hf_ok, hf_message = verify_hf(args.hf_repo_id, args.hf_token_env)
    cf_ok, cf_message = verify_cloudflare(args.cloudflare_token_env)

    print(hf_message)
    print(cf_message)
    if not (hf_ok and cf_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
