#!/bin/bash
#
# Fast IaC Data Scraper (Git Clone Method)
# 
# This script is 10-100x faster than API scraping because:
# 1. Git's protocol is optimized for bulk transfer
# 2. No API rate limits
# 3. No network latency per file
# 4. Saturates your network bandwidth
#
# Usage: bash dev/fast_scrape_iac.sh

set -e

OUTPUT_DIR="data/iac_raw_cloned"
TEMP_DIR="/tmp/iac_clones"

echo "========================================="
echo "Fast IaC Data Scraper (Git Clone Method)"
echo "========================================="
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"/{terraform,kubernetes,ansible,crossplane,docker}
mkdir -p "$TEMP_DIR"

# Top IaC repositories (curated list of high-quality repos)
declare -a REPOS=(
    # Terraform - AWS
    "terraform-aws-modules/terraform-aws-vpc"
    "terraform-aws-modules/terraform-aws-eks"
    "terraform-aws-modules/terraform-aws-rds"
    "terraform-aws-modules/terraform-aws-security-group"
    "terraform-aws-modules/terraform-aws-s3-bucket"
    "terraform-aws-modules/terraform-aws-iam"
    "cloudposse/terraform-aws-components"
    "gruntwork-io/terragrunt-infrastructure-live-example"
    
    # Terraform - GCP
    "terraform-google-modules/terraform-google-network"
    "terraform-google-modules/terraform-google-kubernetes-engine"
    "terraform-google-modules/terraform-google-sql-db"
    
    # Terraform - Azure
    "Azure/terraform-azurerm-aks"
    "Azure/terraform-azurerm-network"
    
    # Kubernetes manifests
    "kubernetes/examples"
    "kubernetes/website"
    "argoproj/argo-cd"
    "istio/istio"
    "prometheus-operator/kube-prometheus"
    "grafana/grafana"
    
    # Ansible
    "ansible/ansible-examples"
    "geerlingguy/ansible-for-devops"
    "elastic/ansible-elasticsearch"
    
    # Crossplane
    "crossplane/crossplane"
    "upbound/platform-ref-aws"
    "upbound/platform-ref-gcp"
)

echo "Will clone ${#REPOS[@]} repositories"
echo "Output: $OUTPUT_DIR"
echo ""

# Clone each repo
TOTAL_REPOS=${#REPOS[@]}
CURRENT=0

for repo in "${REPOS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[$CURRENT/$TOTAL_REPOS] Cloning $repo..."
    
    REPO_NAME=$(basename "$repo")
    CLONE_PATH="$TEMP_DIR/$REPO_NAME"
    
    # Skip if already cloned
    if [ -d "$CLONE_PATH" ]; then
        echo "  ✓ Already cloned, skipping..."
        continue
    fi
    
    # Clone with depth=1 (shallow clone, much faster)
    if git clone --depth 1 "https://github.com/$repo.git" "$CLONE_PATH" 2>&1 | grep -v "Cloning into"; then
        echo "  ✓ Cloned successfully"
    else
        echo "  ✗ Failed to clone, skipping..."
        continue
    fi
done

echo ""
echo "========================================="
echo "Extracting IaC Files..."
echo "========================================="
echo ""

# Extract IaC files from cloned repos
python3 << 'PYTHON_SCRIPT'
import os
import shutil
from pathlib import Path
from collections import defaultdict

TEMP_DIR = "/tmp/iac_clones"
OUTPUT_DIR = "data/iac_raw_cloned"

stats = defaultdict(int)

# File patterns
patterns = {
    "terraform": [".tf", ".tfvars"],
    "kubernetes": [".yaml", ".yml"],
    "ansible": [".yaml", ".yml"],
    "docker": ["Dockerfile"],
}

# Keywords to detect file type (for YAML disambiguation)
keywords = {
    "kubernetes": ["apiVersion", "kind"],
    "ansible": ["tasks:", "hosts:", "playbook"],
}

def detect_type(file_path):
    """Detect IaC type based on extension and content."""
    name = file_path.name.lower()
    
    # Terraform
    if name.endswith((".tf", ".tfvars")):
        return "terraform"
    
    # Docker
    if name == "dockerfile" or name.startswith("dockerfile."):
        return "docker"
    
    # YAML files need content inspection
    if name.endswith((".yaml", ".yml")):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(2000)  # Read first 2KB
                
                # Check for Kubernetes
                if "apiVersion" in content and "kind:" in content:
                    return "kubernetes"
                
                # Check for Ansible
                if "tasks:" in content or "hosts:" in content:
                    return "ansible"
                
                # Check for Crossplane
                if "crossplane" in content.lower():
                    return "crossplane"
        except:
            pass
    
    return None

def copy_iac_files():
    """Copy IaC files from cloned repos to organized output."""
    temp_path = Path(TEMP_DIR)
    output_path = Path(OUTPUT_DIR)
    
    if not temp_path.exists():
        print(f"Error: {TEMP_DIR} does not exist")
        return
    
    for repo_dir in temp_path.iterdir():
        if not repo_dir.is_dir() or repo_dir.name == ".git":
            continue
        
        print(f"Processing: {repo_dir.name}")
        
        # Walk through all files in the repo
        for file_path in repo_dir.rglob("*"):
            # Skip git files, tests, examples in some cases
            if any(skip in str(file_path) for skip in [".git/", "__pycache__", ".terraform/", "node_modules/"]):
                continue
            
            # Skip directories
            if not file_path.is_file():
                continue
            
            # Skip very large files (likely binary)
            try:
                if file_path.stat().st_size > 500_000:  # 500KB
                    continue
                if file_path.stat().st_size < 50:  # Too small
                    continue
            except:
                continue
            
            # Detect type
            iac_type = detect_type(file_path)
            if not iac_type:
                continue
            
            # Create safe filename
            rel_path = file_path.relative_to(repo_dir)
            safe_name = str(rel_path).replace("/", "_").replace("\\", "_")
            dest_file = f"{repo_dir.name}_{safe_name}"
            
            # Copy to output
            dest_path = output_path / iac_type / dest_file
            try:
                shutil.copy2(file_path, dest_path)
                stats[iac_type] += 1
            except Exception as e:
                print(f"  Error copying {file_path}: {e}")
    
    # Print statistics
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    total = sum(stats.values())
    for iac_type, count in sorted(stats.items()):
        pct = (count / total * 100) if total > 0 else 0
        print(f"{iac_type:15s}: {count:6d} files ({pct:5.1f}%)")
    print(f"{'TOTAL':15s}: {total:6d} files")
    print("="*60)

if __name__ == "__main__":
    copy_iac_files()
PYTHON_SCRIPT

echo ""
echo "========================================="
echo "Cleaning up..."
echo "========================================="
echo ""

# Optional: Remove cloned repos to save space
read -p "Delete cloned repos in $TEMP_DIR? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$TEMP_DIR"
    echo "✓ Cleaned up temporary files"
else
    echo "Keeping cloned repos in: $TEMP_DIR"
fi

echo ""
echo "========================================="
echo "🎉 Fast Scraping Complete!"
echo "========================================="
echo ""
echo "Files saved to: $OUTPUT_DIR"
echo ""
echo "Next step:"
echo "  python3 dev/repackage_iac_data.py \\"
echo "    --input-dir $OUTPUT_DIR \\"
echo "    --output-dir ~/.cache/nanochat/iac_data \\"
echo "    --include-synthetic --include-docs"
