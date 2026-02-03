"""
IaC Data Scraper for IaC-GPT Training

This script scrapes high-quality Infrastructure-as-Code from GitHub for:
- Terraform (.tf, .tfvars)
- Kubernetes (.yaml, .yml for k8s manifests)
- Ansible (.yml, .yaml for playbooks)
- Crossplane (.yaml for compositions)
- Dockerfiles

The goal is to build a curated 70% primary corpus for training IaC-GPT.

Usage:
    1. Set your GITHUB_TOKEN environment variable
    2. Run: python dev/scrape_iac_data.py --output-dir data/iac_raw --max-repos 500

The script will:
    - Search for high-quality repos with IaC content
    - Filter by stars/recent activity to ensure quality
    - Download raw files and save to organized directories
    - Track progress and support resumption
"""

import os
import sys
import time
import json
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime, timedelta
from collections import defaultdict

# GitHub API Configuration
GITHUB_API = "https://api.github.com"
HEADERS = {}  # Will be populated with token if available


# Quality Filters
MIN_STARS = 10  # Minimum stars to consider a repo
MIN_FILE_SIZE = 100  # Minimum file size in bytes (filter out trivial files)
MAX_FILE_SIZE = 500_000  # Maximum file size in bytes (filter out massive generated files)

# IaC File Patterns
IAC_PATTERNS = {
    "terraform": {
        "extensions": [".tf", ".tfvars"],
        "keywords": ["resource", "module", "variable", "output", "provider"],
        "exclude_paths": ["examples/", "test/", ".terragrunt"],
    },
    "kubernetes": {
        "extensions": [".yaml", ".yml"],
        "keywords": ["apiVersion", "kind", "metadata", "spec"],
        "require_keywords": ["apiVersion", "kind"],  # Must have these
        "exclude_paths": [".github/", "test/", "tests/"],
    },
    "ansible": {
        "extensions": [".yml", ".yaml"],
        "keywords": ["hosts:", "tasks:", "roles:", "playbook"],
        "require_keywords": ["tasks:"],  # Ansible playbooks must have tasks
        "exclude_paths": [".github/", "test/", "tests/"],
    },
    "crossplane": {
        "extensions": [".yaml", ".yml"],
        "keywords": ["apiVersion: apiextensions.crossplane.io", "Composition", "CompositeResourceDefinition"],
        "require_keywords": ["crossplane"],
        "exclude_paths": [".github/", "test/", "tests/"],
    },
    "docker": {
        "extensions": ["Dockerfile"],
        "keywords": ["FROM", "RUN", "COPY", "WORKDIR"],
        "exclude_paths": ["test/", "tests/", "examples/"],
    },
}


class IaCDataScraper:
    def __init__(self, output_dir: str, github_token: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each IaC type
        for iac_type in IAC_PATTERNS.keys():
            (self.output_dir / iac_type).mkdir(exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.output_dir / "progress.json"
        self.progress = self.load_progress()
        
        # Statistics
        self.stats = defaultdict(int)
        
        # GitHub API setup
        if github_token:
            # Use Bearer for fine-grained tokens (github_pat_*), token for classic (ghp_*)
            auth_type = "Bearer" if github_token.startswith("github_pat_") else "token"
            HEADERS["Authorization"] = f"{auth_type} {github_token}"
            HEADERS["Accept"] = "application/vnd.github.v3+json"
            print(f"✓ GitHub token configured ({auth_type} auth)")
        else:
            print("WARNING: No GitHub token provided. You'll be limited to 60 requests/hour.")
            print("Set GITHUB_TOKEN environment variable for 5000 requests/hour.")
    
    def load_progress(self) -> Dict:
        """Load progress from previous runs to support resumption."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "scraped_repos": set(),
            "file_count": defaultdict(int),
            "last_updated": None,
        }
    
    def save_progress(self):
        """Save current progress."""
        # Convert sets to lists for JSON serialization
        progress_serializable = {
            "scraped_repos": list(self.progress["scraped_repos"]),
            "file_count": dict(self.progress["file_count"]),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress_serializable, f, indent=2)
    
    def github_search(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search GitHub repositories with rate limiting."""
        results = []
        page = 1
        per_page = 30
        
        while len(results) < max_results:
            try:
                url = f"{GITHUB_API}/search/repositories"
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": per_page,
                    "page": page,
                }
                
                response = requests.get(url, headers=HEADERS, params=params, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 403:
                    reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                    wait_seconds = max(reset_time - time.time(), 60)
                    print(f"Rate limited. Waiting {wait_seconds:.0f} seconds...")
                    time.sleep(wait_seconds)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if not data.get('items'):
                    break
                
                results.extend(data['items'])
                page += 1
                
                # Be nice to GitHub
                time.sleep(2)
                
            except Exception as e:
                print(f"Error searching GitHub: {e}")
                break
        
        return results[:max_results]
    
    def get_repo_tree(self, owner: str, repo: str, branch: str = "main") -> List[Dict]:
        """Get the file tree of a repository."""
        try:
            # Try main first, then master
            for branch_name in [branch, "master", "main"]:
                url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch_name}?recursive=1"
                response = requests.get(url, headers=HEADERS, timeout=30)
                
                if response.status_code == 200:
                    return response.json().get('tree', [])
                elif response.status_code == 404 and branch_name != "main":
                    continue
                else:
                    break
            
            return []
        except Exception as e:
            print(f"Error getting repo tree for {owner}/{repo}: {e}")
            return []
    
    def download_file(self, owner: str, repo: str, path: str, branch: str = "main") -> str:
        """Download raw file content from GitHub."""
        try:
            # Use raw.githubusercontent.com for raw content
            for branch_name in [branch, "master", "main"]:
                url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch_name}/{path}"
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 404 and branch_name != "main":
                    continue
                else:
                    break
            
            return None
        except Exception as e:
            print(f"Error downloading {owner}/{repo}/{path}: {e}")
            return None
    
    def is_valid_iac_file(self, content: str, iac_type: str) -> bool:
        """Validate if file content matches IaC type characteristics."""
        if not content or len(content) < MIN_FILE_SIZE:
            return False
        
        if len(content) > MAX_FILE_SIZE:
            return False
        
        patterns = IAC_PATTERNS[iac_type]
        
        # Check for required keywords
        if "require_keywords" in patterns:
            if not any(keyword.lower() in content.lower() for keyword in patterns["require_keywords"]):
                return False
        
        # Check for general keywords (at least one should match)
        keyword_matches = sum(1 for kw in patterns["keywords"] if kw.lower() in content.lower())
        if keyword_matches < 1:
            return False
        
        return True
    
    def categorize_file(self, path: str, content: str) -> str:
        """Determine which IaC category a file belongs to."""
        path_lower = path.lower()
        
        # Check each IaC type
        for iac_type, patterns in IAC_PATTERNS.items():
            # Check if path should be excluded
            if any(exclude in path_lower for exclude in patterns.get("exclude_paths", [])):
                continue
            
            # Check extension match
            if any(path.endswith(ext) for ext in patterns["extensions"]):
                # For YAML files, we need to differentiate between K8s, Ansible, and Crossplane
                if iac_type in ["kubernetes", "ansible", "crossplane"]:
                    if self.is_valid_iac_file(content, iac_type):
                        return iac_type
                else:
                    # For Terraform and Docker, extension is enough
                    if self.is_valid_iac_file(content, iac_type):
                        return iac_type
        
        return None
    
    def scrape_repo(self, repo_info: Dict) -> int:
        """Scrape IaC files from a single repository."""
        owner = repo_info['owner']['login']
        repo_name = repo_info['name']
        full_name = f"{owner}/{repo_name}"
        
        # Skip if already processed
        if full_name in self.progress.get("scraped_repos", set()):
            print(f"Skipping {full_name} (already scraped)")
            return 0
        
        print(f"\n{'='*60}")
        print(f"Scraping: {full_name}")
        print(f"Stars: {repo_info.get('stargazers_count', 0)} | "
              f"Updated: {repo_info.get('updated_at', 'Unknown')}")
        print(f"{'='*60}")
        
        # Get repository file tree
        tree = self.get_repo_tree(owner, repo_name, repo_info.get('default_branch', 'main'))
        
        files_saved = 0
        for item in tree:
            if item['type'] != 'blob':  # Only process files, not directories
                continue
            
            path = item['path']
            
            # Download file content
            content = self.download_file(owner, repo_name, path, repo_info.get('default_branch', 'main'))
            if not content:
                continue
            
            # Categorize the file
            iac_type = self.categorize_file(path, content)
            if not iac_type:
                continue
            
            # Save the file
            # Create a unique filename: repo_owner_reponame_filepath
            safe_path = path.replace('/', '_').replace('\\', '_')
            filename = f"{owner}_{repo_name}_{safe_path}"
            output_path = self.output_dir / iac_type / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            files_saved += 1
            self.progress["file_count"][iac_type] = self.progress["file_count"].get(iac_type, 0) + 1
            self.stats[iac_type] += 1
            
            print(f"  [{iac_type}] {path} ({len(content)} bytes)")
            
            # Rate limiting
            time.sleep(0.5)
        
        # Mark repo as scraped
        if "scraped_repos" not in self.progress:
            self.progress["scraped_repos"] = set()
        self.progress["scraped_repos"].add(full_name)
        self.save_progress()
        
        print(f"\nSaved {files_saved} files from {full_name}")
        return files_saved
    
    def scrape(self, max_repos: int = 100):
        """Main scraping orchestration."""
        print("Starting IaC Data Scraper")
        print(f"Output directory: {self.output_dir}")
        print(f"Target repositories: {max_repos}")
        print()
        
        # Define search queries for each IaC type
        queries = [
            "terraform aws stars:>50 language:HCL",
            "terraform gcp stars:>20 language:HCL",
            "terraform azure stars:>20 language:HCL",
            "kubernetes manifest stars:>30",
            "ansible playbook stars:>30",
            "crossplane composition stars:>5",
            "dockerfile stars:>50",
        ]
        
        all_repos = []
        for query in queries:
            print(f"Searching: {query}")
            repos = self.github_search(query, max_results=max_repos // len(queries))
            all_repos.extend(repos)
            time.sleep(2)
        
        # Deduplicate by full_name
        unique_repos = {f"{r['owner']['login']}/{r['name']}": r for r in all_repos}
        all_repos = list(unique_repos.values())
        
        # Filter by stars
        all_repos = [r for r in all_repos if r.get('stargazers_count', 0) >= MIN_STARS]
        
        # Sort by stars descending
        all_repos.sort(key=lambda x: x.get('stargazers_count', 0), reverse=True)
        
        print(f"\nFound {len(all_repos)} unique repositories to scrape")
        print()
        
        # Scrape each repository
        total_files = 0
        for idx, repo in enumerate(all_repos[:max_repos], 1):
            print(f"\nProgress: {idx}/{min(len(all_repos), max_repos)}")
            try:
                files_saved = self.scrape_repo(repo)
                total_files += files_saved
            except Exception as e:
                print(f"Error scraping {repo['full_name']}: {e}")
                continue
        
        # Print final statistics
        print("\n" + "="*60)
        print("SCRAPING COMPLETE")
        print("="*60)
        print(f"Total files collected: {total_files}")
        for iac_type, count in self.stats.items():
            print(f"  {iac_type}: {count} files")
        print(f"\nAll files saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Scrape IaC code from GitHub for IaC-GPT training")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/iac_raw",
        help="Output directory for scraped files (default: data/iac_raw)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=500,
        help="Maximum number of repositories to scrape (default: 500)",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )
    
    args = parser.parse_args()
    
    # Get GitHub token from args or environment
    github_token = args.github_token or os.getenv("GITHUB_TOKEN")
    
    # Create scraper and run
    scraper = IaCDataScraper(args.output_dir, github_token)
    scraper.scrape(max_repos=args.max_repos)


if __name__ == "__main__":
    main()
