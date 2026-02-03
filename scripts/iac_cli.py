#!/usr/bin/env python3
"""
IaC-GPT Command-Line Interface

A specialized CLI for Infrastructure-as-Code generation and auditing.
This tool pipes IaC-GPT output directly into .tf, .yaml, or other IaC files.

Usage:
    # Generate Terraform module
    python scripts/iac_cli.py generate --type terraform --service eks --output eks_cluster.tf
    
    # Audit existing infrastructure
    python scripts/iac_cli.py audit --path infrastructure/ --report audit_report.txt
    
    # Translate between tools
    python scripts/iac_cli.py translate --from ansible --to crossplane --input deploy.yml --output deploy.yaml
    
    # Interactive mode
    python scripts/iac_cli.py interactive
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List
import re


def load_model(model_path: Optional[str] = None):
    """Load the IaC-GPT model."""
    # Import nanochat modules
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from nanochat.engine import Engine
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.common import get_base_dir
    
    if model_path is None:
        # Try to find the latest IaC-GPT model
        base_dir = Path(get_base_dir())
        logs_dir = base_dir / "logs"
        
        # Look for models with "iac_gpt" in the name
        iac_models = sorted([d for d in logs_dir.glob("iac_gpt_*") if d.is_dir()])
        
        if iac_models:
            model_path = str(iac_models[-1] / "latest_checkpoint")
            print(f"Using model: {model_path}")
        else:
            print("ERROR: No IaC-GPT model found!")
            print("Please train a model first using: bash runs/speedrun_iac.sh")
            sys.exit(1)
    
    # Load tokenizer
    tokenizer_dir = Path(get_base_dir()) / "tokenizer"
    if not tokenizer_dir.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_dir}")
        sys.exit(1)
    
    tokenizer = RustBPETokenizer(str(tokenizer_dir))
    
    # Load model
    engine = Engine(model_path, tokenizer)
    
    return engine, tokenizer


def generate_iac(engine, tokenizer, iac_type: str, service: str, output_file: Optional[str] = None):
    """Generate Infrastructure-as-Code for a specific service."""
    
    # Craft the prompt based on IaC type and service
    prompts = {
        "terraform": {
            "eks": "# Create a production-ready AWS EKS cluster with the following:\n# - VPC with public and private subnets\n# - EKS control plane\n# - Node group with autoscaling\n# - Required IAM roles and policies\n\n",
            "vpc": "# Create an AWS VPC module with:\n# - Public and private subnets across 3 AZs\n# - Internet gateway for public subnets\n# - NAT gateway for private subnets\n# - Route tables\n\n",
            "rds": "# Create an AWS RDS PostgreSQL instance with:\n# - Multi-AZ deployment\n# - Automated backups\n# - Security groups\n# - Parameter group\n\n",
        },
        "kubernetes": {
            "deployment": "# Create a Kubernetes deployment for a web application with:\n# - 3 replicas\n# - Health checks\n# - Resource limits\n# - Rolling update strategy\n\napiVersion: apps/v1\nkind: Deployment\n",
            "service": "# Create a Kubernetes LoadBalancer service\n\napiVersion: v1\nkind: Service\n",
        },
        "ansible": {
            "deploy_app": "# Ansible playbook to deploy a web application\n# - Install dependencies\n# - Configure nginx\n# - Deploy application code\n# - Restart services\n\n---\n",
        },
    }
    
    # Get the prompt
    if iac_type not in prompts or service not in prompts[iac_type]:
        print(f"ERROR: Unknown combination: {iac_type}/{service}")
        print(f"Available types: {list(prompts.keys())}")
        return None
    
    prompt = prompts[iac_type][service]
    
    print(f"\n{'='*60}")
    print(f"Generating {iac_type.upper()} code for: {service}")
    print(f"{'='*60}\n")
    print("Prompt:")
    print(prompt)
    print("\nGenerating...")
    print("-" * 60)
    
    # Generate
    tokens = tokenizer.encode(prompt)
    max_new_tokens = 1024
    
    # Use the engine to generate
    generated_tokens = engine.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=50,
    )
    
    # Decode
    generated_text = tokenizer.decode(generated_tokens)
    
    # Extract just the generated part (after the prompt)
    generated_code = generated_text[len(prompt):]
    
    # Clean up (stop at common end markers)
    end_markers = ['\n\n\n\n', '# END', '# ---', 'User:', 'Assistant:']
    for marker in end_markers:
        if marker in generated_code:
            generated_code = generated_code.split(marker)[0]
    
    # Full output
    full_output = prompt + generated_code
    
    print(full_output)
    print("-" * 60)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_output)
        print(f"\n✅ Saved to: {output_file}")
    
    return full_output


def audit_iac(path: str, report_file: Optional[str] = None):
    """Audit existing IaC code for security issues and best practices."""
    
    print(f"\n{'='*60}")
    print(f"Auditing Infrastructure-as-Code at: {path}")
    print(f"{'='*60}\n")
    
    path_obj = Path(path)
    issues = []
    
    # Security patterns to detect
    security_patterns = {
        "public_s3": {
            "pattern": r'acl\s*=\s*"public-read"',
            "severity": "CRITICAL",
            "message": "Public S3 bucket detected - potential data exposure",
        },
        "open_ingress": {
            "pattern": r'cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]',
            "severity": "CRITICAL",
            "message": "0.0.0.0/0 ingress rule - unrestricted access",
        },
        "no_encryption": {
            "pattern": r'encrypted\s*=\s*false',
            "severity": "HIGH",
            "message": "Encryption disabled",
        },
        "missing_tags": {
            "pattern": r'resource\s+"aws_',
            "check_missing": r'tags\s*=',
            "severity": "MEDIUM",
            "message": "Missing resource tags",
        },
    }
    
    # Scan files
    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.glob("**/*.tf")) + list(path_obj.glob("**/*.yaml")) + list(path_obj.glob("**/*.yml"))
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            for issue_type, config in security_patterns.items():
                if re.search(config["pattern"], content):
                    # Check if it's a "missing" check
                    if "check_missing" in config:
                        if not re.search(config["check_missing"], content):
                            issues.append({
                                "file": str(file_path),
                                "severity": config["severity"],
                                "message": config["message"],
                                "type": issue_type,
                            })
                    else:
                        # Find line number
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines, 1):
                            if re.search(config["pattern"], line):
                                issues.append({
                                    "file": str(file_path),
                                    "line": line_num,
                                    "severity": config["severity"],
                                    "message": config["message"],
                                    "type": issue_type,
                                })
                                break
        
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
    
    # Print report
    if issues:
        print(f"\n🔍 Found {len(issues)} issue(s):\n")
        
        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        issues.sort(key=lambda x: severity_order.get(x["severity"], 4))
        
        for issue in issues:
            icon = "❌" if issue["severity"] == "CRITICAL" else "⚠️"
            location = f"{issue['file']}"
            if "line" in issue:
                location += f":{issue['line']}"
            
            print(f"{icon} {issue['severity']}: {issue['message']}")
            print(f"   Location: {location}\n")
    else:
        print("✅ No security issues found!\n")
    
    # Save report
    if report_file:
        with open(report_file, 'w') as f:
            f.write(f"IaC Security Audit Report\n")
            f.write(f"{'='*60}\n")
            f.write(f"Path: {path}\n")
            f.write(f"Files scanned: {len(files)}\n")
            f.write(f"Issues found: {len(issues)}\n\n")
            
            for issue in issues:
                f.write(f"{issue['severity']}: {issue['message']}\n")
                location = f"{issue['file']}"
                if "line" in issue:
                    location += f":{issue['line']}"
                f.write(f"Location: {location}\n\n")
        
        print(f"📄 Report saved to: {report_file}")
    
    return issues


def interactive_mode(engine, tokenizer):
    """Interactive chat mode for IaC questions."""
    
    print("\n" + "="*60)
    print("IaC-GPT Interactive Mode")
    print("="*60)
    print("\nAsk me anything about Infrastructure-as-Code!")
    print("Examples:")
    print("  - Create a Terraform module for an EKS cluster")
    print("  - How do I secure my Kubernetes deployments?")
    print("  - Show me an Ansible playbook to deploy nginx")
    print("\nType 'exit' to quit.\n")
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not user_input:
                continue
            
            # Build prompt with history
            prompt = ""
            for turn in conversation_history[-3:]:  # Last 3 turns for context
                prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
            
            prompt += f"User: {user_input}\nAssistant: "
            
            # Generate
            tokens = tokenizer.encode(prompt)
            generated_tokens = engine.generate(
                tokens,
                max_new_tokens=512,
                temperature=0.8,
                top_k=40,
            )
            
            generated_text = tokenizer.decode(generated_tokens)
            response = generated_text[len(prompt):].split('\n\n')[0]  # First paragraph
            
            print(f"\n🤖 IaC-GPT: {response}")
            
            # Save to history
            conversation_history.append({
                "user": user_input,
                "assistant": response,
            })
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="IaC-GPT CLI - Infrastructure-as-Code Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate IaC code")
    gen_parser.add_argument("--type", required=True, choices=["terraform", "kubernetes", "ansible"],
                           help="Type of IaC to generate")
    gen_parser.add_argument("--service", required=True, help="Service to generate (e.g., eks, vpc, rds)")
    gen_parser.add_argument("--output", help="Output file path")
    gen_parser.add_argument("--model", help="Model checkpoint path")
    
    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Audit existing IaC for security issues")
    audit_parser.add_argument("--path", required=True, help="Path to IaC files or directory")
    audit_parser.add_argument("--report", help="Save audit report to file")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive chat mode")
    interactive_parser.add_argument("--model", help="Model checkpoint path")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        engine, tokenizer = load_model(args.model)
        generate_iac(engine, tokenizer, args.type, args.service, args.output)
    
    elif args.command == "audit":
        # Audit doesn't need the model
        audit_iac(args.path, args.report)
    
    elif args.command == "interactive":
        engine, tokenizer = load_model(args.model)
        interactive_mode(engine, tokenizer)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
