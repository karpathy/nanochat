#!/usr/bin/env python3
"""
Verify that scheduling_strategy and max_wait_duration are runtime parameters
"""
import json
import sys

def verify_runtime_parameters():
    print("=== Verifying Runtime Scheduling Parameters ===\n")
    
    # Load compiled pipeline
    try:
        with open('nanochat_pipeline.json', 'r') as f:
            pipeline = json.load(f)
    except FileNotFoundError:
        print("❌ Error: nanochat_pipeline.json not found")
        print("   Run: python3 vertex_pipelines/pipeline.py --gcp-project nzp-nanochat ...")
        return False
    
    # Check root input parameters
    root_params = pipeline['root']['inputDefinitions']['parameters']
    
    print("✓ Pipeline root parameters:")
    for param in ['scheduling_strategy', 'max_wait_duration']:
        if param in root_params:
            info = root_params[param]
            print(f"  • {param}:")
            print(f"      Type: {info['parameterType']}")
            print(f"      Default: {info.get('defaultValue', 'N/A')}")
            print(f"      Optional: {info.get('isOptional', False)}")
        else:
            print(f"  ❌ Missing: {param}")
            return False
    
    print()
    
    # Check custom-training-job task parameters
    custom_job_task = pipeline['root']['dag']['tasks']['custom-training-job']
    task_params = custom_job_task['inputs']['parameters']
    
    print("✓ Custom Job task parameter bindings:")
    for param in ['strategy', 'max_wait_duration']:
        if param in task_params:
            binding = task_params[param]
            if 'componentInputParameter' in binding:
                print(f"  • {param} → {binding['componentInputParameter']}")
            elif 'runtimeValue' in binding:
                print(f"  ⚠ {param} → runtime constant (not parameterized!)")
                return False
        else:
            print(f"  ❌ Missing: {param}")
            return False
    
    print()
    print("=== Verification Summary ===")
    print("✅ scheduling_strategy is a RUNTIME parameter")
    print("✅ max_wait_duration is a RUNTIME parameter")
    print("✅ Both are correctly bound to Custom Job inputs")
    print()
    print("Benefits:")
    print("  • No recompilation needed to change FLEX_START ↔ SPOT ↔ STANDARD")
    print("  • No Docker rebuild needed for deployment strategy changes")
    print("  • Single pipeline JSON can be reused with different strategies")
    print()
    return True

if __name__ == "__main__":
    success = verify_runtime_parameters()
    sys.exit(0 if success else 1)
