"""
Configuration loader.

Supports loading configuration from:
1. Python files (.py) - executed in a restricted context.
2. JSON files (.json) - supports flat dicts or `{"parameters": ...}` format.
3. Command-line arguments - `--key=value`.

Usage:
    from nanochat.configurator import get_config
    config = get_config(globals(), sys.argv[1:])
    globals().update(config)
"""

import os
import sys
import json
from ast import literal_eval

def print0(s="", **kwargs):
    """Print only if RANK is 0 (master process) or if RANK is not set."""
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def get_config(defaults: dict, argv: list = None) -> dict:
    """
    Parses configuration from files and command-line arguments.

    Args:
        defaults: A dictionary of default configuration values (e.g., globals()).
        argv: List of command-line arguments (default: sys.argv[1:]).

    Returns:
        A dictionary containing only the keys from `defaults` that were updated.
    """
    if argv is None:
        argv = sys.argv[1:]

    updates = {}

    for arg in argv:
        if '=' not in arg:
            # Assume it's a config file
            if arg.startswith('--'):
                continue # ignore flags like --help

            config_file = arg
            if not os.path.exists(config_file):
                print0(f"Warning: Config file {config_file} not found.")
                continue

            print0(f"Overriding config with {config_file}")

            # Load based on extension
            if config_file.endswith('.json'):
                with open(config_file, 'r') as f:
                    data = json.load(f)

                
                # Support the tune_system.py output format (from tuning-profiles branch)
                if "parameters" in data and isinstance(data["parameters"], dict):
                    file_config = data["parameters"]
                else:
                    file_config = data
            else:
                # Assume Python file (legacy support)
                # We execute it in a context seeded with defaults + previous updates
                # This ensures code in the config file (like `a = b * 2`) works if `b` is defined.
                context = defaults.copy()
                context.update(updates)
                with open(config_file) as f:
                    exec(f.read(), {}, context)

                # Extract only the keys that were in defaults (and thus valid config keys)
                file_config = {k: v for k, v in context.items() if k in defaults}

            # Apply updates from file
            for k, v in file_config.items():
                if k in defaults:
                    # Type checking logic could be refined here if strictness is needed
                    updates[k] = v
                else:
                    # Depending on strictness, we might want to warn or pass
                    pass

        else:
            # Command line argument --key=value
            if not arg.startswith('--'):
                continue

            key, val = arg.split('=', 1)
            key = key[2:]

            if key in defaults:
                try:
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    attempt = val

                # Type check
                default_val = defaults[key]
                if default_val is not None:
                    attempt_type = type(attempt)
                    default_type = type(default_val)
                    assert attempt_type == default_type, f"Type mismatch for {key}: {attempt_type} != {default_type}"

                print0(f"Overriding: {key} = {attempt}")
                updates[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}")

    return updates