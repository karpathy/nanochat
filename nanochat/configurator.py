"""
This script provides a simple, unconventional configuration management system for nanochat.
It is not a standard Python module but is instead executed directly using `exec`,
allowing it to modify the global scope of the calling script. This design choice
prioritizes simplicity and avoids the complexity of more formal configuration
libraries.

The configurator supports two types of overrides:
1.  **Configuration files:** A Python file can be provided as a command-line
    argument. The configurator will execute this file, which can be used to set
    default configuration values.
2.  **Command-line arguments:** Key-value pairs in the format `--key=value` can
    be provided to override specific settings. The script attempts
    to infer the correct data type for the value (e.g., int, float, bool).

Example usage:
$ python train.py config/override_file.py --batch_size=32

A more conventional approach would use a library like `argparse` or `hydra`.
"""

import os
import sys
from ast import literal_eval

def print0(s="",**kwargs):
    """Prints a message only on the main process (rank 0)."""
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

# Parse command-line arguments for configuration
for arg in sys.argv[1:]:
    if '=' not in arg:
        # If the argument does not contain '=', it is assumed to be a config file.
        assert not arg.startswith('--')
        config_file = arg
        print0(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print0(f.read())
        exec(open(config_file).read())
    else:
        # If the argument contains '=', it is assumed to be a key-value override.
        assert arg.startswith('--')
        key, val = arg.split('=', 1)
        key = key[2:]
        if key in globals():
            try:
                # Attempt to evaluate the value to infer its type (e.g., int, bool).
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # If evaluation fails, treat the value as a string.
                attempt = val
            # Ensure that the overridden value has the same type as the default.
            if globals()[key] is not None:
                attempt_type = type(attempt)
                default_type = type(globals()[key])
                assert attempt_type == default_type, f"Type mismatch for key '{key}': expected {default_type}, got {attempt_type}"
            # Update the global variable with the new value.
            print0(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
