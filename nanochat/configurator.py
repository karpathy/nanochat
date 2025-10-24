"""
Poor Man's Configurator v3. Clean refactored version. Example usage:
$ python train.py config/override_file.py --batch_size=32

Improved version with better separation of concerns and cleaner architecture
while maintaining the same simple usage pattern.
"""

import os
import sys
from ast import literal_eval


def print0(s="", **kwargs):
    """Print only from rank 0 in distributed settings"""
    if int(os.environ.get("RANK", 0)) == 0:
        print(s, **kwargs)


class ConfigManager:
    """Clean configurator with explicit global injection control"""

    def __init__(self):
        self.config = {}

    def load(self, config_file=None, **overrides):
        """Load configuration from file and apply overrides"""
        if config_file:
            with open(config_file) as f:
                exec(f.read(), {}, self.config)
        self.config.update(overrides)
        return self

    def inject_globals(self):
        """Inject config vars into global namespace"""
        for k, v in self.config.items():
            if not k.startswith("_"):
                globals()[k] = v


def parse_args(args=None):
    """Parse command line arguments like original version"""
    args = args or sys.argv
    config_file, overrides = None, {}

    for arg in args[1:]:
        if "=" not in arg:
            # Config file
            assert not arg.startswith("--"), f"Invalid config file: {arg}"
            config_file = arg
            print0(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                print0(f.read())
        else:
            # Key=value override
            assert arg.startswith("--"), f"Override must start with --: {arg}"
            key, val = arg.split("=", 1)
            key = key[2:]
            
            if key not in globals():
                raise ValueError(f"Unknown config key: {key}")
            
            # Try to parse the value
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            
            # Type check if global has a non-None value
            if globals()[key] is not None:
                default_type, attempt_type = type(globals()[key]), type(attempt)
                assert attempt_type == default_type, f"Type mismatch: {attempt_type} != {default_type}"
            
            print0(f"Overriding: {key} = {attempt}")
            overrides[key] = attempt

    return config_file, overrides


# Execute configuration
config_file, overrides = parse_args()
ConfigManager().load(config_file, **overrides).inject_globals()
