"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import os
import sys
from ast import literal_eval

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

for i, arg in enumerate(sys.argv[1:]):
    if '=' not in arg:
        # assume it's the name of a config file, unless it's a flag-like argument
        if not arg.startswith('-'):
            config_file = arg
            print0(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                print0(f.read())
            exec(open(config_file).read())
        else:
            # it's a flag-like argument, e.g. -i mid or --task-name MMLU
            # we will assume it is handled by argparse and skip it
            pass
    else:
        # assume it's a --key=value argument
        if not arg.startswith('--'):
            continue # ignore
        key, val = arg.split('=', 1)
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            if globals()[key] is not None:
                attempt_type = type(attempt)
                default_type = type(globals()[key])
                if attempt_type != default_type:
                    print0(f"Warning: type mismatch for {key}. Overriding {default_type} with {attempt_type}")
            # cross fingers
            print0(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            print0(f"Warning: unknown config key: {key}")
