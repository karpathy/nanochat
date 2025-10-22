"""
Test configurator.py usage patterns based on actual script usage.
Tests the exec-based configurator like it's used in scripts.
"""
import tempfile
import os
import sys
import subprocess


def test_configurator_exec_usage():
    """Test configurator.py like it's used in actual scripts"""
    
    # Create a config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
# Training config
batch_size = 64
learning_rate = 0.001
max_steps = 1000
eval_every = 100
""")
        config_file = f.name
    
    try:
        # Test script that mimics how scripts use configurator
        test_script = f"""
import os
import sys
sys.path.insert(0, r'{os.getcwd()}')

# Default configuration (like in scripts)
batch_size = 32
learning_rate = 0.01
max_steps = 500
eval_every = 50
device = "cuda"
run = "test"

# Capture config keys before configurator (like scripts do)
config_keys = [
    k for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]

# Mock command line args
sys.argv = ['script.py', r'{config_file}', '--batch_size=128', '--eval_every=200']

# Execute configurator (like scripts do)
exec(open(os.path.join("nanochat", "configurator.py")).read())

# Check final values
print(f"batch_size={{batch_size}}")
print(f"learning_rate={{learning_rate}}")
print(f"max_steps={{max_steps}}")
print(f"eval_every={{eval_every}}")
print(f"device={{device}}")
print(f"run={{run}}")

# Verify config keys are preserved
user_config = {{k: globals()[k] for k in config_keys}}
print(f"config_keys_count={{len(user_config)}}")
"""
        
        result = subprocess.run([sys.executable, '-c', test_script],
                              capture_output=True, text=True)
        
        if result.stderr:
            print(f"Error: {result.stderr}")
        
        # Verify the configurator worked correctly
        assert "batch_size=128" in result.stdout  # CLI override
        assert "learning_rate=0.001" in result.stdout  # From config file
        assert "max_steps=1000" in result.stdout  # From config file
        assert "eval_every=200" in result.stdout  # CLI override
        assert "device=cuda" in result.stdout  # Unchanged default
        assert "run=test" in result.stdout  # Unchanged default
        assert "config_keys_count=6" in result.stdout  # All 6 config vars tracked
        
    finally:
        os.unlink(config_file)


def test_configurator_torchrun_style():
    """Test configurator.py with torchrun-style arguments"""
    
    test_script = f"""
import os
import sys
sys.path.insert(0, r'{os.getcwd()}')

# Set up DDP environment variables (like torchrun does)
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

# Default training config
device_batch_size = 16
sequence_length = 1024
learning_rate = 0.0003
warmup_steps = 100
max_steps = 10000

# Mock torchrun command line
sys.argv = ['script.py', '--device_batch_size=32', '--max_steps=5000']

# Execute configurator
exec(open(os.path.join("nanochat", "configurator.py")).read())

print(f"device_batch_size={{device_batch_size}}")
print(f"sequence_length={{sequence_length}}")
print(f"learning_rate={{learning_rate}}")
print(f"warmup_steps={{warmup_steps}}")
print(f"max_steps={{max_steps}}")
"""
    
    result = subprocess.run([sys.executable, '-c', test_script],
                          capture_output=True, text=True)
    
    if result.stderr:
        print(f"Error: {result.stderr}")
    
    # Verify overrides worked
    assert "device_batch_size=32" in result.stdout  # Override
    assert "sequence_length=1024" in result.stdout  # Default
    assert "learning_rate=0.0003" in result.stdout  # Default
    assert "warmup_steps=100" in result.stdout  # Default
    assert "max_steps=5000" in result.stdout  # Override


def test_configurator_print0_ddp():
    """Test that print0 works correctly in DDP environment"""
    
    # Test rank 0 (should print)
    test_script_rank0 = f"""
import os
import sys
sys.path.insert(0, r'{os.getcwd()}')

os.environ['RANK'] = '0'
batch_size = 32

sys.argv = ['script.py', '--batch_size=64']
exec(open(os.path.join("nanochat", "configurator.py")).read())
"""
    
    result_rank0 = subprocess.run([sys.executable, '-c', test_script_rank0],
                                capture_output=True, text=True)
    
    # Test rank 1 (should not print)
    test_script_rank1 = f"""
import os
import sys
sys.path.insert(0, r'{os.getcwd()}')

os.environ['RANK'] = '1'
batch_size = 32

sys.argv = ['script.py', '--batch_size=64']
exec(open(os.path.join("nanochat", "configurator.py")).read())
"""
    
    result_rank1 = subprocess.run([sys.executable, '-c', test_script_rank1],
                                capture_output=True, text=True)
    
    # Rank 0 should show override message
    assert "Overriding: batch_size = 64" in result_rank0.stdout
    
    # Rank 1 should be silent (no output)
    assert result_rank1.stdout.strip() == ""


def test_configurator_type_checking():
    """Test type checking like original configurator"""
    
    test_script = f"""
import os
import sys
sys.path.insert(0, r'{os.getcwd()}')

os.environ['RANK'] = '0'
batch_size = 32  # int

try:
    sys.argv = ['script.py', '--batch_size=hello']  # string override
    exec(open(os.path.join("nanochat", "configurator.py")).read())
    print("ERROR: Should have failed")
except AssertionError as e:
    print("SUCCESS: Type mismatch caught")
"""
    
    result = subprocess.run([sys.executable, '-c', test_script],
                          capture_output=True, text=True)
    
    assert "Type mismatch caught" in result.stdout

