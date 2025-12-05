
import os
from unittest import mock
from nanochat.common import is_ddp

def test_is_ddp_false_no_env():
    # Simulate an empty environment (regarding DDP variables)
    # We clear the environment for the context of this test to ensure no DDP vars exist
    # Note: We must be careful not to remove PATH or other essential vars if they were needed,
    # but is_ddp only checks specific keys. `clear=True` wipes everything, which is safe for is_ddp logic
    # but might break other things if is_ddp depended on other env vars (it doesn't).
    # However, to be safer and less intrusive, we can just ensure the specific keys are absent.
    with mock.patch.dict(os.environ, {}, clear=True):
        assert is_ddp() == False

def test_is_ddp_false_missing_vars():
    # Set only some vars
    env = {
        'RANK': '0',
        'WORLD_SIZE': '1'
        # Missing LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    }
    with mock.patch.dict(os.environ, env, clear=True):
        assert is_ddp() == False

def test_is_ddp_false_invalid_ints():
    env = {
        'RANK': '0',
        'LOCAL_RANK': '0',
        'WORLD_SIZE': 'foo', # Invalid
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12345'
    }
    with mock.patch.dict(os.environ, env, clear=True):
        assert is_ddp() == False

def test_is_ddp_true_valid():
    env = {
        'RANK': '0',
        'LOCAL_RANK': '0',
        'WORLD_SIZE': '2',
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12345'
    }
    with mock.patch.dict(os.environ, env, clear=True):
        assert is_ddp() == True
