"""
Test the sandboxed Python execution used by tool use / HumanEval.
Each test adversarially checks one property of the sandbox.

python -m pytest tests/test_execution.py -v
"""

import os
import time
import pytest
from nanochat.execution import execute_code


def test_happy_path():
    result = execute_code("print('hello world')")
    assert result.success
    assert result.stdout == "hello world\n"


def test_exception_is_captured():
    result = execute_code("1/0")
    assert not result.success
    assert "ZeroDivisionError" in result.error


def test_timeout_kills_infinite_loop():
    start = time.time()
    result = execute_code("while True: pass", timeout=2.0)
    elapsed = time.time() - start
    assert result.timeout and not result.success
    assert elapsed < 5, f"process was not killed promptly ({elapsed:.1f}s)"


def test_memory_limit():
    # 1GB allocation against the 256MB default limit
    result = execute_code("x = bytearray(1024 * 1024 * 1024)")
    assert not result.success
    assert result.memory_exceeded


@pytest.mark.parametrize("evil", [
    "import os; os.system('echo pwned')",
    "import shutil; shutil.rmtree('/tmp')",
    "import subprocess; subprocess.Popen(['ls'])",
    "import os; os.kill(1, 9)",
    "import os; os.fork()",
])
def test_destructive_functions_disabled(evil):
    result = execute_code(evil)
    assert not result.success


def test_stdin_disabled():
    result = execute_code("input()")
    assert not result.success


def test_writes_go_to_tempdir():
    result = execute_code("open('landmine.txt', 'w').write('x'); print(open('landmine.txt').read())")
    assert result.success and result.stdout == "x\n"
    assert not os.path.exists("landmine.txt"), "file leaked outside the sandbox tempdir"


def test_environment_is_scrubbed():
    os.environ["FAKE_SECRET_KEY"] = "hunter2"
    try:
        result = execute_code("import os; print('FAKE_SECRET_KEY' in os.environ)")
        assert result.success and result.stdout.strip() == "False"
    finally:
        del os.environ["FAKE_SECRET_KEY"]


def test_tricky_string_content():
    # quotes, backslashes and braces must survive being embedded into the runner
    tricky = '''s = 'it\\'s "quoted" \\\\ {braces} \\n'; print(len(s))'''
    result = execute_code(tricky)
    assert result.success


def test_fresh_globals():
    # user code must not see the guard prelude's imports
    result = execute_code("print('GUARD' in dir())")
    assert result.success and result.stdout.strip() == "False"
