import builtins
import io
import os
from contextlib import contextmanager

import pytest

import nanochat.execution as execution


def test_execution_result_repr_variants():
    r = execution.ExecutionResult(success=True, stdout="ok\n", stderr="")
    s = repr(r)
    assert "success=True" in s
    assert "stdout='ok\\n'" in s

    r2 = execution.ExecutionResult(success=False, stdout="", stderr="err", error="bad", timeout=True, memory_exceeded=True)
    s2 = repr(r2)
    assert "timeout=True" in s2
    assert "memory_exceeded=True" in s2
    assert "error='bad'" in s2
    assert "stderr='err'" in s2


def test_write_only_string_io_and_capture_io():
    s = execution.WriteOnlyStringIO()
    assert s.readable() is False
    with pytest.raises(IOError):
        s.read()
    with pytest.raises(IOError):
        s.readline()
    with pytest.raises(IOError):
        s.readlines()

    with execution.capture_io() as (out, err):
        print("hello")
    assert "hello" in out.getvalue()
    assert err.getvalue() == ""


def test_time_limit_and_chdir_and_tempdir(tmp_path):
    with execution.time_limit(0.5):
        x = 1 + 1
    assert x == 2

    with pytest.raises(execution.TimeoutException):
        with execution.time_limit(0.01):
            import time
            time.sleep(0.05)

    cwd = os.getcwd()
    with execution.chdir("."):
        assert os.getcwd() == cwd
    with execution.chdir(str(tmp_path)):
        assert os.getcwd() == str(tmp_path)
    assert os.getcwd() == cwd

    with execution.create_tempdir() as d:
        assert os.path.isdir(d)
    assert not os.path.exists(d)


def test_reliability_guard(monkeypatch):
    # Snapshot a small subset we assert was modified/restored in test cleanup.
    old_exit = builtins.exit
    old_quit = builtins.quit

    class FakePlatform:
        @staticmethod
        def uname():
            class U:
                system = "Linux"
            return U()

    calls = {"rlimit": 0, "disabled": 0}
    monkeypatch.setattr(execution, "platform", FakePlatform())

    class FakeResource:
        RLIMIT_AS = 1
        RLIMIT_DATA = 2
        RLIMIT_STACK = 3

        @staticmethod
        def setrlimit(_k, _v):
            calls["rlimit"] += 1

    import sys
    keys = ["ipdb", "joblib", "resource", "psutil", "tkinter"]
    old_modules = {k: sys.modules.get(k, None) for k in keys}

    class FakeOS:
        def __init__(self):
            self.environ = {}
            self.kill = lambda *a, **k: None
            self.system = lambda *a, **k: None
            self.putenv = lambda *a, **k: None
            self.remove = lambda *a, **k: None
            self.removedirs = lambda *a, **k: None
            self.rmdir = lambda *a, **k: None
            self.fchdir = lambda *a, **k: None
            self.setuid = lambda *a, **k: None
            self.fork = lambda *a, **k: None
            self.forkpty = lambda *a, **k: None
            self.killpg = lambda *a, **k: None
            self.rename = lambda *a, **k: None
            self.renames = lambda *a, **k: None
            self.truncate = lambda *a, **k: None
            self.replace = lambda *a, **k: None
            self.unlink = lambda *a, **k: None
            self.fchmod = lambda *a, **k: None
            self.fchown = lambda *a, **k: None
            self.chmod = lambda *a, **k: None
            self.chown = lambda *a, **k: None
            self.chroot = lambda *a, **k: None
            self.lchflags = lambda *a, **k: None
            self.lchmod = lambda *a, **k: None
            self.lchown = lambda *a, **k: None
            self.getcwd = lambda: "."
            self.chdir = lambda _p: None

    class FakeShutil:
        def __init__(self):
            self.rmtree = lambda *a, **k: None
            self.move = lambda *a, **k: None
            self.chown = lambda *a, **k: None

    class FakeSubprocess:
        def __init__(self):
            self.Popen = object

    monkeypatch.setitem(sys.modules, "resource", FakeResource)
    monkeypatch.setitem(sys.modules, "os", FakeOS())
    monkeypatch.setitem(sys.modules, "shutil", FakeShutil())
    monkeypatch.setitem(sys.modules, "subprocess", FakeSubprocess())
    monkeypatch.setattr(execution.faulthandler, "disable", lambda: calls.__setitem__("disabled", 1))
    execution.reliability_guard(1024)
    assert calls["rlimit"] == 3
    assert calls["disabled"] == 1
    assert builtins.exit is None
    assert builtins.quit is None

    # Restore minimal globals so current test process remains healthy.
    builtins.exit = old_exit
    builtins.quit = old_quit
    if isinstance(__builtins__, dict):
        __builtins__["help"] = help
    else:
        setattr(__builtins__, "help", help)
    for k, v in old_modules.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def test_unsafe_execute_success_and_error_paths(monkeypatch):
    # Avoid mutating process-wide dangerous globals during direct _unsafe_execute calls.
    monkeypatch.setattr(execution, "reliability_guard", lambda maximum_memory_bytes=None: None)

    out = {}
    execution._unsafe_execute("print('hi')", timeout=1.0, maximum_memory_bytes=None, result_dict=out)
    assert out["success"] is True
    assert "hi" in out["stdout"]
    assert out["stderr"] == ""

    out2 = {}
    execution._unsafe_execute("raise MemoryError('oom')", timeout=1.0, maximum_memory_bytes=None, result_dict=out2)
    assert out2["memory_exceeded"] is True
    assert "Memory limit exceeded" in out2["error"]

    out3 = {}
    execution._unsafe_execute("raise ValueError('bad')", timeout=1.0, maximum_memory_bytes=None, result_dict=out3)
    assert out3["success"] is False
    assert "ValueError: bad" in out3["error"]

    @contextmanager
    def boom(_seconds):
        raise execution.TimeoutException("Timed out!")
        yield  # pragma: no cover

    monkeypatch.setattr(execution, "time_limit", boom)
    out4 = {}
    execution._unsafe_execute("print('x')", timeout=1.0, maximum_memory_bytes=None, result_dict=out4)
    assert out4["timeout"] is True
    assert out4["error"] == "Execution timed out"


def test_execute_code_paths(monkeypatch):
    # Normal path with real machinery.
    r = execution.execute_code("print('ok')", timeout=1.0)
    assert r.success is True
    assert "ok" in r.stdout

    # p.is_alive() path.
    class FakeDict(dict):
        pass

    class FakeManager:
        def dict(self):
            return FakeDict()

    class FakeProcessAlive:
        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.killed = False

        def start(self):
            return None

        def join(self, timeout):
            del timeout
            return None

        def is_alive(self):
            return True

        def kill(self):
            self.killed = True

    monkeypatch.setattr(execution.multiprocessing, "Manager", lambda: FakeManager())
    monkeypatch.setattr(execution.multiprocessing, "Process", FakeProcessAlive)
    r2 = execution.execute_code("print(1)")
    assert r2.timeout is True
    assert "process killed" in r2.error

    # Empty result_dict path.
    class FakeProcessDone(FakeProcessAlive):
        def is_alive(self):
            return False

    monkeypatch.setattr(execution.multiprocessing, "Process", FakeProcessDone)
    r3 = execution.execute_code("print(2)")
    assert r3.success is False
    assert "no result returned" in r3.error
