//! Embedded RustPython runtime used by the isolated `exec` worker.

use std::io::{self, Read};

use anyhow::{Context, Result};
use rustpython::{InterpreterBuilder, InterpreterBuilderExt};

const RUNNER_PRELUDE: &str = r#"
import builtins, os, sys
sys.argv = ["-c"]
try:
    import faulthandler
    faulthandler.disable()
except ImportError:
    pass
builtins.exit = None
builtins.quit = None
builtins.help = None
os.environ.clear()
os.environ["PATH"] = "/usr/bin:/bin"
os.environ["OMP_NUM_THREADS"] = "1"
for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
             "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
             "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
             "lchflags", "lchmod", "lchown", "getcwd", "chdir"):
    if hasattr(os, name): setattr(os, name, None)
try:
    import shutil
    for name in ("rmtree", "move", "chown"):
        if hasattr(shutil, name): setattr(shutil, name, None)
except ImportError:
    pass
try:
    import subprocess
    subprocess.Popen = None
except ImportError:
    pass
for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
    sys.modules[name] = None
"#;

/// Read one source file from stdin, execute it, and return a process exit code.
pub fn run_worker(memory_limit_bytes: Option<u64>) -> Result<u8> {
    let mut source = String::new();
    io::stdin().read_to_string(&mut source)?;
    let source_literal = serde_json::to_string(&source)?;
    let interpreter = InterpreterBuilder::new().init_stdlib().interpreter();
    if let Some(error) = interpreter.enter(|vm| vm.run_simple_string(RUNNER_PRELUDE).err()) {
        return Ok(interpreter.finalize(Some(error)).min(u8::MAX as u32) as u8);
    }
    // The CPython guard imports its support modules before setting rlimits.
    // Initialize RustPython and apply the equivalent prelude first, otherwise
    // the VM's startup allocations consume the user program's entire budget.
    apply_memory_limit(memory_limit_bytes)?;
    let program = format!(
        "exec(compile({source_literal}, \"<llm>\", \"exec\"), {{\"__name__\": \"__main__\"}})"
    );
    let exit_code = interpreter.run(|vm| vm.run_simple_string(&program).map(|_| ()));
    Ok(exit_code.min(u8::MAX as u32) as u8)
}

#[cfg(all(unix, not(target_os = "macos")))]
fn apply_memory_limit(limit: Option<u64>) -> Result<()> {
    let Some(limit) = limit.and_then(|value| libc::rlim_t::try_from(value).ok()) else {
        return Ok(());
    };
    let resource_limit = libc::rlimit {
        rlim_cur: limit,
        rlim_max: limit,
    };
    for (resource, name) in [
        (libc::RLIMIT_AS, "RLIMIT_AS"),
        (libc::RLIMIT_DATA, "RLIMIT_DATA"),
        (libc::RLIMIT_STACK, "RLIMIT_STACK"),
    ] {
        if unsafe { libc::setrlimit(resource, &resource_limit) } != 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("set execution memory limit {name}"));
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn apply_memory_limit(_limit: Option<u64>) -> Result<()> {
    // nanochat.execution skips rlimits on Darwin because those limits are not
    // reliable there; keep the embedded worker's behavior aligned.
    Ok(())
}

#[cfg(windows)]
fn apply_memory_limit(limit: Option<u64>) -> Result<()> {
    use std::{mem::size_of, ptr::null};
    use windows_sys::Win32::{
        Foundation::CloseHandle,
        System::{
            JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_JOB_MEMORY,
                JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                JobObjectExtendedLimitInformation, SetInformationJobObject,
            },
            Threading::GetCurrentProcess,
        },
    };

    let Some(limit) = limit.and_then(|value| usize::try_from(value).ok()) else {
        return Ok(());
    };

    // A job object supplies the process memory ceiling that setrlimit provides
    // on Unix. The worker is already a child process, so the limit cannot
    // affect the model server or the Rust parent.
    let job = unsafe { CreateJobObjectW(null(), null()) };
    if job.is_null() {
        return Err(std::io::Error::last_os_error()).context("create execution memory-limit job");
    }
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    limits.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY;
    limits.ProcessMemoryLimit = limit;
    limits.JobMemoryLimit = limit;
    let configured = unsafe {
        SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
    };
    if configured == 0 {
        let error = std::io::Error::last_os_error();
        unsafe { CloseHandle(job) };
        return Err(error).context("set execution memory limit");
    }
    let assigned = unsafe { AssignProcessToJobObject(job, GetCurrentProcess()) };
    if assigned == 0 {
        let error = std::io::Error::last_os_error();
        unsafe { CloseHandle(job) };
        return Err(error).context("assign execution worker to memory-limit job");
    }
    unsafe { CloseHandle(job) };
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn apply_memory_limit(_limit: Option<u64>) -> Result<()> {
    Ok(())
}
