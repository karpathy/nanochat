//! Process runner used by HumanEval execution and the standalone `exec` CLI.
//!
//! This mirrors nanoAgent's existing behavior: a fresh child process, scrubbed
//! environment, temporary working directory, captured output, timeout, and an
//! optional address-space limit. It is an accident-prevention boundary, not a
//! security sandbox for hostile code.

use std::{
    io::{Read, Write},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use serde::Serialize;
use tempfile::TempDir;

pub const DEFAULT_WORKER_MEMORY_MIB: u64 = 512;

#[derive(Debug, Serialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub error: Option<String>,
    pub timeout: bool,
    pub memory_exceeded: bool,
}

/// Run code in an isolated child process hosting the embedded RustPython VM.
pub fn execute_code(
    code: &str,
    timeout: Duration,
    maximum_memory_bytes: Option<u64>,
) -> Result<ExecutionResult> {
    let temp_dir = TempDir::new().context("create temporary execution directory")?;
    let executable = std::env::current_exe().context("resolve rustAgent executable")?;
    #[cfg(windows)]
    let worker_path = std::env::var_os("PATH").unwrap_or_default();
    #[cfg(not(windows))]
    let worker_path = std::ffi::OsString::from("/usr/bin:/bin");
    let mut command = Command::new(executable);
    command
        .arg("__python-worker")
        .current_dir(temp_dir.path())
        .env_clear()
        .env("PATH", worker_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    for name in ["SYSTEMROOT", "WINDIR", "TEMP", "TMP"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    // RustPython needs these Windows variables to start, but the user program
    // clears the process environment in its prelude before executing source.
    if let Some(memory_limit) = maximum_memory_bytes {
        command
            .arg("--memory-limit-bytes")
            .arg(memory_limit.to_string());
    }

    let mut child = command.spawn().context("start RustPython worker")?;
    let mut stdin = child.stdin.take().context("write RustPython source")?;
    stdin.write_all(code.as_bytes())?;
    drop(stdin);
    let mut stdout = child.stdout.take().context("capture Python stdout")?;
    let mut stderr = child.stderr.take().context("capture Python stderr")?;
    // Drain both pipes during execution so large outputs cannot block the child.
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stdout.read_to_end(&mut bytes);
        bytes
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stderr.read_to_end(&mut bytes);
        bytes
    });
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().context("wait for Python process")? {
            break status;
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Ok(ExecutionResult {
                success: false,
                stdout: String::new(),
                stderr: String::new(),
                error: Some("Execution timed out (process killed)".into()),
                timeout: true,
                memory_exceeded: false,
            });
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = decode_text(
        &stdout_reader
            .join()
            .map_err(|_| anyhow::anyhow!("stdout reader panicked"))?,
    );
    let stderr = decode_text(
        &stderr_reader
            .join()
            .map_err(|_| anyhow::anyhow!("stderr reader panicked"))?,
    );
    let success = status.success();
    let error = (!success).then(|| {
        stderr
            .trim()
            .lines()
            .last()
            .unwrap_or("Execution failed")
            .to_owned()
    });
    let memory_exceeded = stderr.contains("MemoryError");
    Ok(ExecutionResult {
        success,
        stdout,
        stderr,
        error,
        timeout: false,
        memory_exceeded,
    })
}

/// Match `subprocess.run(text=True)` universal-newline decoding on all hosts.
fn decode_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}
