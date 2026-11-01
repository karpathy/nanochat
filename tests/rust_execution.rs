use std::process::{Command, Output};

fn execute(code: &str, timeout_seconds: u64, memory_mib: u64) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rust-agent"))
        .args(["exec", "--timeout-seconds"])
        .arg(timeout_seconds.to_string())
        .args(["--memory-mib"])
        .arg(memory_mib.to_string())
        .arg(code)
        .output()
        .expect("launch rustAgent exec")
}

#[test]
fn embedded_python_worker_runs_code_with_fresh_globals_and_scrubbed_environment() {
    let output = Command::new(env!("CARGO_BIN_EXE_rust-agent"))
        .env("RUSTAGENT_EXEC_TEST_SECRET", "should-not-reach-worker")
        .args([
            "exec",
            "--memory-mib",
            "0",
            "import os; print('hello'); print(__import__('sys').argv); print('RUSTAGENT_EXEC_TEST_SECRET' in os.environ); print('sys' in dir())",
        ])
        .output()
        .expect("launch rustAgent exec");

    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "hello\n['-c']\nFalse\nFalse\n"
    );
}

#[test]
fn embedded_python_worker_reports_exceptions_and_eof_on_stdin() {
    let error = execute("1 / 0", 5, 0);
    assert!(!error.status.success());
    assert!(String::from_utf8_lossy(&error.stderr).contains("ZeroDivisionError"));

    let stdin = execute("input()", 5, 0);
    assert!(!stdin.status.success());
    assert!(String::from_utf8_lossy(&stdin.stderr).contains("EOFError"));
}

#[test]
fn embedded_python_worker_kills_timed_out_code() {
    let output = execute("while True: pass", 1, 0);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Execution timed out (process killed)"));
}

#[test]
fn embedded_python_worker_enforces_memory_limit() {
    let output = execute("x = bytearray(1024 * 1024 * 1024)", 5, 256);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("MemoryError"));
}

#[test]
fn embedded_python_worker_disables_destructive_process_and_filesystem_calls() {
    for code in [
        "import os; os.system('echo pwned')",
        "import shutil; shutil.rmtree('/tmp')",
        "import subprocess; subprocess.Popen(['ls'])",
        "import os; os.kill(1, 9)",
        "import os; os.fork()",
    ] {
        let output = execute(code, 5, 0);
        assert!(
            !output.status.success(),
            "unexpectedly allowed Python code: {code}"
        );
    }
}

#[test]
fn embedded_python_worker_cleans_up_its_temporary_working_directory() {
    let marker = format!("rustagent-exec-{}.txt", std::process::id());
    let code = format!("open({marker:?}, 'w').write('private'); print('done')");
    let output = execute(&code, 5, 0);

    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "done\n");
    assert!(!std::path::Path::new(&marker).exists());
}
