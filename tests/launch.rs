use std::process::Command;

#[test]
fn rust_launcher_runs_each_local_rank_and_propagates_success() {
    let output = Command::new(env!("CARGO_BIN_EXE_rust-agent"))
        .args(["launch", "--nproc-per-node", "2", "exec", "print(42)"])
        .output()
        .expect("start rust-agent launcher");

    assert!(
        output.status.success(),
        "launcher failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut lines = String::from_utf8(output.stdout)
        .expect("worker output is UTF-8")
        .lines()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines, ["42", "42"]);
}
