from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dev.mlx_logging import add_logging_args, write_summary_log

REPO = Path(__file__).resolve().parent.parent


PURE_COMPILE_SNIPPET = r'''
import json
import mlx
import mlx.core as mx

def f(x):
    return x * 2 + 1

compiled = mx.compile(f)
x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
y = compiled(x)
mx.eval(y)
print(json.dumps({
    "mlx_version": getattr(mlx, "__version__", "unknown"),
    "result": y.tolist(),
}))
'''


STATEFUL_COMPILE_SNIPPET = r'''
import json
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dev.mlx_compile_utils import capture_training_state, iter_optimizer_state

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = mx.array([[1.0]], dtype=mx.float32)

    def __call__(self, x):
        return x @ self.w

model = Tiny()
optimizer = optim.AdamW(learning_rate=0.1)
x = mx.array([[2.0]], dtype=mx.float32)
y = mx.array([[0.0]], dtype=mx.float32)
loss_and_grad = nn.value_and_grad(model, lambda batch, targets: mx.mean(mx.square(model(batch) - targets)))

def step():
    loss, grads = loss_and_grad(x, y)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters(), optimizer.state)
    return loss, model.w

compiled_step = mx.compile(step)
losses = []
weights = []
for _ in range(3):
    loss, weight = compiled_step()
    mx.eval(loss, weight, model.parameters(), optimizer.state)
    losses.append(float(loss))
    weights.append(float(weight.item()))

print(json.dumps({
    "mlx_version": getattr(mlx, "__version__", "unknown"),
    "losses": losses,
    "weights": weights,
}))
'''


EXPLICIT_STATEFUL_COMPILE_SNIPPET = r'''
import json
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dev.mlx_compile_utils import capture_training_state, iter_optimizer_state

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = mx.array([[1.0]], dtype=mx.float32)

    def __call__(self, x):
        return x @ self.w

model = Tiny()
optimizer = optim.AdamW(learning_rate=0.1)
x = mx.array([[2.0]], dtype=mx.float32)
y = mx.array([[0.0]], dtype=mx.float32)
loss_and_grad = nn.value_and_grad(model, lambda batch, targets: mx.mean(mx.square(model(batch) - targets)))
captured_state = capture_training_state(model, optimizer)

def step(batch, targets):
    loss, grads = loss_and_grad(batch, targets)
    optimizer.update(model, grads)
    return loss, model.w

compiled_step = mx.compile(
    step,
    inputs=captured_state,
    outputs=captured_state,
)
losses = []
weights = []
for _ in range(3):
    loss, weight = compiled_step(x, y)
    mx.eval(loss, weight, model.parameters(), *iter_optimizer_state(optimizer))
    losses.append(float(loss))
    weights.append(float(weight.item()))

print(json.dumps({
    "mlx_version": getattr(mlx, "__version__", "unknown"),
    "losses": losses,
    "weights": weights,
}))
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe whether the local MLX install supports compiled stateful training updates")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use for child probes")
    add_logging_args(parser)
    parser.set_defaults(log_prefix="phase4c_compiled_probe")
    return parser.parse_args()


def run_child(python_executable: str, code: str) -> dict[str, object]:
    completed = subprocess.run(
        [python_executable, "-c", code],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    payload: dict[str, object] = {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode == 0 and completed.stdout.strip():
        try:
            payload["parsed_stdout"] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            pass
    return payload


def main() -> None:
    args = parse_args()
    pure = run_child(args.python, PURE_COMPILE_SNIPPET)
    stateful = run_child(args.python, STATEFUL_COMPILE_SNIPPET)
    explicit_stateful = run_child(args.python, EXPLICIT_STATEFUL_COMPILE_SNIPPET)

    summary = {
        "python": args.python,
        "probes": {
            "pure_compile": pure,
            "stateful_compile": stateful,
            "explicit_stateful_compile": explicit_stateful,
        },
        "assessment": {
            "pure_compile_works": pure["returncode"] == 0,
            "stateful_compile_works": stateful["returncode"] == 0,
            "explicit_stateful_compile_works": explicit_stateful["returncode"] == 0,
            "supports_compiled_stateful_training": pure["returncode"] == 0 and explicit_stateful["returncode"] == 0,
        },
    }

    log_path = write_summary_log(
        summary,
        log_dir=args.log_dir,
        script_name="mlx_compiled_training_probe",
        log_prefix=args.log_prefix,
        depth=0,
        input_mode="probe",
    )
    summary["logging"] = {
        "log_dir": args.log_dir,
        "log_path": log_path,
    }
    if log_path is not None:
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()