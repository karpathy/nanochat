"""Exercise the real base-training CLI with local CPU-only fixtures."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import suppress
from pathlib import Path

import nanochat
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from nanochat.tokenizer import RustBPETokenizer


@unittest.skipUnless(os.name == "posix", "requires owned POSIX process-group cleanup")
class ResumeTrainingHorizonTests(unittest.TestCase):
    def fixture(self, root):
        tokenizer = RustBPETokenizer.train_from_iterator(["hello world " * 100], 270)
        tokenizer.save(str(root / "tokenizer"))
        torch.save(
            torch.ones(tokenizer.get_vocab_size(), dtype=torch.int32),
            root / "tokenizer/token_bytes.pt",
        )
        data = root / "base_data_climbmix"
        data.mkdir()
        for index in range(2):
            pq.write_table(
                pa.table({"text": ["hello world"] * 1000}),
                data / f"shard_{index:05d}.parquet",
            )
        status, output = self.run_cli(root, 2, resume=False)
        self.assertEqual(status, 0, output)
        checkpoint = root / "base_checkpoints/fixture"
        self.assertEqual(
            json.loads((checkpoint / "meta_000002.json").read_text())["step"], 2
        )
        optimizer = torch.load(checkpoint / "optim_000002_rank0.pt", weights_only=True)
        self.assertTrue(
            any(state.get("step") == 2 for state in optimizer["state"].values())
        )

    def run_cli(self, root, horizon, resume=True, evaluate=False):
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.base_train",
            "--device-type=cpu",
            "--depth=1",
            "--aspect-ratio=32",
            "--head-dim=16",
            "--max-seq-len=8",
            "--window-pattern=L",
            "--device-batch-size=1",
            "--total-batch-size=8",
            f"--num-iterations={horizon}",
            "--model-tag=fixture",
            f"--eval-every={1 if evaluate else -1}",
            "--eval-tokens=8",
            "--core-metric-every=-1",
            "--sample-every=-1",
        ]
        if resume:
            command.append("--resume-from-step=2")
        environment = dict(
            os.environ,
            NANOCHAT_BASE_DIR=str(root),
            TORCH_COMPILE_DISABLE="1",
            HF_HUB_OFFLINE="1",
            WANDB_MODE="disabled",
            OMP_NUM_THREADS="1",
            NANOCHAT_DTYPE="float32",
        )
        for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
            environment.pop(key, None)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
            cwd=Path(nanochat.__file__).resolve().parents[1],
        )
        try:
            output, _ = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            self.fail("CLI exceeded 20s; output:\n" + output[-12000:])
        finally:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
        return process.returncode, output

    def test_resume_beyond_horizon_is_rejected_before_an_optimizer_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            status, output = self.run_cli(root, 1)
            self.assertNotEqual(status, 0, output)
            self.assertIn(
                "Cannot resume from checkpoint step 2 beyond training horizon 1", output
            )
            self.assertNotIn("step 00002/", output)

    def test_resume_at_horizon_finishes_without_an_optimizer_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            status, output = self.run_cli(root, 2, evaluate=True)
            self.assertEqual(status, 0, output)
            self.assertNotIn("step 00002/", output)
            self.assertIn("Step 00002 | Validation bpb:", output)
            checkpoint = root / "base_checkpoints/fixture"
            self.assertTrue((checkpoint / "model_000002.pt").is_file())
            self.assertIsNotNone(
                json.loads((checkpoint / "meta_000002.json").read_text())["val_bpb"]
            )
            optimizer = torch.load(
                checkpoint / "optim_000002_rank0.pt", weights_only=True
            )
            self.assertTrue(
                any(state.get("step") == 2 for state in optimizer["state"].values())
            )

    def test_resume_before_horizon_trains_to_the_requested_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            status, output = self.run_cli(root, 3)
            self.assertEqual(status, 0, output)
            self.assertIn("step 00002/00003", output)
            self.assertNotIn("step 00003/00003", output)
            self.assertTrue(
                (root / "base_checkpoints/fixture/model_000003.pt").is_file()
            )
