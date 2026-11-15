"""
Author: Mani Sarkar
Created: 2026-11-14
Scope: chat_sft resume — SftLoaderState round-trip, config-mismatch guard,
       should_save_checkpoint pure function, shell helpers (resume_flag / find-resume-step).
"""
import json
import os
import subprocess
import sys

import pytest

from nanochat.checkpoint_manager import (
    assert_resume_config_matches,
    apply_sft_resume_state,
    should_save_checkpoint,
)
from nanochat.sft_loader import SftLoaderState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fake_checkpoint(checkpoint_dir, step, world_size=1, extra_meta=None):
    """Write a minimal but structurally complete checkpoint so find_safe_resume_step sees it."""
    import torch
    os.makedirs(checkpoint_dir, exist_ok=True)
    meta = {"step": step, "val_bpb": 1.5, **(extra_meta or {})}
    with open(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"), "w") as f:
        json.dump(meta, f)
    torch.save({}, os.path.join(checkpoint_dir, f"model_{step:06d}.pt"))
    for r in range(world_size):
        torch.save({}, os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{r}.pt"))


# ---------------------------------------------------------------------------
# § should_save_checkpoint — pure function tests
# ---------------------------------------------------------------------------

class TestShouldSaveCheckpoint:
    """
    Scenario: Pure timing/step check, no DDP, no I/O.
    Slice: checkpoint_manager.should_save_checkpoint
    """

    def test_last_step_always_saves(self):
        # Given: last_step=True, any other condition
        # When: called
        # Then: returns True
        assert should_save_checkpoint(10, -1, 0.0, 300, 0, True) is True

    def test_step_zero_does_not_save(self):
        # Given: step=0
        # When: called
        # Then: returns False (step 0 is the initialisation step, not a training step)
        assert should_save_checkpoint(0, -1, 1000.0, 300, 0, False) is False

    def test_resume_step_skipped(self):
        # Given: step equals resume_from_step
        # When: called
        # Then: returns False (avoid double-saving the checkpoint we just loaded)
        assert should_save_checkpoint(5, 5, 1000.0, 300, 0, False) is False

    def test_time_due_triggers_save(self):
        # Given: elapsed >= save_every_secs, step > 0, not resume step, not last_step
        # When: called
        # Then: returns True
        assert should_save_checkpoint(3, -1, 350.0, 300, 0, False) is True

    def test_time_not_due_no_save(self):
        # Given: elapsed < save_every_secs, no step_due
        # When: called
        # Then: returns False
        assert should_save_checkpoint(3, -1, 100.0, 300, 0, False) is False

    def test_step_due_triggers_save(self):
        # Given: step is a multiple of save_every=10, elapsed not due
        # When: called
        # Then: returns True
        assert should_save_checkpoint(10, -1, 0.0, 300, 10, False) is True

    def test_step_not_divisible_no_save(self):
        # Given: step not a multiple of save_every, time not due
        # When: called
        # Then: returns False
        assert should_save_checkpoint(7, -1, 0.0, 300, 10, False) is False

    def test_save_every_zero_disables_step_trigger(self):
        # Given: save_every=0 (disabled), time not due
        # When: called
        # Then: returns False
        assert should_save_checkpoint(10, -1, 0.0, 300, 0, False) is False

    def test_save_every_secs_zero_disables_time_trigger(self):
        # Given: save_every_secs=0 (disabled), save_every=0 (disabled)
        # When: called on a non-last step
        # Then: returns False
        assert should_save_checkpoint(5, -1, 9999.0, 0, 0, False) is False

    def test_inf_elapsed_always_time_due(self):
        # Given: elapsed=inf (broadcast-agreed time_due on all DDP ranks)
        # When: called with save_every_secs finite
        # Then: returns True
        assert should_save_checkpoint(3, -1, float("inf"), 300, 0, False) is True

    def test_zero_elapsed_never_time_due(self):
        # Given: elapsed=0 (broadcast-agreed not time_due)
        # When: called with large save_every_secs and no step due
        # Then: returns False
        assert should_save_checkpoint(3, -1, 0.0, 300, 0, False) is False


# ---------------------------------------------------------------------------
# § SftLoaderState round-trip
# ---------------------------------------------------------------------------

class TestSftLoaderStateRoundTrip:
    """
    Scenario: Serialise and deserialise SftLoaderState; ensure the round-trip is lossless.
    Slice: nanochat.sft_loader.SftLoaderState
    """

    def test_initial_state_uses_ddp_rank(self):
        # Given: ddp_rank=2
        # When: initial() is called
        # Then: cursor and consumed start at ddp_rank
        s = SftLoaderState.initial(ddp_rank=2)
        assert s.cursor == 2
        assert s.consumed == 2
        assert s.epoch == 1
        assert s.it == 0
        assert s.conv_buffer == []

    def test_round_trip_empty_buffer(self):
        # Given: a state with no conv_buffer
        # When: serialised to dict and back
        # Then: all fields are preserved
        s = SftLoaderState(cursor=10, consumed=8, epoch=2, it=5, conv_buffer=[])
        s2 = SftLoaderState.from_dict(s.to_dict(), ddp_rank=0, ddp_world_size=1)
        assert s2.cursor == 10
        assert s2.consumed == 8
        assert s2.epoch == 2
        assert s2.it == 5
        assert s2.conv_buffer == []

    def test_round_trip_with_buffer(self):
        # Given: a state with a non-empty conv_buffer (token ids + loss masks)
        # When: serialised and deserialised
        # Then: buffer contents are preserved exactly
        buf = [[[1, 2, 3], [1, 1, 0]], [[4, 5], [0, 1]]]
        s = SftLoaderState(cursor=3, consumed=1, epoch=1, it=1, conv_buffer=buf)
        d = s.to_dict()
        s2 = SftLoaderState.from_dict(d, ddp_rank=0, ddp_world_size=1)
        assert s2.conv_buffer == buf

    def test_from_dict_missing_keys_use_defaults(self):
        # Given: an empty dict (e.g., fresh start, no saved state)
        # When: from_dict is called with ddp_rank=1
        # Then: defaults mirror initial() for that rank
        s = SftLoaderState.from_dict({}, ddp_rank=1, ddp_world_size=4)
        assert s.cursor == 1
        assert s.consumed == 1
        assert s.epoch == 1
        assert s.it == 0

    def test_json_serialisable(self):
        # Given: a state with a buffer
        # When: to_dict() output is json-serialised and parsed back
        # Then: values are unchanged (plain Python, no Tensors)
        buf = [[[10, 20], [1, 0]]]
        s = SftLoaderState(cursor=5, consumed=3, epoch=2, it=4, conv_buffer=buf)
        raw = json.dumps(s.to_dict())
        d = json.loads(raw)
        assert d["cursor"] == 5
        assert d["conv_buffer"] == buf

    @pytest.mark.parametrize("rank,saved_cursor,expected_cursor", [
        (0, 0, 0),   # rank 0 keeps its own cursor
        (1, 0, 1),   # rank 1 gets block_base + 1 = 0 + 1
        (2, 0, 2),   # rank 2 gets 0 + 2
        (3, 0, 3),   # rank 3 gets 0 + 3
        (1, 4, 5),   # saved cursor 4 → block_base 4; rank 1 → 5
        (2, 8, 10),  # saved cursor 8 → block_base 8; rank 2 → 10
    ])
    def test_ddp_ranks_get_distinct_cursors(self, rank, saved_cursor, expected_cursor):
        """
        Scenario: DDP resume — each rank re-derives its own cursor from rank-0's saved value.
        Slice: SftLoaderState.from_dict DDP re-derivation
        ### Given: world_size=4, checkpoint written by rank 0 with saved_cursor
        ### When: from_dict is called for this rank
        ### Then: cursor equals expected_cursor (distinct per rank)
        """
        # Given
        d = {"cursor": saved_cursor, "consumed": saved_cursor}
        # When
        s = SftLoaderState.from_dict(d, ddp_rank=rank, ddp_world_size=4)
        # Then
        assert s.cursor == expected_cursor

    def test_ddp_non_zero_rank_gets_empty_conv_buffer(self):
        """
        Scenario: conv_buffer belongs to rank 0 only; other ranks start empty.
        Slice: SftLoaderState.from_dict conv_buffer DDP
        ### Given: saved state has a non-empty conv_buffer
        ### When: from_dict is called for rank 1 of 2
        ### Then: conv_buffer is empty (rank 1 must refill from its own row group)
        """
        # Given
        d = {"cursor": 0, "consumed": 0, "conv_buffer": [[1, 2], [3, 4]]}
        # When
        s = SftLoaderState.from_dict(d, ddp_rank=1, ddp_world_size=2)
        # Then
        assert s.conv_buffer == []

    def test_ddp_rank0_keeps_saved_conv_buffer(self):
        """
        Scenario: rank 0 resumes exactly where it left off, including the conv_buffer.
        Slice: SftLoaderState.from_dict conv_buffer rank 0
        ### Given: saved state has a non-empty conv_buffer
        ### When: from_dict is called for rank 0 of 2
        ### Then: conv_buffer is preserved
        """
        # Given
        buf = [[1, 2], [3, 4]]
        d = {"cursor": 0, "consumed": 0, "conv_buffer": buf}
        # When
        s = SftLoaderState.from_dict(d, ddp_rank=0, ddp_world_size=2)
        # Then
        assert s.conv_buffer == buf

    def test_missing_cursor_defaults_to_rank_not_re_derived(self):
        """
        Scenario: No saved state; default cursor is rank-specific, not re-derived.
        Slice: SftLoaderState.from_dict empty-dict path
        ### Given: empty dict, rank=1, world_size=4
        ### When: from_dict is called
        ### Then: cursor == 1 (the default), not 2 (which the formula would give if applied)
        """
        # Given / When
        s = SftLoaderState.from_dict({}, ddp_rank=1, ddp_world_size=4)
        # Then
        assert s.cursor == 1
        assert s.consumed == 1

    def test_snapshot_replay_round_trip(self):
        """
        Scenario: Serialize and deserialize SftLoaderState; confirm lossless round-trip.
        Slice: nanochat.sft_loader.SftLoaderState snapshot-replay
        ### Given: a SftLoaderState with known values (cursor=5, consumed=3, epoch=2, it=10)
        ### When: serialized via to_dict() and deserialized via from_dict(d, ddp_rank=0)
        ### Then: all scalar fields match exactly
        """
        # Given
        original = SftLoaderState(cursor=5, consumed=3, epoch=2, it=10, conv_buffer=[])
        # When
        d = original.to_dict()
        restored = SftLoaderState.from_dict(d, ddp_rank=0, ddp_world_size=1)
        # Then
        assert restored.cursor == 5
        assert restored.consumed == 3
        assert restored.epoch == 2
        assert restored.it == 10
        assert restored.conv_buffer == []


# § Config-mismatch guard
# ---------------------------------------------------------------------------

_GUARD_KEYS = ("device_batch_size", "max_seq_len", "total_batch_size", "num_iterations")
_SAVED = {"device_batch_size": 8, "max_seq_len": 512, "total_batch_size": 16384, "num_iterations": 1500}


class TestConfigMismatchGuard:
    """
    Scenario: Resuming with changed training dimensions must raise ValueError.
    Slice: nanochat.checkpoint_manager.assert_resume_config_matches (called by scripts.chat_sft).
    """

    @pytest.mark.parametrize("changed_key,current", [
        ("device_batch_size", 16),
        ("max_seq_len", 1024),
        ("total_batch_size", 32768),
        ("num_iterations", 500),
    ])
    def test_mismatch_raises_naming_the_key(self, changed_key, current):
        """
        Scenario: The saved checkpoint and the CLI disagree on one key.
        Slice: assert_resume_config_matches
        ### Given: saved meta with the original values
        ### When: the current config changes one key
        ### Then: ValueError names that key
        """
        current_cfg = {**_SAVED, changed_key: current}
        with pytest.raises(ValueError, match=changed_key):
            assert_resume_config_matches(_SAVED, current_cfg, _GUARD_KEYS)

    def test_matching_config_does_not_raise(self):
        """
        Scenario: Config matches the checkpoint.
        Slice: assert_resume_config_matches
        ### Given: identical saved and current values
        ### When: the guard runs
        ### Then: no exception
        """
        assert_resume_config_matches(_SAVED, dict(_SAVED), _GUARD_KEYS)

    def test_key_missing_from_old_checkpoint_is_not_compared(self):
        """
        Scenario: An older checkpoint lacks a guard key.
        Slice: assert_resume_config_matches
        ### Given: saved meta without num_iterations
        ### When: the current num_iterations differs
        ### Then: no exception (the key cannot be compared)
        """
        saved = {k: v for k, v in _SAVED.items() if k != "num_iterations"}
        assert_resume_config_matches(saved, {**_SAVED, "num_iterations": 500}, _GUARD_KEYS)


# ---------------------------------------------------------------------------
# § Shell: find-resume-step CLI and resume_flag
# ---------------------------------------------------------------------------

class TestShellHelpers:
    """
    Scenario: Shell helpers correctly detect safe resume steps via the Python CLI.
    Slice: nanochat.checkpoint_manager find-resume-step CLI, runs/_stage_lib.sh resume_flag.
    """

    def _run(self, *args, cwd=None, env=None):
        result = subprocess.run(
            args, check=False, capture_output=True, text=True, cwd=cwd, env=env
        )
        return result

    def _stage_lib_env(self):
        """Environment that makes _stage_lib.sh use this session's venv Python."""
        return {**os.environ, "PYTHON": sys.executable}

    def test_find_resume_step_empty_dir(self, tmp_path):
        """
        Scenario: Empty checkpoint directory → returns -1.
        Slice: find-resume-step CLI
        ### Given: empty tmp_path
        ### When: CLI invoked
        ### Then: last line is '-1'
        """
        r = self._run(sys.executable, "-m", "nanochat.checkpoint_manager",
                      "find-resume-step", str(tmp_path), "1")
        assert r.returncode == 0
        assert r.stdout.strip().split("\n")[-1] == "-1"

    def test_find_resume_step_complete_checkpoint(self, tmp_path):
        """
        Scenario: Directory with one complete checkpoint → returns its step.
        Slice: find-resume-step CLI
        ### Given: checkpoint dir with step 7
        ### When: CLI invoked
        ### Then: last line is '7'
        """
        _write_fake_checkpoint(str(tmp_path), step=7)
        r = self._run(sys.executable, "-m", "nanochat.checkpoint_manager",
                      "find-resume-step", str(tmp_path), "1")
        assert r.returncode == 0
        assert r.stdout.strip().split("\n")[-1] == "7"

    def test_find_resume_step_missing_optim_skipped(self, tmp_path):
        """
        Scenario: Checkpoint missing optimiser shard is treated as unsafe.
        Slice: find-resume-step CLI
        ### Given: meta + model but no optim file for step 3
        ### When: CLI invoked
        ### Then: last line is '-1'
        """
        import torch
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(tmp_path / "meta_000003.json", "w") as f:
            json.dump({"step": 3}, f)
        torch.save({}, tmp_path / "model_000003.pt")
        # optim missing intentionally
        r = self._run(sys.executable, "-m", "nanochat.checkpoint_manager",
                      "find-resume-step", str(tmp_path), "1")
        assert r.returncode == 0
        assert r.stdout.strip().split("\n")[-1] == "-1"

    def test_resume_flag_emits_flag(self, tmp_path):
        """
        Scenario: resume_flag outputs --resume-from-step=N when a safe checkpoint exists.
        Slice: runs/_stage_lib.sh resume_flag
        ### Given: checkpoint dir with step 5
        ### When: resume_flag chat_sft <dir> is called
        ### Then: stdout contains '--resume-from-step=5'
        """
        _write_fake_checkpoint(str(tmp_path), step=5)
        stage_lib = os.path.join(os.path.dirname(__file__), "..", "runs", "_stage_lib.sh")
        r = self._run(
            "bash", "-c",
            f'source "{stage_lib}" && resume_flag chat_sft "{tmp_path}" 1',
            env=self._stage_lib_env(),
        )
        assert r.returncode == 0, r.stderr
        assert "--resume-from-step=5" in r.stdout

    def test_resume_flag_empty_dir_emits_nothing(self, tmp_path):
        """
        Scenario: No checkpoint → resume_flag emits nothing (fresh start).
        Slice: runs/_stage_lib.sh resume_flag
        ### Given: empty dir
        ### When: resume_flag called
        ### Then: stdout is empty
        """
        stage_lib = os.path.join(os.path.dirname(__file__), "..", "runs", "_stage_lib.sh")
        r = self._run(
            "bash", "-c",
            f'source "{stage_lib}" && resume_flag chat_sft "{tmp_path}" 1',
            env=self._stage_lib_env(),
        )
        assert r.returncode == 0
        assert r.stdout.strip() == ""

    def test_resume_flag_force_stages_suppresses_flag(self, tmp_path):
        """
        Scenario: FORCE_STAGES=chat_sft overrides the checkpoint → no --resume-from-step.
        Slice: runs/_stage_lib.sh resume_flag + stage_forced
        ### Given: checkpoint dir with step 5, FORCE_STAGES=chat_sft in env
        ### When: resume_flag called
        ### Then: stdout is empty
        """
        _write_fake_checkpoint(str(tmp_path), step=5)
        stage_lib = os.path.join(os.path.dirname(__file__), "..", "runs", "_stage_lib.sh")
        env = {**self._stage_lib_env(), "FORCE_STAGES": "chat_sft",
               "NANOCHAT_BASE_DIR": str(tmp_path)}
        r = self._run(
            "bash", "-c",
            f'source "{stage_lib}" && resume_flag chat_sft "{tmp_path}" 1',
            env=env,
        )
        assert r.returncode == 0
        assert "--resume-from-step" not in r.stdout


# ---------------------------------------------------------------------------
# B2 coverage — apply_sft_resume_state applies checkpoint to model and optimizer
# ---------------------------------------------------------------------------

class TestApplySftResumeState:
    """
    Scenario: chat_sft resume path restores SFT checkpoint weights into model and optimizer.
    Slice: nanochat/checkpoint_manager.py apply_sft_resume_state
    """

    def test_model_state_dict_is_loaded(self):
        """
        Scenario: apply_sft_resume_state calls load_state_dict on the model with checkpoint data.
        Slice: apply_sft_resume_state — model side
        ### Given: a model mock and a state dict
        ### When: apply_sft_resume_state is called
        ### Then: model.load_state_dict is called with the given model data
        """
        ### Given
        loaded = []
        class FakeModel:
            def load_state_dict(self, d): loaded.append(('model', d))
        class FakeOpt:
            def load_state_dict(self, d): loaded.append(('opt', d))
        model_data = {"weight": [1.0, 2.0]}
        optim_data = {"state": {}}

        ### When
        apply_sft_resume_state(FakeModel(), FakeOpt(), model_data, optim_data)

        ### Then
        assert loaded[0] == ('model', model_data), "model.load_state_dict must be called with checkpoint data"

    def test_optimizer_state_dict_is_loaded(self):
        """
        Scenario: apply_sft_resume_state calls load_state_dict on the optimizer with checkpoint data.
        Slice: apply_sft_resume_state — optimizer side
        ### Given: a model mock and an optimizer mock
        ### When: apply_sft_resume_state is called
        ### Then: optimizer.load_state_dict is called with the given optimizer data
        """
        ### Given
        loaded = []
        class FakeModel:
            def load_state_dict(self, d): loaded.append(('model', d))
        class FakeOpt:
            def load_state_dict(self, d): loaded.append(('opt', d))
        model_data = {"weight": [1.0]}
        optim_data = {"state": {"lr": 1e-4}}

        ### When
        apply_sft_resume_state(FakeModel(), FakeOpt(), model_data, optim_data)

        ### Then
        assert loaded[1] == ('opt', optim_data), "optimizer.load_state_dict must be called with checkpoint data"

    def test_deleting_function_breaks_these_tests(self):
        """
        Scenario: falsification — if apply_sft_resume_state is a no-op, tests fail.
        Slice: apply_sft_resume_state — falsifiability proof
        ### Given: a model that records whether load_state_dict was called
        ### When: apply_sft_resume_state is called
        ### Then: both model and optimizer recorded exactly one load_state_dict call
        """
        ### Given
        model_calls, opt_calls = [], []
        class FakeModel:
            def load_state_dict(self, d): model_calls.append(d)
        class FakeOpt:
            def load_state_dict(self, d): opt_calls.append(d)
        model_data = {"w": 42}
        optim_data = {"s": 99}

        ### When
        apply_sft_resume_state(FakeModel(), FakeOpt(), model_data, optim_data)

        ### Then
        assert len(model_calls) == 1 and model_calls[0] is model_data
        assert len(opt_calls) == 1 and opt_calls[0] is optim_data
