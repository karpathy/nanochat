"""Discovered steps must map back to existing loader checkpoint filenames."""
import os
import tempfile
import unittest
from pathlib import Path

from nanochat.checkpoint_manager import find_last_step


class CheckpointDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def file(self, name):
        path = self.directory / name
        path.write_bytes(b"checkpoint fixture; discovery does not deserialize weights")
        return path

    def test_backup_names_do_not_select_a_nonexistent_loader_path(self):
        self.file("model_000001.pt")
        self.file("backup_model_999999.pt")
        self.file("model_999999_backup.pt")
        self.assertEqual(find_last_step(self.directory), 1)

    def test_a_matching_directory_is_not_a_checkpoint_file(self):
        self.file("model_000001.pt")
        (self.directory / "model_999999.pt").mkdir()
        self.assertEqual(find_last_step(self.directory), 1)

    def test_noncanonical_padding_does_not_select_a_different_filename(self):
        self.file("model_000010.pt")
        self.file("model_99.pt")
        self.file("model_0000099.pt")
        self.assertEqual(find_last_step(self.directory), 10)

    def test_numeric_order_and_more_than_six_digits_are_preserved(self):
        self.file("model_000000.pt")
        self.assertEqual(find_last_step(self.directory), 0)
        self.file("model_000009.pt")
        self.file("model_000010.pt")
        self.file("model_1000000.pt")
        self.assertEqual(find_last_step(self.directory), 1000000)

    @unittest.skipIf(os.name == "nt", "symlink fixture requires Unix permissions")
    def test_a_live_canonical_symlink_remains_discoverable(self):
        weights = self.file("weights.pt")
        (self.directory / "model_000002.pt").symlink_to(weights)
        self.assertEqual(find_last_step(self.directory), 2)

    @unittest.skipIf(os.name == "nt", "symlink fixture requires Unix permissions")
    def test_a_dangling_symlink_does_not_select_a_missing_file(self):
        self.file("model_000001.pt")
        (self.directory / "model_999999.pt").symlink_to(self.directory / "missing.pt")
        self.assertEqual(find_last_step(self.directory), 1)

    def test_no_eligible_checkpoint_keeps_the_existing_error(self):
        self.file("backup_model_999999.pt")
        self.file("model_99.pt")
        (self.directory / "model_888888.pt").mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "No checkpoints found"):
            find_last_step(self.directory)


if __name__ == "__main__":
    unittest.main()
