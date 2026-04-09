import json
import tempfile
import unittest
from pathlib import Path

from tasks.customjson import CustomJSON


class CustomJSONTests(unittest.TestCase):
    def _write_jsonl(self, directory, name, lines):
        path = Path(directory) / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_loads_valid_conversations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_jsonl(
                tmpdir,
                "valid.jsonl",
                [
                    json.dumps(
                        [
                            {"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello"},
                        ]
                    )
                ],
            )

            task = CustomJSON(str(path))

        self.assertEqual(task.num_examples(), 1)
        self.assertEqual(task.get_example(0)["messages"][1]["content"], "Hello")

    def test_invalid_json_reports_file_and_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_jsonl(
                tmpdir,
                "broken.jsonl",
                [
                    json.dumps(
                        [
                            {"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello"},
                        ]
                    ),
                    '{"role": "user"',
                ],
            )

            with self.assertRaisesRegex(ValueError, rf"{path}:2: invalid JSON"):
                CustomJSON(str(path))

    def test_invalid_message_structure_reports_file_and_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_jsonl(
                tmpdir,
                "wrong-role.jsonl",
                [
                    json.dumps(
                        [
                            {"role": "assistant", "content": "Hi"},
                            {"role": "assistant", "content": "Hello"},
                        ]
                    )
                ],
            )

            with self.assertRaisesRegex(ValueError, rf"{path}:1: message 0 has role assistant but should be user"):
                CustomJSON(str(path))


if __name__ == "__main__":
    unittest.main()
