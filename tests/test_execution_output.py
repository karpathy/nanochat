"""Output decoding regressions; runnable without the training dependencies.

python -m unittest discover -s tests -p test_execution_output.py -v
"""

import unittest
from nanochat.execution import execute_code


class ExecutionOutputTests(unittest.TestCase):
    def test_invalid_stdout_bytes_do_not_escape_parent(self):
        result = execute_code("import sys; sys.stdout.buffer.write(b'before\\xffafter')")
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "before\ufffdafter")
        self.assertEqual(result.stderr, "")
        self.assertIsNone(result.error)

    def test_invalid_stderr_bytes_do_not_escape_parent(self):
        result = execute_code("import sys; sys.stderr.buffer.write(b'before\\xffafter')")
        self.assertTrue(result.success)
        self.assertEqual(result.stderr, "before\ufffdafter")
        self.assertIsNone(result.error)

    def test_invalid_output_preserves_execution_failure(self):
        result = execute_code(
            "import sys; sys.stdout.buffer.write(b'\\xff'); "
            "sys.stderr.buffer.write(b'\\xfe\\n'); raise ValueError('bad result')"
        )
        self.assertFalse(result.success)
        self.assertEqual(result.stdout, "\ufffd")
        self.assertIn("\ufffd\n", result.stderr)
        self.assertEqual(result.error, "ValueError: bad result")
        self.assertFalse(result.timeout)
        self.assertFalse(result.memory_exceeded)

    def test_valid_utf8_is_preserved_on_both_streams(self):
        result = execute_code(
            "import sys; sys.stdout.buffer.write('你好 🌍'.encode('utf-8')); "
            "sys.stderr.buffer.write('café'.encode('utf-8'))"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "你好 🌍")
        self.assertEqual(result.stderr, "café")

    def test_incomplete_utf8_at_end_of_stream(self):
        result = execute_code("import sys; sys.stdout.buffer.write(b'prefix\\xe2\\x82')")
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "prefix\ufffd")


if __name__ == "__main__":
    unittest.main()
