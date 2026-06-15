import unittest

from tasks.common import Task


class ListTask(Task):

    def __init__(self, examples, **kwargs):
        super().__init__(**kwargs)
        self.examples = examples

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.examples)

    def get_example(self, index):
        return self.examples[index]


class TestTaskIndexing(unittest.TestCase):

    def test_getitem_respects_logical_slice_bounds(self):
        task = ListTask(["a", "b", "c", "d"], start=1, stop=3)

        self.assertEqual(len(task), 2)
        self.assertEqual(task[0], "b")
        self.assertEqual(task[1], "c")
        with self.assertRaises(AssertionError):
            _ = task[-1]
        with self.assertRaises(AssertionError):
            _ = task[2]


if __name__ == "__main__":
    unittest.main()
