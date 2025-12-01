"""
Japanese instruction-following dataset from izumi-lab.
https://huggingface.co/datasets/izumi-lab/llm-japanese-dataset

This dataset contains 9M+ Japanese instruction-output pairs,
converted to the conversation format used by nanochat for SFT.
"""

from datasets import load_dataset
from tasks.common import Task


class JapaneseInstruct(Task):
    """
    Japanese instruction-following dataset.
    Converts instruction/input/output format to messages format.
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        # The dataset only has a "train" split
        assert split == "train", "JapaneseInstruct only has 'train' split"
        self.ds = load_dataset("izumi-lab/llm-japanese-dataset", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        instruction = row.get("instruction", "") or ""
        input_text = row.get("input", "") or ""
        output = row.get("output", "") or ""

        # Combine instruction and input
        if input_text.strip():
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        # Build conversation in messages format
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]

        conversation = {
            "messages": messages,
        }
        return conversation
