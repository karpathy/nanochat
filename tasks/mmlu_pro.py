'''
The MMLU pro dataset.
https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

Compared to the original MMLU, there are three major differences:
    -> The original MMLU dataset only contains 4 options, MMLU-Pro increases it to 10 options.
    -> The original MMLU dataset contains mostly knowledge-driven questions without requiring much reasoning,
       MMLU-Pro contains more reasoning-driven questions, which are more challenging for LLMs.
    -> With 24 different prompt styles tested, the sensitivity of model scores to 
       prompt variations decreased from 4-5% in MMLU to just 2% in MMLU-Pro
'''

from datasets import load_dataset
from tasks.common import Task, render_mc

class MMLU_Pro(Task):

    letters = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in {"test","val"}, f"split : {split} must be test|val" 
        self.split = split
        self.ds = load_dataset("TIGER-lab/MMLU-Pro",split=split)

    @property
    def eval_type(self):
        return 'categorical'
    
    def num_examples(self):
        return len(self.ds)
    
    def get_example(self, index):
        row = self.ds[index] 
        question = row["question"] # question : str
        choices = row["options"] # options : list[str]

        assert len(choices)==10 # MMLU-pro = 10 choices

        user_message = render_mc(question, self.letters, choices) 
        assistant_message = row["answer"] # answer : str, e.g. "A", "B", ...

        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        
        ]
        conversation = {
            "messages": messages,
            "letters": self.letters
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters, f"MMLU answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation['messages'][-1]['content']
        return assistant_response == assistant_message