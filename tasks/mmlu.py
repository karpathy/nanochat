#-*--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_#
#                                                                             #
#                       This is the MMLU dataset.                             #
#         https://huggingface.co/datasets/cais/mmlu                           #
#                                                                             #
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

from datasets import load_dataset
from tasks.common import Task, render_mc

class MMLU(Task):
    """
    The MMLU class is a task that evaluates a model's performance on the MMLU dataset.
    MMLU (Massive Multitask Language Understanding) is a benchmark designed to measure knowledge
    acquired during pretraining by evaluating models exclusively in zero-shot and few-shot settings.
    This makes the benchmark more challenging and more similar to how we evaluate humans.
    The benchmark covers 57 subjects across STEM, the humanities, the social sciences, and more.
    It ranges in difficulty from an elementary level to a professional level,
    and it tests both world knowledge and problem solving abilities.
    Subjects include elementary mathematics, US history, computer science, law, and more.
    """

    # The letters used to label the multiple choice options.
    letters = ('A', 'B', 'C', 'D')
    # A list of all the subject groups in the MMLU dataset.
    groups = ('abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions')

    def __init__(self, subset, split, **kwargs):
        """
        Initializes the MMLU task.
        Args:
            subset (str): The subset of the dataset to use. Must be 'all' or 'auxiliary_train'.
            split (str): The split of the dataset to use. Must be 'train', 'validation', 'dev', or 'test'.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        assert subset in ["all", "auxiliary_train"], f"subset {subset} must be all|auxiliary_train"
        assert split in ["train", "validation", "dev", "test"], f"split {split} must be train|validation|dev|test"
        if subset == "auxiliary_train":
            assert split == "train", "auxiliary_train must be split into train"
        self.subset = subset
        self.split = split
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)
        if subset == "auxiliary_train":
            # The 'auxiliary_train' subset has a nested structure where the actual data is in a 'train' column.
            # This mapping function unnests the data, making it consistent with the other subsets.
            self.ds = self.ds.map(lambda row: row['train'], remove_columns=['train'])

    @property
    def eval_type(self):
        """
        Returns the evaluation type for this task.
        MMLU is a multiple-choice task, so the evaluation is categorical.
        """
        return 'categorical'

    def num_examples(self):
        """
        Returns the total number of examples in the dataset.
        """
        return len(self.ds)

    def get_example(self, index):
        """
        Retrieves a single example from the dataset at the specified index.
        Args:
            index (int): The index of the example to retrieve.
        Returns:
            dict: A dictionary representing the conversation, including messages, subject, and letters for choices.
        """
        row = self.ds[index]
        question = row["question"] # The question text
        choices = row["choices"] # The text of each choice
        answer = row["answer"] # Index of the answer, e.g. 0,1,2,3 (for A,B,C,D)
        subject = row["subject"] # e.g. "college_biology", "college_chemistry", etc.
        assert len(choices) == 4, "MMLU should have 4 choices"
        # create and return the Conversation object
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        conversation = {
            "messages": messages,
            "subject": subject, # useful for grouping metrics by subject later
            "letters": self.letters, # useful during evaluation to constrain the assistant's prediction
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Evaluates the model's response against the correct answer.
        Args:
            conversation (dict): The conversation dictionary containing the context.
            assistant_response (str): The model's response.
        Returns:
            bool: True if the assistant's response is correct, False otherwise.
        """
        # This assert ensures that the model's response is one of the valid choices.
        # This is a safeguard to prevent unexpected evaluation behavior.
        assert assistant_response in self.letters, f"MMLU answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return assistant_response == assistant_message
