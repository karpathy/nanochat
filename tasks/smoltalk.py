#--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*#
#_-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
#                                                                           #
#         SmolTalk by HuggingFace. Good "general" conversational dataset.   #
#   https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk             #
#   We use the "smol" version, which is more appropriate for smaller models.#
#                                                                           #
#_-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
#--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*#

from datasets import load_dataset
from tasks.common import Task

class SmolTalk(Task):
    """
    The SmolTalk class handles the smol-smoltalk dataset, a conversational dataset from HuggingFace.
    It's designed for general-purpose conversational models and is particularly suited for smaller models due to its size.
    The training set contains approximately 460,000 examples, while the test set has around 24,000.

    Python equivalent:
    A dictionary where keys are split names ('train', 'test') and values are lists of conversations.
    Each conversation is a list of dictionaries, where each dictionary has 'role' and 'content' keys.
    Example:
    {
        "train": [
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"}
            ],
            # ... more conversations
        ],
        "test": [
            # ... test conversations
        ]
    }
    """

    def __init__(self, split, **kwargs):
        """
        Initializes the SmolTalk task.
        Args:
            split (str): The dataset split to load, must be either "train" or "test".
            **kwargs: Additional keyword arguments passed to the parent Task class.
        """
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        # Load the specified split of the dataset and shuffle it for randomness.
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        """
        Returns the total number of examples in the loaded dataset split.
        """
        return self.length

    def get_example(self, index):
        """
        Retrieves a single conversational example from the dataset.
        Args:
            index (int): The index of the example to retrieve.
        Returns:
            dict: A dictionary containing the conversation messages.
        """
        row = self.ds[index]
        messages = row["messages"]
        # ---------------------------------------------------------------------
        # Perform sanity checks to ensure the data format is as expected.
        # These asserts can be removed later for performance, but are useful for debugging.

        # A conversation can optionally start with a system message.
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:] # The rest of the conversation after the system message.
        else:
            rest_messages = messages

        # There should be at least one user-assistant exchange.
        assert len(rest_messages) >= 2, "SmolTalk messages must have at least 2 messages"

        # Check that roles alternate correctly (user, assistant, user, ...).
        for i, message in enumerate(rest_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"
        # ---------------------------------------------------------------------

        # Return the conversation in the standard format.
        conversation = {
            "messages": messages,
        }
        return conversation
