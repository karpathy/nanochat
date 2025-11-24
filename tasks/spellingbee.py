#--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*#
#_-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
#                                                                           #
#                       The Spelling Bee Task                               #
#                                                                           #
#_-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
#--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*#
"""
This module defines tasks intended to improve a model's spelling and counting abilities.

For example, a task might be: "How many 'r's are in strawberry?" -> 3

A key feature of this task is that the assistant is guided to solve the problem
by combining manual counting with Python code verification. This promotes a robust
problem-solving process in the model. Future versions could introduce small errors
to train the model on error detection and recovery.

This file contains two main tasks:
1. SpellingBee: Counts the occurrences of a specific letter in a word.
2. SimpleSpelling: A simpler task focused on correctly spelling words.

The primary goal is (1), but (2) is included to address a fundamental challenge for LLMs:
mapping tokens (semantic units) to the individual characters that form a word.
While larger models often learn this implicitly, smaller models benefit from explicit
training on this skill.

To preview examples from these tasks, run this script directly:
`python -m tasks.spellingbee`
"""

import re
import random
from tasks.common import Task
from nanochat.common import download_file_with_lock

# Define the alphabet for random letter selection.
LETTERS = "abcdefghijklmnopqrstuvwxyz"
# URL for a comprehensive list of English words.
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"

# Regex to find the final numerical answer, same as in the gsm8k task.
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    Extracts the numerical answer from a string, which is marked with "####".
    This function is designed to parse the final answer from the model's output.
    It handles integers, floats, and commas.

    For example:
    - "The answer is #### 42." -> "42"
    - "After calculation, we get #### 3,141.59" -> "3141.59"
    """
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None

# A diverse set of templates for user messages to augment the training data.
# This helps the model generalize to different ways a user might ask the same question.
# Includes templates in multiple languages for broader applicability.
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # Chinese (Simplified)
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]

class SpellingBee(Task):
    """
    A task to count the occurrences of a letter in a word.
    The assistant's response is structured to first perform a manual count,
    then verify the result using a Python tool call. This encourages a
    "show your work" and "double-check" approach.
    """

    def __init__(self, size=1000, split="train", **kwargs):
        """
        Initializes the SpellingBee task.
        Args:
            size (int): The number of examples to generate for this task.
            split (str): The dataset split, either "train" or "test".
            **kwargs: Additional arguments for the parent Task class.
        """
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.split = split
        # Download the word list if it's not already cached.
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        self.words = words

    @property
    def eval_type(self):
        """ This task requires a generative evaluation, as the response format is complex. """
        return 'generative'

    def num_examples(self):
        """ Returns the number of examples in this task. """
        return self.size

    def get_example(self, index):
        """
        Generates a single example for the SpellingBee task.
        Args:
            index (int): An index to seed the random number generator for reproducibility.
        Returns:
            dict: A conversation dictionary representing the task example.
        """
        # Use the index to seed the random generator for deterministic example generation.
        seed = index if self.split == "train" else -(index + 1)
        rng = random.Random(seed)

        # Select a random word and a letter to count.
        word = rng.choice(self.words)
        # Usually pick a letter from the word, but sometimes a random one.
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)

        # Calculate the correct answer.
        count = word.count(letter)

        # Create a user message using a random template for variety.
        template = rng.choice(USER_MSG_TEMPLATES)
        if rng.random() < 0.3:
            template = template.lower()
        quote_options = ['', "'", '"']
        letter_quote = rng.choice(quote_options)
        word_quote = rng.choice(quote_options)
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)
        if rng.random() < 0.5:
            user_msg += "?"

        # Construct the ideal assistant response as a series of parts.
        assistant_parts = []
        word_letters = ",".join(list(word))
        # Part 1: Manual counting process.
        manual_text = f"""We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.

First spell the word out:
{word}:{word_letters}

Then count the occurrences of '{letter}':
"""
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"

        manual_text += f"\nThis gives us {running_count}."
        assistant_parts.append({"type": "text", "text": manual_text})
        # Part 2: Transition to Python verification.
        assistant_parts.append({"type": "text", "text": "\n\nLet me double check this using Python:\n\n"})
        # Part 3: The Python tool call itself.
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({"type": "python", "text": python_expr})
        # Part 4: The output from the Python tool.
        assistant_parts.append({"type": "python_output", "text": str(count)})
        # Part 5: The final conclusion.
        assistant_parts.append({"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"})

        # Assemble the full conversation.
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Evaluates the assistant's response to determine if it's correct.
        This is similar to the evaluation in the gsm8k task.
        Args:
            conversation (dict): The original conversation.
            assistant_response (str): The generated response from the assistant.
        Returns:
            int: 1 if the answer is correct, 0 otherwise.
        """
        assert isinstance(assistant_response, str), "Assuming a simple string response for now"
        # Extract the ground truth answer from the original conversation.
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "The last message should be from the assistant"
        assert isinstance(assistant_message['content'], list), "Content is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text']

        # Extract the reference number and the predicted number.
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)

        # Compare and return the result.
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Provides a simple binary reward (0 or 1) based on the evaluation result.
        This is used during reinforcement learning.
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float


class SimpleSpelling(Task):
    """
    A simpler task designed to train the model on basic spelling.
    This helps smaller models learn the correspondence between tokens and characters.
    """

    def __init__(self, size=1000, split="train", **kwargs):
        """
        Initializes the SimpleSpelling task.
        Args:
            size (int): The number of examples to generate.
            split (str): The dataset split, "train" or "test".
            **kwargs: Additional arguments for the parent Task class.
        """
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        rng = random.Random(42)
        rng.shuffle(words)  # Use a different word order than SpellingBee for variety.
        self.words = words

    @property
    def eval_type(self):
        """ This task uses generative evaluation. """
        return 'generative'

    def num_examples(self):
        """ Returns the number of examples in this task. """
        return self.size

    def get_example(self, index):
        """
        Generates a single example for the SimpleSpelling task.
        Args:
            index (int): An index for seeding the random number generator.
        Returns:
            dict: A conversation dictionary for the task.
        """
        seed = index if self.split == "train" else -(index + 1)
        rng = random.Random(seed)
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))

        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation


if __name__ == "__main__":
    # This block allows for previewing the generated examples from the tasks.

    # Preview the SpellingBee task.
    print("--- SpellingBee Task Preview ---")
    task = SpellingBee()
    for i in range(10):
        ex = task.get_example(i)
        print("=" * 100)
        print(f"User: {ex['messages'][0]['content']}")
        print("-" * 100)
        print("Assistant:")
        assistant_parts = ex['messages'][1]['content']
        for part in assistant_parts:
            if part['type'] == 'text':
                print(part['text'], end='')
            elif part['type'] == 'python':
                print(f"<<Py: {part['text']} -> ", end='')
            elif part['type'] == 'python_output':
                print(f"Out: {part['text']}>>", end='')
        print("\n" + "-" * 100)

    # # To preview the SimpleSpelling task, uncomment the following lines.
    # print("\n\n--- SimpleSpelling Task Preview ---")
    # task = SimpleSpelling()
    # for i in range(10):
    #     ex = task.get_example(i)
    #     print("=" * 100)
    #     print(f"User: {ex['messages'][0]['content']}")
    #     print("-" * 100)
    #     print(f"Assistant: {ex['messages'][1]['content']}")

    # # To scrutinize the tokenization of the last example, uncomment these lines.
    # from nanochat.tokenizer import get_tokenizer
    # tokenizer = get_tokenizer()
    # ids, mask = tokenizer.render_conversation(ex)
    # print("\n--- Tokenization of Last Example ---")
    # print(tokenizer.visualize_tokenization(ids, mask, with_token_id=True))
