import pytest

from tasks.gsm8k import DATASET_CONFIGS, GSM8K

# Simple test to check we are getting the correct rows from the gsm8k datasets.
# It does not verify the actual content of the dataset itself.
EXPECTED_COUNTS = {
    ("main", "train"): 7473,
    ("main", "test"): 1319,
    ("socratic", "train"): 7473,
    ("socratic", "test"): 1319,
    ("platinum", "test"): 1209,
}


@pytest.mark.parametrize(
    "subset, split, expected",
    [
        (subset, split, count)
        for (subset, split), count in sorted(EXPECTED_COUNTS.items())
    ],
)
def test_gsm8k_real_dataset_counts(subset, split, expected):
    task = GSM8K(subset=subset, split=split)
    assert task.num_examples() == expected


def test_gsm8k_conversation_structure():
    task = GSM8K(subset="main", split="test")
    conversation = task.get_example(0)
    assert conversation["messages"][0]["role"] == "user"
    assert conversation["messages"][1]["role"] == "assistant"
    assert isinstance(conversation["messages"][0]["content"], str)
    assert isinstance(conversation["messages"][1]["content"], list)


def test_gsm8k_invalid_split_guard():
    for subset, config in DATASET_CONFIGS.items():
        disallowed = {"train", "test"} - config["splits"]
        for split in disallowed:
            with pytest.raises(AssertionError):
                GSM8K(subset=subset, split=split)
