"""Task definitions for evaluation and training."""

from nanochat.tasks.base import Task, TaskMixture, TaskSequence, render_mc
from nanochat.tasks.mmlu import MMLU
from nanochat.tasks.arc import ARC
from nanochat.tasks.gsm8k import GSM8K
from nanochat.tasks.humaneval import HumanEval
from nanochat.tasks.smoltalk import SmolTalk
from nanochat.tasks.spellingbee import SpellingBee
from nanochat.tasks.customjson import CustomJSON

__all__ = [
    "Task",
    "TaskMixture",
    "TaskSequence",
    "render_mc",
    "MMLU",
    "ARC",
    "GSM8K",
    "HumanEval",
    "SmolTalk",
    "SpellingBee",
    "CustomJSON",
]
