"""
The conversation datasets, downloaded from the HuggingFace hub once into
$NANOCHAT_BASE_DIR/task_data/. Two consumers: scripts/chat_prepare.py packs SmolTalk plus
MMLU's auxiliary_train (the multiple-choice format primer) into the SFT shards, and
scripts/chat_eval.py scores the ARC and MMLU test splits.

A conversation is {"messages": [{"role": ..., "content": ...}, ...]}. For multiple
choice it also carries "letters", the answer options, and the assistant's message is
the correct letter.
"""

import os
import json
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from nanochat.common import get_base_dir


class HubDataset:
    """
    Minimal stand-in for a HuggingFace datasets Dataset: wraps a pyarrow
    Table and offers lazy row access and a seeded shuffle.
    """

    def __init__(self, table, permutation=None):
        self.table = table
        self.permutation = permutation

    def __len__(self):
        return self.table.num_rows

    def shuffle(self, seed):
        # matches datasets.Dataset.shuffle(seed=seed) exactly, row order comes out identical
        permutation = np.random.default_rng(seed).permutation(len(self))
        return HubDataset(self.table, permutation)

    def __getitem__(self, index):
        physical_index = index if self.permutation is None else int(self.permutation[index])
        row = {column: self.table[column][physical_index].as_py() for column in self.table.column_names}
        return row


def load_hub_dataset(repo_id, subset="default", split="train"):
    """
    Minimal stand-in for HuggingFace datasets.load_dataset(repo_id, subset, split=split).
    Every dataset on the hub has an auto-generated parquet export. We list the parquet
    shards via the hub API, download them (once) into the local cache directory, and
    read them with pyarrow. Under torchrun, only one rank downloads, the others wait.
    """
    base_dir = get_base_dir()
    slug = repo_id.replace("/", "--")
    shards_dir = os.path.join(base_dir, "task_data", slug, subset, split)
    # the manifest is written last, so its existence means the download completed
    manifest_path = os.path.join(shards_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        os.makedirs(shards_dir, exist_ok=True)
        with FileLock(manifest_path + ".lock"):
            # only a single rank acquires the lock and downloads, the others block
            # here and then skip the download because they recheck the manifest
            if not os.path.exists(manifest_path):
                listing_url = f"https://huggingface.co/api/datasets/{repo_id}/parquet/{subset}/{split}"
                with urllib.request.urlopen(listing_url) as response:
                    shard_urls = json.loads(response.read())
                filenames = []
                for shard_index, shard_url in enumerate(shard_urls):
                    filename = f"{shard_index:05d}.parquet"
                    print(f"Downloading {shard_url} ...")
                    with urllib.request.urlopen(shard_url) as response:
                        content = response.read()
                    with open(os.path.join(shards_dir, filename), "wb") as f:
                        f.write(content)
                    filenames.append(filename)
                with open(manifest_path, "w") as f:
                    json.dump(filenames, f)
    with open(manifest_path, "r") as f:
        filenames = json.load(f)
    shard_paths = [os.path.join(shards_dir, filename) for filename in filenames]
    tables = [pq.read_table(path) for path in shard_paths]
    table = pa.concat_tables(tables)
    return HubDataset(table)


def render_mc(question, letters, choices):
    """
    The multiple choice prompt. Two details that matter for small models: the letter
    comes *after* its choice (better binding), and there is no whitespace before it,
    so the prompt's "A" is the same token as the assistant's bare "A" reply.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


class SmolTalk:
    """smol-smoltalk (HuggingFaceTB), the version of SmolTalk for small models.
    General conversations: an optional system message, then user/assistant turns.
    train is 460K rows, test is 24K."""

    def __init__(self, split):
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        self.ds = load_hub_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        row = self.ds[index]
        conversation = {"messages": row["messages"]}
        return conversation


class MMLU:
    """cais/mmlu, 4-way multiple choice over 57 subjects. test is the eval;
    auxiliary_train (the train splits of ARC, OBQA, RACE and MC-TEST) is the SFT
    format primer."""

    letters = ("A", "B", "C", "D")

    def __init__(self, split):
        assert split in ["auxiliary_train", "test"], "MMLU split must be auxiliary_train|test"
        self.ds = load_hub_dataset("cais/mmlu", "all", split=split).shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        row = self.ds[index]
        assert len(row["choices"]) == 4, "MMLU should have 4 choices"
        user_message = render_mc(row["question"], self.letters, row["choices"])
        assistant_message = self.letters[row["answer"]] # answer is the index 0..3
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        conversation = {"messages": messages, "letters": self.letters}
        return conversation


class ARC:
    """allenai/ai2_arc, grade-school science multiple choice in two difficulties.
    The letters vary per question (mostly A-D, sometimes 1-4 or A-E)."""

    def __init__(self, subset, split):
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        self.ds = load_hub_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        row = self.ds[index]
        letters = row["choices"]["label"]
        answer = row["answerKey"]
        assert answer in letters, f"ARC answer {answer} must be one of {letters}"
        user_message = render_mc(row["question"], letters, row["choices"]["text"])
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]
        conversation = {"messages": messages, "letters": letters}
        return conversation
