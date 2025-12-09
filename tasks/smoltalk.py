"""
SmolTalk by HuggingFace. Good "general" conversational dataset.
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
We use the "smol" version, which is more appropriate for smaller models.
Supports loading from HuggingFace Hub or local parquet files.
"""

from datasets import load_dataset
import os
import random
import pyarrow.parquet as pq
from tasks.common import Task

class SmolTalk(Task):
    """ smol-smoltalk dataset. train is 460K rows, test is 24K rows. """

    def __init__(self, split, parquet_dir=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        
        # If parquet_dir is provided, load from local parquet files
        if parquet_dir is not None:
            self._load_from_parquet(parquet_dir, split)
        else:
            # Load from HuggingFace Hub (original behavior)
            try:
                self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
                self.length = len(self.ds)
                self.use_parquet = False
            except Exception as e:
                raise ConnectionError(
                    f"Failed to load SmolTalk dataset from HuggingFace Hub: {e}\n"
                    f"Please provide parquet_dir parameter to load from local parquet files."
                ) from e

    def _load_from_parquet(self, parquet_dir, split):
        """Load data from local parquet files."""
        if not os.path.isdir(parquet_dir):
            raise ValueError(f"Parquet directory does not exist: {parquet_dir}")
        
        # Find parquet files matching the split
        if split == "train":
            parquet_files = sorted([
                os.path.join(parquet_dir, f) 
                for f in os.listdir(parquet_dir) 
                if f.endswith('.parquet') and 'train' in f
            ])
        else:  # test
            parquet_files = sorted([
                os.path.join(parquet_dir, f) 
                for f in os.listdir(parquet_dir) 
                if f.endswith('.parquet') and 'test' in f
            ])
        
        if not parquet_files:
            raise ValueError(f"No {split} parquet files found in {parquet_dir}")
        
        # Load all conversations from parquet files
        self.conversations = []
        for filepath in parquet_files:
            try:
                pf = pq.ParquetFile(filepath)
                # Read all row groups
                for rg_idx in range(pf.num_row_groups):
                    rg = pf.read_row_group(rg_idx)
                    # Check if 'messages' column exists
                    if 'messages' in rg.column_names:
                        messages_list = rg.column('messages').to_pylist()
                        for messages in messages_list:
                            # Validate message structure
                            if isinstance(messages, list) and len(messages) >= 1:
                                self.conversations.append(messages)
            except Exception as e:
                print(f"Warning: Failed to read {filepath}: {e}")
                continue
        
        if not self.conversations:
            raise ValueError(f"No conversations loaded from parquet files in {parquet_dir}")
        
        # Shuffle with fixed seed for reproducibility
        random.Random(42).shuffle(self.conversations)
        
        self.length = len(self.conversations)
        self.use_parquet = True
        print(f"Loaded {self.length} conversations from {len(parquet_files)} parquet files (split={split})")

    def num_examples(self):
        return self.length

    def get_example(self, index):
        if self.use_parquet:
            messages = self.conversations[index]
        else:
            row = self.ds[index]
            messages = row["messages"]
        
        # ---------------------------------------------------------------------
        # sanity checking asserts here
        # TODO: we could remove these asserts later, for now just don't want any footguns
        # there is an optional system message at the beginning
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message.get("role") == "system":
            rest_messages = messages[1:] # optional system message is OK
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, f"SmolTalk messages must have at least 2 messages, got {len(rest_messages)}"
        for i, message in enumerate(rest_messages):
            # user and assistant alternate as user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message.get("role") == expected_role, f"Message {i} has role {message.get('role')} but should be {expected_role}"
            assert isinstance(message.get("content"), str), "Content must be a string"
        # ---------------------------------------------------------------------
        # create and return the Conversation object (ok to emit the system message too)
        conversation = {
            "messages": messages,
        }
        return conversation
