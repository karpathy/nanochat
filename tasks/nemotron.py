"""
Nemotron Post-Training Dataset v2 by NVIDIA.
https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2

This dataset includes:
- math: 239,467 samples
- code: 175,000 samples  
- stem: 355,000 samples
- chat: 627,720 samples
- multilingual: ~1M samples per language (ja, de, it, es, fr)

Total: ~6.3M samples across all categories
"""

from datasets import load_dataset
from tasks.common import Task

class Nemotron(Task):
    """
    Nemotron Post-Training Dataset v2.
    
    Args:
        categories: List of categories to load. Can include:
                   ["math", "code", "stem", "chat", 
                    "multilingual_ja", "multilingual_de", "multilingual_it", 
                    "multilingual_es", "multilingual_fr"]
                   Default: ["math", "code", "stem", "chat"] (English only)
        split: Dataset split, typically "train" (default: "train")
        **kwargs: Additional arguments for Task parent class (start, stop, step)
    
    Example:
        # Load only math and code
        ds = Nemotron(categories=["math", "code"], split="train")
        
        # Load everything including multilingual
        ds = Nemotron(categories=["math", "code", "stem", "chat", 
                                  "multilingual_ja", "multilingual_de"], 
                      split="train")
    """
    
    # Available categories and their sizes
    CATEGORY_SIZES = {
        "math": 239467,
        "code": 175000,
        "stem": 355000,
        "chat": 627720,
        "multilingual_ja": 975202,
        "multilingual_de": 1015314,
        "multilingual_it": 1016503,
        "multilingual_es": 935704,
        "multilingual_fr": 1001504,
    }
    
    def __init__(self, categories=None, split="train", **kwargs):
        super().__init__(**kwargs)
        
        # Default to English-only categories
        if categories is None:
            categories = ["math", "code", "stem", "chat"]
        
        # Validate categories
        for cat in categories:
            assert cat in self.CATEGORY_SIZES, \
                f"Invalid category '{cat}'. Must be one of {list(self.CATEGORY_SIZES.keys())}"
        
        self.categories = categories
        self.split = split
        
        # Load the dataset
        # Note: The dataset uses "SFT" as the config name
        print(f"Loading Nemotron dataset with categories: {categories}")
        
        # Load all requested categories
        self.datasets = []
        self.category_offsets = [0]  # cumulative offsets for indexing
        
        for category in categories:
            try:
                # Load each category separately
                ds = load_dataset(
                    "nvidia/Nemotron-Post-Training-Dataset-v2",
                    split=category,
                    trust_remote_code=True
                )
                self.datasets.append(ds)
                self.category_offsets.append(self.category_offsets[-1] + len(ds))
                print(f"  ✓ Loaded {category}: {len(ds):,} samples")
            except Exception as e:
                print(f"  ✗ Failed to load {category}: {e}")
                raise
        
        self.length = self.category_offsets[-1]
        print(f"Total Nemotron samples: {self.length:,}")
    
    def num_examples(self):
        return self.length
    
    def get_example(self, index):
        """
        Convert Nemotron dataset format to conversation format.
        
        Nemotron format has columns like:
        - 'prompt': the user query
        - 'response': the assistant response
        - possibly 'system': system message
        """
        # Find which category this index belongs to
        category_idx = 0
        for i, offset in enumerate(self.category_offsets[1:]):
            if index < offset:
                category_idx = i
                break
        
        # Get the local index within the category
        local_index = index - self.category_offsets[category_idx]
        row = self.datasets[category_idx][local_index]
        
        # Convert to conversation format
        # The Nemotron dataset typically has 'prompt' and 'response' fields
        messages = []
        
        # Add system message if present
        if 'system' in row and row['system']:
            messages.append({
                "role": "system",
                "content": row['system']
            })
        
        # Add user prompt
        if 'prompt' in row:
            messages.append({
                "role": "user", 
                "content": row['prompt']
            })
        
        # Add assistant response
        if 'response' in row:
            messages.append({
                "role": "assistant",
                "content": row['response']
            })
        
        # Fallback: if the dataset uses a different schema, try 'messages' directly
        if not messages and 'messages' in row:
            messages = row['messages']
        
        # Ensure we have at least a user and assistant message
        assert len(messages) >= 2, \
            f"Conversation must have at least 2 messages, got {len(messages)}"
        
        conversation = {
            "messages": messages,
        }
        return conversation


if __name__ == "__main__":
    # Test the Nemotron dataset
    print("Testing Nemotron dataset loader...")
    
    # Test 1: Load a small subset
    print("\n=== Test 1: Load math and code categories ===")
    ds = Nemotron(categories=["math", "code"], split="train")
    print(f"Dataset length: {len(ds):,}")
    
    # Get a sample
    print("\n=== Test 2: Get a sample conversation ===")
    sample = ds[0]
    print("Sample conversation:")
    for msg in sample["messages"]:
        print(f"  {msg['role']}: {msg['content'][:100]}...")
    
    print("\n✓ Nemotron dataset loader test completed!")

