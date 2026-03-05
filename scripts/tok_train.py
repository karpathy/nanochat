"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import time
from typing import Iterator, Tuple
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=2**16, help='Vocabulary size (default: 32768 = 2^15)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator() -> Iterator[str]:
    """文档文本迭代器
    Args:
        None
    
    Yields:
        str: 文档文本
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            # 若文档长度超过配置上限，则截断
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            # 如果已经处理的字符数超过配置上限，则停止迭代
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
start = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
end = time.time()
train_time = end - start
print(f"Training time: {train_time:.2f}s")

def train(iterator: Iterator[str], vocab_size: int) -> Tuple[RustBPETokenizer, float]:
    """训练BPE分词器
    Args:
        iterator (Iterator[str]): 文本迭代器
        vocab_size (int): 词表大小
    
    Returns:
        Tuple[RustBPETokenizer, float]: 训练好的分词器和训练时间（秒）
    """
    start = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(iterator, vocab_size)
    end = time.time()
    train_time = end - start
    return tokenizer, train_time
# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

def sanity_check(tokenizer: RustBPETokenizer):
    """对分词器进行快速的内联检查，确保编码和解码的一致性
    Args:
        tokenizer (RustBPETokenizer): 需要检查的分词器
    
    Raises:
        AssertionError: 如果编码和解码不一致，则抛出断言错误
    """
    test_text = """Hello world! This is a test.
    Numbers: 123, 4567, 89
    Contractions: I'm, you're, it's
    Special chars: @#$%^&*()
    Unicode: 你好世界 🌍"""
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

def generate_token_bytes(tokenizer: RustBPETokenizer, save_path: str) -> torch.Tensor:
    """生成一个张量，表示每个token id对应的字节数，并保存到磁盘
    Args:
        tokenizer (RustBPETokenizer): 已训练好的分词器
        save_path (str): 保存token_bytes张量的路径
    
    Returns:
        torch.Tensor: 包含每个token id对应的字节数的张量
    """
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
    token_bytes = []
    for token_id in range(vocab_size):
        token_str = token_strings[token_id] # the Python string representation of this token
        if token_str in special_set:
            token_bytes.append(0) # special characters are not counted
        else:
            id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
            token_bytes.append(id_bytes)
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
    with open(save_path, "wb") as f:
        torch.save(token_bytes, f)
    print(f"Saved token_bytes to {save_path}")
    return token_bytes

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])

def log_tokenizer_training(args: argparse.Namespace, train_time: float, tokenizer: RustBPETokenizer):
    """记录分词器训练的相关信息到报告中
    Args:
        args (argparse.Namespace): 命令行参数
        train_time (float): 训练时间（秒）
        tokenizer (RustBPETokenizer): 已训练好的分词器
    """
    # 计算token_bytes统计信息
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
    token_bytes = []
    for token_id in range(vocab_size):
        token_str = token_strings[token_id] # the Python string representation of this token
        if token_str in special_set:
            token_bytes.append(0) # special characters are not counted
        else:
            id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
            token_bytes.append(id_bytes)
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
    token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
    
    # Log to report
    get_report().log(section="Tokenizer training", data=[
        vars(args), # argparse command line arguments
        {"train_time": train_time},
        {"num_special_tokens": len(special_set)},
        {
            "token_bytes_min": int(token_bytes_nonzero.min().item()),
            "token_bytes_max": int(token_bytes_nonzero.max().item()),
            "token_bytes_mean": token_bytes_nonzero.mean().item(),
            "token_bytes_std": token_bytes_nonzero.std().item(),
        }
    ])