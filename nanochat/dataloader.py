"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

from typing import Iterator, Tuple
import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size, nums_parallel_files=250) -> Iterator[Tuple[list, Tuple[int, int, int]]]:
    """生成器函数，按文档批次加载数据，并支持从中断处恢复。在预训练的训练/测试中使用

    Args:
        split: "train" or "val"
        resume_state_dict: 状态字典，包含 "pq_idx", "rg_idx", "epoch" 用于从中断处恢复
        tokenizer_batch_size: 定义每个批次的文章数量
        nums_parallel_files: 定义整个预训练过程使用的 parquet 文件数量
    
    Yields:
        tuple: (batch, (pq_idx, rg_idx, epoch)) 元组，包含当前批次的文本列表和对应的 parquet 文件索引、row group 索引、epoch
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:min(len(parquet_paths), nums_parallel_files)]
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    # 训练使用除了最后一个文件以外的所有文件，测试使用最后一个文件
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    # 从状态字典中恢复位置
    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            # pf = [
            # [text1, text2, ...],
            # [textN+1, textN+2, ...],
            # ...]
            # 每个 pf 包含多个 row group，每个 row group 包含一批文本数据
            pf = pq.ParquetFile(filepath)
            # 计算每个 ddp 进程应该从哪个 row group 开始
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # 从中断位置的下一个 row group 开始，以避免重复
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups: # 若越界则开始遍历下一个 parquet 文件
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            # 遍历 当前 parquet 文件的 row group，每个 ddp 进程处理不同的 row group
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)  # 读取一个 row group，得到一个表格对象 rg
                batch = rg.column('text').to_pylist()   # 从表格对象中提取文本列，得到一个文本列表 batch
                for i in range(0, len(batch), tokenizer_batch_size):    # 将文本列表分割成更小的批次，每个批次包含 tokenizer_batch_size 个文本
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size    # 每个 ddp 进程跳过已经被其他进程处理的 row group
            pq_idx += 1 # 处理下一个 parquet 文件
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """分布式数据加载器，使用 BOS-aligned best-fit 算法将文本数据打包成固定长度的输入和目标序列，适用于预训练的训练/测试

    Args:
        tokenizer: 用于将文本编码为 token 的 tokenizer 对象
        B: 每个批次的行数(batch size), 用于定义 inputs 和 targets 的批次大小
        T: 每行的 token 数量(序列长度), 用于定义 inputs 和 targets 的序列长度
        split: "train" 或 "val"，指定数据集的划分
        tokenizer_threads: 用于编码文本的线程数量
        tokenizer_batch_size: 每个批次的文本数量，用于编码过程
        device: "cuda" 或 "cpu"，指定数据加载器输出的设备
        resume_state_dict: 状态字典，包含 "pq_idx", "rg_idx", "epoch" 用于从中断处恢复
        buffer_size: 文档缓冲区的大小，定义了在填充行缓冲区之前从生成器中加载多少文档
    
    Returns:
        Iterator[Tuple[torch.Tensor, torch.Tensor, dict]]: 生成器，迭代返回 (inputs, targets, state_dict) 元组，其中 inputs 和 targets 是形状为 (B, T) 的张量，state_dict 包含当前的 "pq_idx", "rg_idx", "epoch" 用于恢复
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # row = [1, 2, 3, 4..., T, T + 1]
    # input = [1, 2, 3, 4..., T]
    # target = [2, 3, 4, 5..., T + 1]
    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        """填充文档缓冲区, 直到达到指定的 buffer_size,
        每次调用都会从 batches 生成器中获取下一个批次的文本数据, 并将其编码为 token 列表后添加到 doc_buffer 中
        """
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        # 一次性编码一个文本列表, 将文章开头设置为 bos_token, 并行使用多个线程加速编码过程
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # 预分配 tokens 的 CPU 和 GPU 缓冲区
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # 填充文档缓冲区直到达到指定大小，以确保有足够的文档可供选择
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                # 计算当前 row 剩余的容量，以确定可以放入多少 tokens
                remaining = row_capacity - pos

                # 循环查找最大且能完全适应剩余空间的文档
                best_idx = -1
                best_len = 0
                # 查找最短的文档，以便在没有完全适合剩余空间的文档时进行裁剪
                shortest_idx = 0
                shortest_len = len(doc_buffer[shortest_idx]) if doc_buffer else float('inf')
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len
                    if doc_len < shortest_len:
                        shortest_idx = i
                        shortest_len = doc_len

                # 如果找到了一个完全适合剩余空间的文档，则将其放入当前行，并从缓冲区中移除
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                # 如果文档缓冲中所有文档长度都大于剩余空间，则选择一个最短的文档进行裁剪，以完全填充剩余空间，并从缓冲区中移除该文档
                else:
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # 将填充好的行缓冲区分割成输入和目标，输入是前 T 个 tokens，目标是后 T 个 tokens
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # 将 CPU 缓冲区的数据复制到 GPU 缓冲区，以便模型训练使用
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
