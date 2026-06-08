import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    timeout_seconds = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SECONDS", "600"))

    print(
        f"rank={rank} local_rank={local_rank} world_size={world_size} "
        f"device={torch.cuda.get_device_name(device)}",
        flush=True,
    )

    dist.init_process_group(
        backend="nccl",
        device_id=device,
        timeout=timedelta(seconds=timeout_seconds),
    )

    dist.barrier()
    print(f"rank={rank} passed initial barrier", flush=True)

    tensor = torch.tensor([rank + 1.0], device=device)
    for i in range(5):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        print(f"rank={rank} all_reduce={i} value={tensor.item()}", flush=True)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print("NCCL smoke completed", flush=True)


if __name__ == "__main__":
    main()
