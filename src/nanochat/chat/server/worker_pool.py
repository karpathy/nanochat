"""Worker and WorkerPool: per-GPU model replicas for parallel inference."""

import asyncio
from dataclasses import dataclass
from typing import List, Optional

import torch

from nanochat.evaluation.engine import Engine
from nanochat.training.checkpoint import load_model


@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""

    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object


class WorkerPool:
    """Pool of workers, each with a model replica on a different GPU."""

    def __init__(self, device_type: str, base_dir: str, num_gpus: Optional[int] = None):
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if device_type == "cuda" else 1
        self.device_type = device_type
        self.base_dir = base_dir
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue[Worker] = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None) -> None:
        """Load model on each GPU."""
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert self.device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."

        for gpu_id in range(self.num_gpus):
            if self.device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(self.device_type)
                print(f"Loading model on {self.device_type}...")

            model, tokenizer, _ = load_model(self.base_dir, source, device, model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            worker = Worker(gpu_id=gpu_id, device=device, engine=engine, tokenizer=tokenizer)
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """Get an available worker from the pool."""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker) -> None:
        """Return a worker to the pool."""
        await self.available_workers.put(worker)
