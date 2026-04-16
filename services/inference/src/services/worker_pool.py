from __future__ import annotations

import asyncio
import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from engine import Engine
from engine.checkpoint_manager import load_model_from_dir
from engine.common import autodetect_device_type, compute_init

logger = logging.getLogger(__name__)


@dataclass
class Worker:
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
    model_tag: str


class WorkerPool:
    def __init__(self, num_workers: int, device_type: str = ""):
        self.requested_workers = max(1, num_workers)
        self.device_type = device_type or autodetect_device_type()
        self.num_workers = 0
        self.model_tag: str | None = None
        self.workers: list[Worker] = []
        self.available_workers: asyncio.Queue[Worker] = asyncio.Queue()
        self.accepting_requests = True
        self._active_requests = 0
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    @classmethod
    async def create(
        cls,
        checkpoints_root: Path,
        model_tag: str,
        step: int | None,
        num_workers: int,
        device_type: str = "",
    ) -> "WorkerPool":
        pool = cls(num_workers=num_workers, device_type=device_type)
        await pool.initialize(checkpoints_root=checkpoints_root, model_tag=model_tag, step=step)
        return pool

    async def initialize(self, checkpoints_root: Path, model_tag: str, step: int | None = None) -> None:
        compute_init(self.device_type)
        self.model_tag = model_tag
        worker_count = self.requested_workers

        if self.device_type != "cuda":
            worker_count = 1
        else:
            available_gpu_count = torch.cuda.device_count()
            if available_gpu_count < 1:
                raise RuntimeError("CUDA was selected but no CUDA devices are available")
            if worker_count > available_gpu_count:
                logger.warning(
                    "Requested %s workers but only %s CUDA device(s) are available; clamping",
                    worker_count,
                    available_gpu_count,
                )
                worker_count = available_gpu_count

        self.num_workers = worker_count
        self.accepting_requests = True

        for worker_index in range(worker_count):
            if self.device_type == "cuda":
                device = torch.device(f"cuda:{worker_index}")
                gpu_id = worker_index
            else:
                device = torch.device(self.device_type)
                gpu_id = 0

            model, tokenizer, _ = load_model_from_dir(
                str(checkpoints_root),
                device,
                phase="eval",
                model_tag=model_tag,
                step=step,
            )
            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=Engine(model, tokenizer),
                tokenizer=tokenizer,
                model_tag=model_tag,
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        logger.info("Initialized %s inference worker(s) for model %s", self.num_workers, model_tag)

    async def acquire_worker(self) -> Worker:
        if not self.accepting_requests:
            raise RuntimeError("Worker pool is draining for a model swap")
        worker = await self.available_workers.get()
        self._active_requests += 1
        self._idle_event.clear()
        return worker

    async def release_worker(self, worker: Worker) -> None:
        self._active_requests = max(0, self._active_requests - 1)
        await self.available_workers.put(worker)
        if self._active_requests == 0:
            self._idle_event.set()

    async def drain(self) -> None:
        self.accepting_requests = False
        if self._active_requests > 0:
            await self._idle_event.wait()

    def resume_accepting_requests(self) -> None:
        self.accepting_requests = True

    def snapshot(self) -> dict[str, object]:
        available = self.available_workers.qsize()
        return {
            "model_tag": self.model_tag,
            "device_type": self.device_type,
            "total_workers": len(self.workers),
            "available_workers": available,
            "busy_workers": len(self.workers) - available,
            "draining": not self.accepting_requests,
            "workers": [
                {
                    "gpu_id": worker.gpu_id,
                    "device": str(worker.device),
                    "model_tag": worker.model_tag,
                }
                for worker in self.workers
            ],
        }

    async def close(self) -> None:
        self.accepting_requests = False
        if self._active_requests > 0:
            await self._idle_event.wait()

        while not self.available_workers.empty():
            self.available_workers.get_nowait()

        self.workers.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
