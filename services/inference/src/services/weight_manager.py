from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import snapshot_download

from config import Settings


@dataclass
class ModelRecord:
    model_tag: str
    path: str
    source: str
    available: bool = False
    loaded: bool = False


class WeightManager:
    def __init__(
        self,
        settings: Settings,
        downloader: Callable[..., Any] = snapshot_download,
    ) -> None:
        self.settings = settings
        self.storage_path = Path(settings.model_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = self.storage_path / "tokenizer"
        self.tokenizer_path.mkdir(parents=True, exist_ok=True)
        os.environ["NANOCHAT_BASE_DIR"] = str(self.storage_path)

        self._downloader = downloader
        self._registry: dict[str, ModelRecord] = {}
        self.current_model: str | None = None

        if settings.default_model_tag:
            self.register_model(settings.default_model_tag)
        self.refresh_registry()

    def _default_source(self, model_tag: str) -> str:
        return f"huggingface:{self.settings.hf_repo_owner}/{model_tag}"

    def _model_dir(self, model_tag: str) -> Path:
        return self.storage_path / model_tag

    def _has_checkpoint_files(self, model_dir: Path) -> bool:
        return any(model_dir.glob("model_*.pt"))

    def _extract_repo_id(self, source: str) -> str | None:
        if source.startswith("huggingface:"):
            return source.split(":", 1)[1]
        return None

    def _refresh_entry(self, model_tag: str) -> None:
        record = self._registry[model_tag]
        record.available = self._has_checkpoint_files(Path(record.path))
        record.loaded = model_tag == self.current_model

    def register_model(self, model_tag: str, source: str | None = None, path: str | Path | None = None) -> ModelRecord:
        existing = self._registry.get(model_tag)
        record = ModelRecord(
            model_tag=model_tag,
            path=str(path or self._model_dir(model_tag)),
            source=source or (existing.source if existing else self._default_source(model_tag)),
            available=False,
            loaded=False,
        )
        self._registry[model_tag] = record
        self._refresh_entry(model_tag)
        return record

    def refresh_registry(self) -> None:
        if not self.storage_path.exists():
            return
        for child in sorted(self.storage_path.iterdir()):
            if child.is_dir() and child.name != "tokenizer":
                self.register_model(child.name, path=child)
        for model_tag in list(self._registry):
            self._refresh_entry(model_tag)

    def set_loaded_model(self, model_tag: str | None) -> None:
        self.current_model = model_tag
        for tag in list(self._registry):
            self._refresh_entry(tag)

    def list_models(self) -> list[dict[str, Any]]:
        self.refresh_registry()
        return [asdict(self._registry[tag]) for tag in sorted(self._registry)]

    def _sync_tokenizer_assets(self, model_dir: Path) -> None:
        for base_dir in (model_dir / "tokenizer", model_dir):
            if not base_dir.exists():
                continue
            for asset_name in ("tokenizer.pkl", "tokenizer.json", "token_bytes.pt"):
                source = base_dir / asset_name
                if source.exists():
                    shutil.copy2(source, self.tokenizer_path / asset_name)

    def _download_repo(self, repo_id: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        self._downloader(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=self.settings.hf_token,
        )

    async def ensure_available(self, model_tag: str) -> ModelRecord:
        record = self.register_model(model_tag)
        if record.available:
            self._sync_tokenizer_assets(Path(record.path))
            return record

        repo_id = self._extract_repo_id(record.source)
        if repo_id is None:
            raise FileNotFoundError(f"Model {model_tag} is not available locally and has no HuggingFace source")

        await asyncio.to_thread(self._download_repo, repo_id, Path(record.path))
        self._sync_tokenizer_assets(Path(record.path))
        self._refresh_entry(model_tag)

        if not self._registry[model_tag].available:
            raise FileNotFoundError(f"Downloaded model {model_tag} but no model_*.pt files were found")

        return self._registry[model_tag]

    async def build_worker_pool(self, model_tag: str, step: int | None = None):
        from services.worker_pool import WorkerPool

        await self.ensure_available(model_tag)
        pool = await WorkerPool.create(
            checkpoints_root=self.storage_path,
            model_tag=model_tag,
            step=step,
            num_workers=self.settings.num_workers,
            device_type=self.settings.resolved_device_type,
        )
        self.set_loaded_model(model_tag)
        return pool
