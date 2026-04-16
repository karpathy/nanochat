from __future__ import annotations

import asyncio
from pathlib import Path

from config import Settings
from services.weight_manager import WeightManager


def test_weight_registry_downloads_and_tracks_models(tmp_path: Path) -> None:
    downloads: list[tuple[str, str, str | None]] = []

    def fake_downloader(*, repo_id: str, local_dir: str, token: str | None = None) -> str:
        downloads.append((repo_id, local_dir, token))
        model_dir = Path(local_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model_000001.pt").write_bytes(b"weights")
        (model_dir / "meta_000001.json").write_text("{}", encoding="utf-8")
        tokenizer_dir = model_dir / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        (tokenizer_dir / "tokenizer.pkl").write_bytes(b"tokenizer")
        return local_dir

    settings = Settings(
        model_storage_path=tmp_path,
        default_model_tag="samosachaat-d12",
        hf_token="hf-secret",
        startup_load_enabled=False,
    )
    manager = WeightManager(settings, downloader=fake_downloader)

    initial_models = manager.list_models()
    assert initial_models[0]["model_tag"] == "samosachaat-d12"
    assert initial_models[0]["available"] is False
    assert initial_models[0]["loaded"] is False

    asyncio.run(manager.ensure_available("samosachaat-d24"))

    models = {entry["model_tag"]: entry for entry in manager.list_models()}
    assert models["samosachaat-d24"]["available"] is True
    assert models["samosachaat-d24"]["source"] == "huggingface:manmohan659/samosachaat-d24"
    assert downloads == [("manmohan659/samosachaat-d24", str(tmp_path / "samosachaat-d24"), "hf-secret")]
    assert (tmp_path / "tokenizer" / "tokenizer.pkl").exists()
