from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from config import Settings
from main import create_app


class FakeRuntime:
    def __init__(self, ready: bool) -> None:
        self.ready = ready

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def health_payload(self) -> dict[str, object]:
        return {
            "status": "ok",
            "ready": self.ready,
            "current_model": "samosachaat-d12" if self.ready else None,
            "total_workers": 1 if self.ready else 0,
            "available_workers": 1 if self.ready else 0,
            "busy_workers": 0,
            "draining": False,
            "workers": [],
        }

    def models_payload(self) -> dict[str, object]:
        return {"current_model": None, "models": []}

    def stats_payload(self) -> dict[str, object]:
        return self.health_payload()

    def require_ready_pool(self):
        raise AssertionError("validation should fail before a worker is requested")

    async def swap_model(self, model_tag: str) -> dict[str, str]:
        await asyncio.sleep(0)
        return {"status": "ok", "current_model": model_tag}


def test_health_endpoint_reports_readiness() -> None:
    settings = Settings(internal_api_key="secret", startup_load_enabled=False)

    not_ready_app = create_app(settings=settings, runtime=FakeRuntime(ready=False))
    with TestClient(not_ready_app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ready"] is False

    ready_app = create_app(settings=settings, runtime=FakeRuntime(ready=True))
    with TestClient(ready_app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["current_model"] == "samosachaat-d12"


def test_generate_validation_rejects_empty_messages() -> None:
    settings = Settings(internal_api_key="secret", startup_load_enabled=False)
    app = create_app(settings=settings, runtime=FakeRuntime(ready=True))

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            headers={"X-Internal-API-Key": "secret"},
            json={"messages": []},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "At least one message is required"


def test_generate_validation_rejects_invalid_role() -> None:
    settings = Settings(internal_api_key="secret", startup_load_enabled=False)
    app = create_app(settings=settings, runtime=FakeRuntime(ready=True))

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            headers={"X-Internal-API-Key": "secret"},
            json={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                ]
            },
        )

    assert response.status_code == 400
    assert "invalid role" in response.json()["detail"]
