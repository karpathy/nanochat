import pytest


@pytest.mark.asyncio
async def test_health_is_unauthenticated(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["service"] == "chat-api"
