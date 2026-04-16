import pytest
import respx

from .conftest import stub_auth_validate


@pytest.mark.asyncio
@respx.mock
async def test_requires_authorization_header(client):
    response = await client.get("/api/conversations")
    assert response.status_code == 401


@pytest.mark.asyncio
@respx.mock
async def test_create_and_list_conversation(client, seeded_user):
    stub_auth_validate(respx.mock, seeded_user)
    headers = {"Authorization": "Bearer valid-token"}

    create = await client.post(
        "/api/conversations", json={"title": "my first chat"}, headers=headers
    )
    assert create.status_code == 201
    convo = create.json()
    assert convo["title"] == "my first chat"
    assert convo["user_id"] == seeded_user["id"]

    listed = await client.get("/api/conversations", headers=headers)
    assert listed.status_code == 200
    payload = listed.json()
    assert len(payload["items"]) == 1
    assert payload["items"][0]["id"] == convo["id"]
    assert payload["grouped"]  # at least one date bucket


@pytest.mark.asyncio
@respx.mock
async def test_update_conversation_title(client, seeded_user):
    stub_auth_validate(respx.mock, seeded_user)
    headers = {"Authorization": "Bearer valid-token"}

    create = await client.post("/api/conversations", json={}, headers=headers)
    convo_id = create.json()["id"]

    updated = await client.put(
        f"/api/conversations/{convo_id}",
        json={"title": "Renamed"},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["title"] == "Renamed"


@pytest.mark.asyncio
@respx.mock
async def test_delete_conversation(client, seeded_user):
    stub_auth_validate(respx.mock, seeded_user)
    headers = {"Authorization": "Bearer valid-token"}

    create = await client.post("/api/conversations", json={}, headers=headers)
    convo_id = create.json()["id"]

    deleted = await client.delete(f"/api/conversations/{convo_id}", headers=headers)
    assert deleted.status_code == 204

    missing = await client.get(f"/api/conversations/{convo_id}", headers=headers)
    assert missing.status_code == 404


@pytest.mark.asyncio
@respx.mock
async def test_user_scoping_prevents_cross_user_access(
    client, seeded_user, other_user
):
    stub_auth_validate(respx.mock, seeded_user, token="alice-token")
    stub_auth_validate(respx.mock, other_user, token="bob-token")

    alice_headers = {"Authorization": "Bearer alice-token"}
    bob_headers = {"Authorization": "Bearer bob-token"}

    alice_convo = await client.post(
        "/api/conversations",
        json={"title": "alice only"},
        headers=alice_headers,
    )
    convo_id = alice_convo.json()["id"]

    bob_view = await client.get(
        f"/api/conversations/{convo_id}", headers=bob_headers
    )
    assert bob_view.status_code == 404

    bob_delete = await client.delete(
        f"/api/conversations/{convo_id}", headers=bob_headers
    )
    assert bob_delete.status_code == 404

    bob_list = await client.get("/api/conversations", headers=bob_headers)
    assert bob_list.json()["items"] == []


@pytest.mark.asyncio
@respx.mock
async def test_invalid_token_is_rejected(client, seeded_user):
    stub_auth_validate(respx.mock, seeded_user, token="valid-token")
    response = await client.get(
        "/api/conversations",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401
