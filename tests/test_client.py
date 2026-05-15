from __future__ import annotations

import json
from pathlib import Path

import httpx

from cerone import (
    ActionContext,
    ActionPayload,
    AgentRegistration,
    CeroneClient,
    ChildAgentRegistration,
)


def test_client_bootstraps_trial_and_validates(tmp_path: Path) -> None:
    captured: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode() or "{}")
        captured.append((request.url.path, body))
        if request.url.path == "/trial/session":
            assert request.headers["User-Agent"].startswith("cerone-python-sdk/")
            return httpx.Response(200, json={"trial_token": "sk_trial_123"})
        if request.url.path == "/v1/validate":
            assert request.headers["X-API-Key"] == "sk_trial_123"
            return httpx.Response(
                200,
                json={
                    "agent_id": "agt_test",
                    "result": "approved",
                    "checks": [],
                    "violations": [],
                    "semantic_alignment": 0.91,
                    "trust_score": 1.0,
                    "latency_ms": 12,
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = httpx.MockTransport(handler)
    client = CeroneClient(
        base_url="https://example.test",
        trial_token_path=tmp_path / "trial_token",
        sync_transport=transport,
        async_transport=transport,
    )

    response = client.validate(
        agent_id="agt_test",
        action=ActionPayload(
            tool="file_read",
            parameters={"path": "README.md"},
            context=ActionContext(source="langchain"),
        ),
    )

    assert response.result == "approved"
    assert [path for path, _ in captured] == ["/trial/session", "/v1/validate"]
    assert (tmp_path / "trial_token").read_text(encoding="utf-8") == "sk_trial_123"


def test_client_can_spawn_child_and_delegate_token(tmp_path: Path) -> None:
    captured: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode() or "{}")
        captured.append((request.url.path, body))
        if request.url.path == "/trial/session":
            return httpx.Response(200, json={"trial_token": "sk_trial_123"})
        if request.url.path == "/v1/certificates":
            return httpx.Response(
                201,
                json={
                    "certificate": {
                        "agent_id": "agt_parent",
                        "purpose": "Coordinate workflow steps.",
                        "capabilities": ["write:validate"],
                        "environment": "development",
                    },
                    "trust_score": 1.0,
                    "revoked": False,
                },
            )
        if request.url.path == "/v1/certificates/spawn":
            assert body["parent_id"] == "agt_parent"
            return httpx.Response(
                201,
                json={
                    "certificate": {
                        "agent_id": "agt_child",
                        "purpose": "Inspect repository files.",
                        "capabilities": ["file_read"],
                        "environment": "development",
                    },
                    "trust_score": 0.98,
                    "revoked": False,
                },
            )
        if request.url.path == "/v1/token/delegate":
            assert body["agent_id"] == "agt_child"
            assert body["parent_agent_id"] == "agt_parent"
            assert body["scope"] == "write:validate"
            return httpx.Response(
                200,
                json={
                    "access_token": "jwt_child",
                    "token_type": "Bearer",
                    "expires_in": 600,
                    "scope": "write:validate",
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = httpx.MockTransport(handler)
    client = CeroneClient(
        base_url="https://example.test",
        trial_token_path=tmp_path / "trial_token",
        sync_transport=transport,
        async_transport=transport,
    )

    parent = client.register_agent(
        AgentRegistration(
            purpose="Coordinate workflow steps.",
            capabilities=["write:validate"],
        )
    )
    child = client.spawn_agent(
        ChildAgentRegistration(
            parent_id=parent.certificate.agent_id,
            purpose="Inspect repository files.",
            capabilities=["file_read"],
        )
    )
    token = client.delegate_token(
        scope="write:validate",
        agent_id=child.certificate.agent_id,
        parent_agent_id=parent.certificate.agent_id,
    )

    assert child.certificate.agent_id == "agt_child"
    assert token.access_token == "jwt_child"
    assert [path for path, _ in captured] == [
        "/trial/session",
        "/v1/certificates",
        "/v1/certificates/spawn",
        "/v1/token/delegate",
    ]
