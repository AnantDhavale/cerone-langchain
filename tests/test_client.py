from __future__ import annotations

import json
from pathlib import Path

import httpx

from cerone import ActionContext, ActionPayload, CeroneClient


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
