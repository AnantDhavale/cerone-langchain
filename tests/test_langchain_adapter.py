from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from langchain_core.tools import tool

from cerone import CeroneActionRejectedError, CeroneApprovalRequiredError, CeroneClient
from cerone_langchain import ToolGovernor, govern_tool


def _build_client(tmp_path: Path, validation_result: str) -> CeroneClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/trial/session":
            return httpx.Response(200, json={"trial_token": "sk_trial_abc"})
        if request.url.path == "/v1/certificates":
            return httpx.Response(
                201,
                json={
                    "certificate": {
                        "agent_id": "agt_langchain",
                        "purpose": "Perform file_read operations to inspect repository files.",
                        "capabilities": ["file_read"],
                        "environment": "development",
                    },
                    "trust_score": 1.0,
                    "revoked": False,
                },
            )
        if request.url.path == "/v1/validate":
            body = json.loads(request.content.decode())
            assert body["action"]["tool"] == "read_file"
            assert body["action"]["parameters"]["path"] == "README.md"
            return httpx.Response(
                200,
                json={
                    "agent_id": "agt_langchain",
                    "result": validation_result,
                    "checks": [],
                    "violations": ["policy review required"] if validation_result != "approved" else [],
                    "semantic_alignment": 0.61,
                    "trust_score": 0.95,
                    "latency_ms": 22,
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = httpx.MockTransport(handler)
    return CeroneClient(
        base_url="https://example.test",
        trial_token_path=tmp_path / "trial_token",
        sync_transport=transport,
        async_transport=transport,
    )


def test_governed_tool_allows_approved_execution(tmp_path: Path) -> None:
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"read:{path}"

    governor = ToolGovernor(
        client=_build_client(tmp_path, "approved"),
        purpose="Perform file_read operations to inspect repository files.",
        capabilities=["file_read"],
        environment="development",
        flagged_behavior="raise",
    )
    wrapped = govern_tool(read_file, governor)

    assert wrapped.invoke({"path": "README.md"}) == "read:README.md"


def test_governed_tool_raises_on_flagged_when_policy_requires_it(tmp_path: Path) -> None:
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"read:{path}"

    governor = ToolGovernor(
        client=_build_client(tmp_path, "flagged"),
        purpose="Perform file_read operations to inspect repository files.",
        capabilities=["file_read"],
        environment="development",
        flagged_behavior="raise",
    )
    wrapped = govern_tool(read_file, governor)

    with pytest.raises(CeroneApprovalRequiredError):
        wrapped.invoke({"path": "README.md"})


def test_governed_tool_raises_on_rejected(tmp_path: Path) -> None:
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"read:{path}"

    governor = ToolGovernor(
        client=_build_client(tmp_path, "rejected"),
        purpose="Perform file_read operations to inspect repository files.",
        capabilities=["file_read"],
        environment="development",
    )
    wrapped = govern_tool(read_file, governor)

    with pytest.raises(CeroneActionRejectedError):
        wrapped.invoke({"path": "README.md"})
