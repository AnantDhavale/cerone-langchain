from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from langchain_core.tools import tool

from cerone import CeroneActionRejectedError, CeroneApprovalRequiredError, CeroneClient
from cerone_langchain import ToolGovernor, govern_agent_tools, govern_tool


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


def _build_workflow_client(tmp_path: Path) -> CeroneClient:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode() or "{}")
        if request.url.path == "/trial/session":
            return httpx.Response(200, json={"trial_token": "sk_trial_workflow"})
        if request.url.path == "/v1/certificates":
            return httpx.Response(
                201,
                json={
                    "certificate": {
                        "agent_id": "agt_parent",
                        "purpose": "Coordinate repository review workflow.",
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
                        "purpose": body["purpose"],
                        "capabilities": body["capabilities"],
                        "environment": "development",
                    },
                    "trust_score": 0.97,
                    "revoked": False,
                },
            )
        if request.url.path == "/v1/token/delegate":
            assert body["agent_id"] == "agt_child"
            assert body["parent_agent_id"] == "agt_parent"
            return httpx.Response(
                200,
                json={
                    "access_token": "jwt_child",
                    "token_type": "Bearer",
                    "expires_in": 600,
                    "scope": body["scope"],
                },
            )
        if request.url.path == "/v1/validate":
            assert body["agent_id"] == "agt_child"
            assert body["access_token"] == "jwt_child"
            assert body["action"]["context"]["workflow_id"] == "wf_repo_review"
            assert body["action"]["context"]["workflow_step"] == "inspect_repository"
            assert body["action"]["context"]["session_id"] == "sess_123"
            assert body["action"]["context"]["agent_name"] == "repo-inspector"
            assert body["action"]["context"]["parent_agent_id"] == "agt_parent"
            assert body["action"]["context"]["metadata"]["team"] == "eng"
            assert "code-review" in body["action"]["context"]["tags"]
            return httpx.Response(
                200,
                json={
                    "agent_id": "agt_child",
                    "result": "approved",
                    "checks": [],
                    "violations": [],
                    "semantic_alignment": 0.83,
                    "trust_score": 0.96,
                    "latency_ms": 14,
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


def test_governed_tool_allows_flagged_execution_when_configured(tmp_path: Path) -> None:
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"read:{path}"

    governor = ToolGovernor(
        client=_build_client(tmp_path, "flagged"),
        purpose="Perform file_read operations to inspect repository files.",
        capabilities=["file_read"],
        environment="development",
        flagged_behavior="allow",
    )
    wrapped = govern_tool(read_file, governor)

    assert wrapped.invoke({"path": "README.md"}) == "read:README.md"


def test_flagged_callback_can_reject_with_custom_policy(tmp_path: Path) -> None:
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"read:{path}"

    decisions: list[tuple[str, dict[str, str]]] = []

    def on_flagged(validation, tool_name: str, parameters: dict[str, str]) -> str:
        decisions.append((tool_name, parameters))
        assert validation.result == "flagged"
        return "reject"

    governor = ToolGovernor(
        client=_build_client(tmp_path, "flagged"),
        purpose="Perform file_read operations to inspect repository files.",
        capabilities=["file_read"],
        environment="development",
        flagged_behavior="allow",
        on_flagged=on_flagged,
    )
    wrapped = govern_tool(read_file, governor)

    with pytest.raises(CeroneActionRejectedError):
        wrapped.invoke({"path": "README.md"})

    assert decisions == [("read_file", {"path": "README.md"})]


def test_async_flagged_callback_can_allow_execution(tmp_path: Path) -> None:
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"read:{path}"

    async def on_flagged(validation, tool_name: str, parameters: dict[str, str]) -> str:
        assert validation.result == "flagged"
        assert tool_name == "read_file"
        assert parameters == {"path": "README.md"}
        return "allow"

    governor = ToolGovernor(
        client=_build_client(tmp_path, "flagged"),
        purpose="Perform file_read operations to inspect repository files.",
        capabilities=["file_read"],
        environment="development",
        flagged_behavior="raise",
        on_flagged=on_flagged,
    )
    wrapped = govern_tool(read_file, governor)

    assert asyncio.run(wrapped.ainvoke({"path": "README.md"})) == "read:README.md"


def test_govern_agent_tools_uses_preset_profile(tmp_path: Path) -> None:
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"read:{path}"

    tools, governor = govern_agent_tools(
        tools=[read_file],
        client=_build_client(tmp_path, "approved"),
        preset="coding_agent",
        capabilities=["file_read"],
        environment="development",
    )

    assert governor.purpose.startswith("Inspect repositories")
    assert "file_read" in governor.capabilities
    assert tools[0].invoke({"path": "README.md"}) == "read:README.md"


def test_spawn_child_governor_supports_workflow_context_and_delegation(tmp_path: Path) -> None:
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"read:{path}"

    client = _build_workflow_client(tmp_path)
    parent = ToolGovernor(
        client=client,
        purpose="Coordinate repository review workflow.",
        capabilities=["write:validate"],
        environment="development",
        session_id="sess_123",
        workflow_id="wf_repo_review",
        default_tags=["code-review"],
        default_metadata={"team": "eng"},
    )

    child = parent.spawn_child_governor(
        preset="coding_agent",
        capabilities=["file_read"],
        workflow_step="inspect_repository",
        agent_name="repo-inspector",
        delegate_scope="write:validate",
    )
    wrapped = govern_tool(read_file, child)

    assert wrapped.invoke({"path": "README.md"}) == "read:README.md"
