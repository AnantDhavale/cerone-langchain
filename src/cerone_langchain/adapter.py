from __future__ import annotations

from threading import Lock
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field

from cerone import (
    ActionContext,
    ActionPayload,
    AgentRegistration,
    CeroneActionRejectedError,
    CeroneApprovalRequiredError,
    CeroneClient,
    CeroneConfigurationError,
)


FlaggedBehavior = Literal["allow", "raise"]


class ToolGovernor:
    """Coordinates Cerone agent bootstrap and pre-tool-call validation."""

    def __init__(
        self,
        *,
        client: CeroneClient,
        purpose: str,
        capabilities: list[str],
        environment: Literal["development", "staging", "production"] = "development",
        flagged_behavior: FlaggedBehavior = "raise",
        blocking: bool = True,
        timeout_ms: int = 1000,
        source: str = "langchain",
        agent_id: str | None = None,
        auto_register: bool = True,
    ) -> None:
        if not purpose.strip():
            raise CeroneConfigurationError("purpose is required")
        if not capabilities:
            raise CeroneConfigurationError("capabilities must include at least one entry")

        self.client = client
        self.purpose = purpose
        self.capabilities = capabilities
        self.environment = environment
        self.flagged_behavior = flagged_behavior
        self.blocking = blocking
        self.timeout_ms = timeout_ms
        self.source = source
        self.agent_id = agent_id
        self.auto_register = auto_register
        self._lock = Lock()

    def ensure_agent_id(self) -> str:
        if self.agent_id:
            return self.agent_id
        if not self.auto_register:
            raise CeroneConfigurationError(
                "agent_id is required when auto_register is disabled"
            )
        with self._lock:
            if self.agent_id:
                return self.agent_id
            cert = self.client.register_agent(
                AgentRegistration(
                    purpose=self.purpose,
                    capabilities=self.capabilities,
                    environment=self.environment,
                )
            )
            self.agent_id = cert.certificate.agent_id
            return self.agent_id

    async def ensure_agent_id_async(self) -> str:
        if self.agent_id:
            return self.agent_id
        if not self.auto_register:
            raise CeroneConfigurationError(
                "agent_id is required when auto_register is disabled"
            )
        cert = await self.client.register_agent_async(
            AgentRegistration(
                purpose=self.purpose,
                capabilities=self.capabilities,
                environment=self.environment,
            )
        )
        self.agent_id = cert.certificate.agent_id
        return self.agent_id

    def validate_or_raise(
        self,
        *,
        tool_name: str,
        tool_input: str | dict[str, Any],
        config: RunnableConfig | None,
    ) -> None:
        validation = self.client.validate(
            agent_id=self.ensure_agent_id(),
            action=self._build_action(tool_name=tool_name, tool_input=tool_input, config=config),
            blocking=self.blocking,
            timeout_ms=self.timeout_ms,
        )
        self._enforce_validation(tool_name, validation)

    async def validate_or_raise_async(
        self,
        *,
        tool_name: str,
        tool_input: str | dict[str, Any],
        config: RunnableConfig | None,
    ) -> None:
        validation = await self.client.validate_async(
            agent_id=await self.ensure_agent_id_async(),
            action=self._build_action(tool_name=tool_name, tool_input=tool_input, config=config),
            blocking=self.blocking,
            timeout_ms=self.timeout_ms,
        )
        self._enforce_validation(tool_name, validation)

    def _build_action(
        self,
        *,
        tool_name: str,
        tool_input: str | dict[str, Any],
        config: RunnableConfig | None,
    ) -> ActionPayload:
        tags = list(config.get("tags") or []) if config else []
        metadata = dict(config.get("metadata") or {}) if config else {}
        run_id = str(config.get("run_id")) if config and config.get("run_id") else None
        parameters = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
        return ActionPayload(
            tool=tool_name,
            parameters=parameters,
            context=ActionContext(
                source=self.source,
                run_id=run_id,
                tags=tags,
                metadata=metadata,
            ),
        )

    def _enforce_validation(self, tool_name: str, validation: Any) -> None:
        reason = validation.primary_reason or f"Cerone blocked {tool_name}"
        if validation.trial_stoploss:
            raise CeroneActionRejectedError("Trial limit reached")
        if validation.result == "rejected":
            raise CeroneActionRejectedError(reason)
        if validation.result == "flagged" and self.flagged_behavior == "raise":
            raise CeroneApprovalRequiredError(reason)


class CeroneTool(BaseTool):
    """LangChain tool wrapper that validates execution with Cerone before running."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    governor: ToolGovernor
    wrapped_tool: BaseTool = Field(exclude=True)

    def invoke(
        self,
        input: str | dict | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_input = _normalize_tool_input(input)
        self.governor.validate_or_raise(
            tool_name=self.wrapped_tool.name,
            tool_input=tool_input,
            config=config,
        )
        return self.wrapped_tool.invoke(input, config=config, **kwargs)

    async def ainvoke(
        self,
        input: str | dict | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_input = _normalize_tool_input(input)
        await self.governor.validate_or_raise_async(
            tool_name=self.wrapped_tool.name,
            tool_input=tool_input,
            config=config,
        )
        return await self.wrapped_tool.ainvoke(input, config=config, **kwargs)

    def run(self, tool_input: str | dict[str, Any], *args: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        self.governor.validate_or_raise(
            tool_name=self.wrapped_tool.name,
            tool_input=tool_input,
            config=config,
        )
        return self.wrapped_tool.run(tool_input, *args, config=config, **kwargs)

    async def arun(self, tool_input: str | dict, *args: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        normalized = tool_input if isinstance(tool_input, dict) else str(tool_input)
        await self.governor.validate_or_raise_async(
            tool_name=self.wrapped_tool.name,
            tool_input=normalized,
            config=config,
        )
        return await self.wrapped_tool.arun(tool_input, *args, config=config, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_tool._run(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return await self.wrapped_tool._arun(*args, **kwargs)


def govern_tool(tool: BaseTool, governor: ToolGovernor) -> CeroneTool:
    """Wrap a LangChain tool so Cerone validates it before execution."""
    return CeroneTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        return_direct=tool.return_direct,
        verbose=tool.verbose,
        callbacks=tool.callbacks,
        tags=tool.tags,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
        handle_validation_error=tool.handle_validation_error,
        response_format=tool.response_format,
        governor=governor,
        wrapped_tool=tool,
    )


def govern_tools(tools: list[BaseTool], governor: ToolGovernor) -> list[CeroneTool]:
    """Wrap a list of LangChain tools with the same Cerone governor."""
    return [govern_tool(tool, governor) for tool in tools]


def _normalize_tool_input(tool_input: Any) -> str | dict[str, Any]:
    if isinstance(tool_input, dict):
        return tool_input
    if hasattr(tool_input, "get") and callable(getattr(tool_input, "get")) and getattr(tool_input, "get")("args", None):
        maybe_args = tool_input.get("args")
        if isinstance(maybe_args, dict):
            return maybe_args
    return str(tool_input)
