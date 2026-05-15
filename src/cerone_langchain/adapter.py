from __future__ import annotations

import inspect
from threading import Lock
from typing import Any, Awaitable, Callable, Literal, TypeAlias

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
    ChildAgentRegistration,
    ValidationResponse,
)

from .presets import AgentPreset, resolve_profile


FlaggedDecision = Literal["allow", "raise", "reject"]
FlaggedBehavior = FlaggedDecision
FlaggedHandler: TypeAlias = Callable[
    [ValidationResponse, str, dict[str, Any]],
    FlaggedDecision | None | Awaitable[FlaggedDecision | None],
]


class ToolGovernor:
    """Coordinates Cerone agent bootstrap and pre-tool-call validation."""

    def __init__(
        self,
        *,
        client: CeroneClient,
        purpose: str | None = None,
        capabilities: list[str] | None = None,
        preset: str | AgentPreset | None = None,
        environment: Literal["development", "staging", "production"] = "development",
        flagged_behavior: FlaggedBehavior = "raise",
        on_flagged: FlaggedHandler | None = None,
        blocking: bool = True,
        timeout_ms: int = 1000,
        source: str = "langchain",
        agent_id: str | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        workflow_id: str | None = None,
        workflow_step: str | None = None,
        default_tags: list[str] | None = None,
        default_metadata: dict[str, Any] | None = None,
        access_token: str | None = None,
        parent_agent_id: str | None = None,
        auto_register: bool = True,
    ) -> None:
        resolved_purpose, resolved_capabilities = resolve_profile(
            preset=preset,
            purpose=purpose,
            capabilities=capabilities,
        )

        self.client = client
        self.purpose = resolved_purpose
        self.capabilities = resolved_capabilities
        self.preset = preset
        self.environment = environment
        self.flagged_behavior = flagged_behavior
        self.on_flagged = on_flagged
        self.blocking = blocking
        self.timeout_ms = timeout_ms
        self.source = source
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.session_id = session_id
        self.workflow_id = workflow_id
        self.workflow_step = workflow_step
        self.default_tags = list(default_tags or [])
        self.default_metadata = dict(default_metadata or {})
        self.access_token = access_token
        self.parent_agent_id = parent_agent_id
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
            access_token=self.access_token,
            blocking=self.blocking,
            timeout_ms=self.timeout_ms,
        )
        self._enforce_validation(tool_name, tool_input, validation)

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
            access_token=self.access_token,
            blocking=self.blocking,
            timeout_ms=self.timeout_ms,
        )
        await self._enforce_validation_async(tool_name, tool_input, validation)

    def _build_action(
        self,
        *,
        tool_name: str,
        tool_input: str | dict[str, Any],
        config: RunnableConfig | None,
    ) -> ActionPayload:
        tags = _merge_unique(self.default_tags, list(config.get("tags") or []) if config else [])
        metadata = dict(self.default_metadata)
        if config and config.get("metadata"):
            metadata.update(dict(config.get("metadata") or {}))
        run_id = str(config.get("run_id")) if config and config.get("run_id") else None
        parameters = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
        return ActionPayload(
            tool=tool_name,
            parameters=parameters,
            context=ActionContext(
                source=self.source,
                agent_name=self.agent_name,
                run_id=run_id,
                session_id=self.session_id,
                workflow_id=self.workflow_id,
                workflow_step=self.workflow_step,
                parent_agent_id=self.parent_agent_id,
                tags=tags,
                metadata=metadata,
            ),
        )

    def bind_context(
        self,
        *,
        agent_name: str | None = None,
        session_id: str | None = None,
        workflow_id: str | None = None,
        workflow_step: str | None = None,
        default_tags: list[str] | None = None,
        default_metadata: dict[str, Any] | None = None,
        access_token: str | None = None,
    ) -> ToolGovernor:
        return self._clone(
            agent_name=self.agent_name if agent_name is None else agent_name,
            session_id=self.session_id if session_id is None else session_id,
            workflow_id=self.workflow_id if workflow_id is None else workflow_id,
            workflow_step=self.workflow_step if workflow_step is None else workflow_step,
            default_tags=(
                list(self.default_tags)
                if default_tags is None
                else _merge_unique(self.default_tags, default_tags)
            ),
            default_metadata=(
                dict(self.default_metadata)
                if default_metadata is None
                else {**self.default_metadata, **default_metadata}
            ),
            access_token=self.access_token if access_token is None else access_token,
        )

    def for_workflow_step(
        self,
        workflow_step: str,
        *,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> ToolGovernor:
        return self.bind_context(
            workflow_step=workflow_step,
            default_metadata=metadata,
            default_tags=tags,
        )

    def delegate_access_token(
        self,
        *,
        scope: str,
        audience: str = "aztp-api",
        ttl_minutes: int = 10,
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        token = self.client.delegate_token(
            scope=scope,
            audience=audience,
            ttl_minutes=ttl_minutes,
            agent_id=self.ensure_agent_id(),
            parent_agent_id=self.parent_agent_id,
            extra_claims=extra_claims,
        )
        return token.access_token

    async def delegate_access_token_async(
        self,
        *,
        scope: str,
        audience: str = "aztp-api",
        ttl_minutes: int = 10,
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        token = await self.client.delegate_token_async(
            scope=scope,
            audience=audience,
            ttl_minutes=ttl_minutes,
            agent_id=await self.ensure_agent_id_async(),
            parent_agent_id=self.parent_agent_id,
            extra_claims=extra_claims,
        )
        return token.access_token

    def spawn_child_governor(
        self,
        *,
        purpose: str | None = None,
        capabilities: list[str] | None = None,
        preset: str | AgentPreset | None = None,
        environment: Literal["development", "staging", "production"] | None = None,
        max_lifespan_hours: int = 24,
        workflow_step: str | None = None,
        agent_name: str | None = None,
        delegate_scope: str | None = None,
        delegate_audience: str = "aztp-api",
        delegate_ttl_minutes: int = 10,
        delegate_extra_claims: dict[str, Any] | None = None,
    ) -> ToolGovernor:
        parent_id = self.ensure_agent_id()
        resolved_purpose, resolved_capabilities = resolve_profile(
            preset=preset,
            purpose=purpose,
            capabilities=capabilities,
        )
        child_environment = environment or self.environment
        cert = self.client.spawn_agent(
            ChildAgentRegistration(
                parent_id=parent_id,
                purpose=resolved_purpose,
                capabilities=resolved_capabilities,
                max_lifespan_hours=max_lifespan_hours,
                environment=child_environment,
            )
        )
        access_token = None
        if delegate_scope is not None:
            token = self.client.delegate_token(
                scope=delegate_scope,
                audience=delegate_audience,
                ttl_minutes=delegate_ttl_minutes,
                agent_id=cert.certificate.agent_id,
                parent_agent_id=parent_id,
                extra_claims=delegate_extra_claims,
            )
            access_token = token.access_token
        return self._clone(
            purpose=resolved_purpose,
            capabilities=resolved_capabilities,
            preset=preset,
            environment=child_environment,
            agent_id=cert.certificate.agent_id,
            agent_name=agent_name,
            workflow_step=workflow_step,
            access_token=access_token,
            parent_agent_id=parent_id,
            auto_register=False,
        )

    async def spawn_child_governor_async(
        self,
        *,
        purpose: str | None = None,
        capabilities: list[str] | None = None,
        preset: str | AgentPreset | None = None,
        environment: Literal["development", "staging", "production"] | None = None,
        max_lifespan_hours: int = 24,
        workflow_step: str | None = None,
        agent_name: str | None = None,
        delegate_scope: str | None = None,
        delegate_audience: str = "aztp-api",
        delegate_ttl_minutes: int = 10,
        delegate_extra_claims: dict[str, Any] | None = None,
    ) -> ToolGovernor:
        parent_id = await self.ensure_agent_id_async()
        resolved_purpose, resolved_capabilities = resolve_profile(
            preset=preset,
            purpose=purpose,
            capabilities=capabilities,
        )
        child_environment = environment or self.environment
        cert = await self.client.spawn_agent_async(
            ChildAgentRegistration(
                parent_id=parent_id,
                purpose=resolved_purpose,
                capabilities=resolved_capabilities,
                max_lifespan_hours=max_lifespan_hours,
                environment=child_environment,
            )
        )
        access_token = None
        if delegate_scope is not None:
            token = await self.client.delegate_token_async(
                scope=delegate_scope,
                audience=delegate_audience,
                ttl_minutes=delegate_ttl_minutes,
                agent_id=cert.certificate.agent_id,
                parent_agent_id=parent_id,
                extra_claims=delegate_extra_claims,
            )
            access_token = token.access_token
        return self._clone(
            purpose=resolved_purpose,
            capabilities=resolved_capabilities,
            preset=preset,
            environment=child_environment,
            agent_id=cert.certificate.agent_id,
            agent_name=agent_name,
            workflow_step=workflow_step,
            access_token=access_token,
            parent_agent_id=parent_id,
            auto_register=False,
        )

    def _enforce_validation(
        self,
        tool_name: str,
        tool_input: str | dict[str, Any],
        validation: ValidationResponse,
    ) -> None:
        reason = validation.primary_reason or f"Cerone blocked {tool_name}"
        if validation.trial_stoploss:
            raise CeroneActionRejectedError("Trial limit reached")
        if validation.result == "rejected":
            raise CeroneActionRejectedError(reason)
        if validation.result != "flagged":
            return

        decision = self.flagged_behavior
        if self.on_flagged is not None:
            maybe_decision = self.on_flagged(validation, tool_name, _coerce_parameters(tool_input))
            if inspect.isawaitable(maybe_decision):
                raise CeroneConfigurationError(
                    "on_flagged returned an awaitable during synchronous execution; "
                    "use a synchronous callback or call ainvoke/arun instead"
                )
            if maybe_decision is not None:
                decision = maybe_decision

        self._apply_flagged_decision(tool_name, reason, decision)

    async def _enforce_validation_async(
        self,
        tool_name: str,
        tool_input: str | dict[str, Any],
        validation: ValidationResponse,
    ) -> None:
        reason = validation.primary_reason or f"Cerone blocked {tool_name}"
        if validation.trial_stoploss:
            raise CeroneActionRejectedError("Trial limit reached")
        if validation.result == "rejected":
            raise CeroneActionRejectedError(reason)
        if validation.result != "flagged":
            return

        decision = self.flagged_behavior
        if self.on_flagged is not None:
            maybe_decision = self.on_flagged(validation, tool_name, _coerce_parameters(tool_input))
            if inspect.isawaitable(maybe_decision):
                maybe_decision = await maybe_decision
            if maybe_decision is not None:
                decision = maybe_decision

        self._apply_flagged_decision(tool_name, reason, decision)

    def _apply_flagged_decision(
        self,
        tool_name: str,
        reason: str,
        decision: FlaggedDecision,
    ) -> None:
        if decision == "allow":
            return
        if decision == "reject":
            raise CeroneActionRejectedError(reason)
        if decision == "raise":
            raise CeroneApprovalRequiredError(reason)
        raise CeroneConfigurationError(f"unsupported flagged decision: {decision}")

    def _clone(self, **overrides: Any) -> ToolGovernor:
        return ToolGovernor(
            client=self.client,
            purpose=overrides.pop("purpose", self.purpose),
            capabilities=overrides.pop("capabilities", list(self.capabilities)),
            preset=overrides.pop("preset", self.preset),
            environment=overrides.pop("environment", self.environment),
            flagged_behavior=overrides.pop("flagged_behavior", self.flagged_behavior),
            on_flagged=overrides.pop("on_flagged", self.on_flagged),
            blocking=overrides.pop("blocking", self.blocking),
            timeout_ms=overrides.pop("timeout_ms", self.timeout_ms),
            source=overrides.pop("source", self.source),
            agent_id=overrides.pop("agent_id", self.agent_id),
            agent_name=overrides.pop("agent_name", self.agent_name),
            session_id=overrides.pop("session_id", self.session_id),
            workflow_id=overrides.pop("workflow_id", self.workflow_id),
            workflow_step=overrides.pop("workflow_step", self.workflow_step),
            default_tags=overrides.pop("default_tags", list(self.default_tags)),
            default_metadata=overrides.pop("default_metadata", dict(self.default_metadata)),
            access_token=overrides.pop("access_token", self.access_token),
            parent_agent_id=overrides.pop("parent_agent_id", self.parent_agent_id),
            auto_register=overrides.pop("auto_register", self.auto_register),
        )


def govern_agent_tools(
    *,
    tools: list[BaseTool],
    client: CeroneClient,
    purpose: str | None = None,
    capabilities: list[str] | None = None,
    preset: str | AgentPreset | None = None,
    environment: Literal["development", "staging", "production"] = "development",
    flagged_behavior: FlaggedBehavior = "raise",
    on_flagged: FlaggedHandler | None = None,
    blocking: bool = True,
    timeout_ms: int = 1000,
    source: str = "langchain",
    agent_id: str | None = None,
    agent_name: str | None = None,
    session_id: str | None = None,
    workflow_id: str | None = None,
    workflow_step: str | None = None,
    default_tags: list[str] | None = None,
    default_metadata: dict[str, Any] | None = None,
    access_token: str | None = None,
    parent_agent_id: str | None = None,
    auto_register: bool = True,
) -> tuple[list[CeroneTool], ToolGovernor]:
    governor = ToolGovernor(
        client=client,
        purpose=purpose,
        capabilities=capabilities,
        preset=preset,
        environment=environment,
        flagged_behavior=flagged_behavior,
        on_flagged=on_flagged,
        blocking=blocking,
        timeout_ms=timeout_ms,
        source=source,
        agent_id=agent_id,
        agent_name=agent_name,
        session_id=session_id,
        workflow_id=workflow_id,
        workflow_step=workflow_step,
        default_tags=default_tags,
        default_metadata=default_metadata,
        access_token=access_token,
        parent_agent_id=parent_agent_id,
        auto_register=auto_register,
    )
    return govern_tools(tools, governor), governor


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


def _coerce_parameters(tool_input: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(tool_input, dict):
        return tool_input
    return {"input": tool_input}


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for value in [*left, *right]:
        if value not in merged:
            merged.append(value)
    return merged
