from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Environment = Literal["development", "staging", "production"]
ValidationResult = Literal["approved", "flagged", "rejected"]


class ActionContext(BaseModel):
    source: str = "langchain"
    agent_name: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    workflow_id: str | None = None
    workflow_step: str | None = None
    parent_agent_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionPayload(BaseModel):
    tool: str
    parameters: dict[str, Any]
    context: ActionContext | None = None


class ValidationRequest(BaseModel):
    agent_id: str
    action: ActionPayload
    access_token: str | None = None
    blocking: bool = True
    timeout_ms: int = 1000


class ValidationCheck(BaseModel):
    check_name: str
    result: str
    details: str | None = None
    execution_time_ms: int | None = None


class ValidationResponse(BaseModel):
    validation_id: str | None = None
    agent_id: str
    result: ValidationResult
    checks: list[ValidationCheck] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    semantic_alignment: float | None = None
    trust_score: float | None = None
    latency_ms: int | None = None
    trial_warning: bool = False
    trial_stoploss: bool = False
    environment_mode: str | None = None
    note: str | None = None
    hint: str | None = None
    timestamp: datetime | None = None

    @property
    def primary_reason(self) -> str | None:
        if self.violations:
            return self.violations[0]
        for check in self.checks:
            if check.details:
                return check.details
        return None


class CertificateData(BaseModel):
    agent_id: str
    purpose: str
    capabilities: list[str] = Field(default_factory=list)
    environment: str | None = None


class CertificateResponse(BaseModel):
    certificate: CertificateData
    trust_score: float | None = None
    revoked: bool | None = None


class TrialSessionResponse(BaseModel):
    trial_token: str
    trial_session_id: str | None = None
    validations_remaining: int | None = None
    trial_stoploss_limit: int | None = None
    hard_validations_limit: int | None = None


class AgentRegistration(BaseModel):
    purpose: str
    capabilities: list[str]
    environment: Environment = "development"


class ChildAgentRegistration(BaseModel):
    parent_id: str
    purpose: str
    capabilities: list[str]
    max_lifespan_hours: int = 24
    environment: Environment = "development"


class DelegatedTokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    scope: str
