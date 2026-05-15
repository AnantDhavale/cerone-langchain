from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx

from .errors import CeroneHTTPError
from .models import (
    ActionPayload,
    AgentRegistration,
    CertificateResponse,
    ChildAgentRegistration,
    DelegatedTokenResponse,
    TrialSessionResponse,
    ValidationRequest,
    ValidationResponse,
)

DEFAULT_BASE_URL = "https://aztp-homer-semantics.onrender.com"
DEFAULT_USER_AGENT = "cerone-python-sdk/0.1.0"


class CeroneClient:
    """Shared Cerone Python client for framework adapters."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 5.0,
        trial_token_path: str | os.PathLike[str] | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        sync_transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.user_agent = user_agent
        self.trial_token_path = Path(
            trial_token_path
            or os.environ.get("CERONE_TRIAL_TOKEN_PATH", "~/.cerone/trial_token")
        ).expanduser()

        headers = {"User-Agent": user_agent}
        if api_key:
            headers["X-API-Key"] = api_key

        self._sync_client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers.copy(),
            transport=sync_transport,
        )
        self._async_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers.copy(),
            transport=async_transport,
        )

    def close(self) -> None:
        self._sync_client.close()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._async_client.aclose())
            return
        loop.create_task(self._async_client.aclose())

    async def aclose(self) -> None:
        self._sync_client.close()
        await self._async_client.aclose()

    def _load_cached_trial_token(self) -> str | None:
        if self.api_key:
            return self.api_key
        try:
            if self.trial_token_path.exists():
                token = self.trial_token_path.read_text(encoding="utf-8").strip()
                if token.startswith("sk_trial_"):
                    return token
        except OSError:
            return None
        return None

    def _persist_trial_token(self, token: str) -> None:
        try:
            self.trial_token_path.parent.mkdir(parents=True, exist_ok=True)
            self.trial_token_path.write_text(token, encoding="utf-8")
        except OSError:
            return

    def _set_api_key(self, token: str) -> None:
        self.api_key = token
        self._sync_client.headers["X-API-Key"] = token
        self._async_client.headers["X-API-Key"] = token

    def ensure_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        cached = self._load_cached_trial_token()
        if cached:
            self._set_api_key(cached)
            return cached
        response = self._sync_client.post("/trial/session", json={})
        self._raise_for_status(response)
        payload = TrialSessionResponse.model_validate(response.json())
        self._set_api_key(payload.trial_token)
        self._persist_trial_token(payload.trial_token)
        return payload.trial_token

    async def ensure_api_key_async(self) -> str:
        if self.api_key:
            return self.api_key
        cached = self._load_cached_trial_token()
        if cached:
            self._set_api_key(cached)
            return cached
        response = await self._async_client.post("/trial/session", json={})
        self._raise_for_status(response)
        payload = TrialSessionResponse.model_validate(response.json())
        self._set_api_key(payload.trial_token)
        self._persist_trial_token(payload.trial_token)
        return payload.trial_token

    def register_agent(self, registration: AgentRegistration) -> CertificateResponse:
        self.ensure_api_key()
        response = self._sync_client.post(
            "/v1/certificates",
            json=registration.model_dump(mode="json"),
        )
        self._raise_for_status(response)
        return CertificateResponse.model_validate(response.json())

    async def register_agent_async(self, registration: AgentRegistration) -> CertificateResponse:
        await self.ensure_api_key_async()
        response = await self._async_client.post(
            "/v1/certificates",
            json=registration.model_dump(mode="json"),
        )
        self._raise_for_status(response)
        return CertificateResponse.model_validate(response.json())

    def spawn_agent(self, registration: ChildAgentRegistration) -> CertificateResponse:
        self.ensure_api_key()
        response = self._sync_client.post(
            "/v1/certificates/spawn",
            json=registration.model_dump(mode="json"),
        )
        self._raise_for_status(response)
        return CertificateResponse.model_validate(response.json())

    async def spawn_agent_async(
        self, registration: ChildAgentRegistration
    ) -> CertificateResponse:
        await self.ensure_api_key_async()
        response = await self._async_client.post(
            "/v1/certificates/spawn",
            json=registration.model_dump(mode="json"),
        )
        self._raise_for_status(response)
        return CertificateResponse.model_validate(response.json())

    def validate(
        self,
        *,
        agent_id: str,
        action: ActionPayload,
        access_token: str | None = None,
        blocking: bool = True,
        timeout_ms: int = 1000,
    ) -> ValidationResponse:
        self.ensure_api_key()
        request = ValidationRequest(
            agent_id=agent_id,
            action=action,
            access_token=access_token,
            blocking=blocking,
            timeout_ms=timeout_ms,
        )
        response = self._sync_client.post("/v1/validate", json=request.model_dump(mode="json"))
        self._raise_for_status(response)
        return ValidationResponse.model_validate(response.json())

    async def validate_async(
        self,
        *,
        agent_id: str,
        action: ActionPayload,
        access_token: str | None = None,
        blocking: bool = True,
        timeout_ms: int = 1000,
    ) -> ValidationResponse:
        await self.ensure_api_key_async()
        request = ValidationRequest(
            agent_id=agent_id,
            action=action,
            access_token=access_token,
            blocking=blocking,
            timeout_ms=timeout_ms,
        )
        response = await self._async_client.post(
            "/v1/validate",
            json=request.model_dump(mode="json"),
        )
        self._raise_for_status(response)
        return ValidationResponse.model_validate(response.json())

    def delegate_token(
        self,
        *,
        scope: str,
        audience: str = "aztp-api",
        ttl_minutes: int = 10,
        agent_id: str | None = None,
        parent_agent_id: str | None = None,
        extra_claims: dict[str, Any] | None = None,
    ) -> DelegatedTokenResponse:
        self.ensure_api_key()
        payload: dict[str, Any] = {
            "scope": scope,
            "audience": audience,
            "ttl_minutes": ttl_minutes,
        }
        if agent_id:
            payload["agent_id"] = agent_id
        if parent_agent_id:
            payload["parent_agent_id"] = parent_agent_id
        if extra_claims:
            payload["extra_claims"] = extra_claims
        response = self._sync_client.post("/v1/token/delegate", json=payload)
        self._raise_for_status(response)
        return DelegatedTokenResponse.model_validate(response.json())

    async def delegate_token_async(
        self,
        *,
        scope: str,
        audience: str = "aztp-api",
        ttl_minutes: int = 10,
        agent_id: str | None = None,
        parent_agent_id: str | None = None,
        extra_claims: dict[str, Any] | None = None,
    ) -> DelegatedTokenResponse:
        await self.ensure_api_key_async()
        payload: dict[str, Any] = {
            "scope": scope,
            "audience": audience,
            "ttl_minutes": ttl_minutes,
        }
        if agent_id:
            payload["agent_id"] = agent_id
        if parent_agent_id:
            payload["parent_agent_id"] = parent_agent_id
        if extra_claims:
            payload["extra_claims"] = extra_claims
        response = await self._async_client.post("/v1/token/delegate", json=payload)
        self._raise_for_status(response)
        return DelegatedTokenResponse.model_validate(response.json())

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail: str
            try:
                payload: Any = response.json()
                if isinstance(payload, dict):
                    detail = str(payload.get("detail") or payload.get("message") or payload)
                else:
                    detail = str(payload)
            except Exception:
                detail = response.text
            raise CeroneHTTPError(detail or str(exc), response.status_code) from exc
