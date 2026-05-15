from __future__ import annotations

from dataclasses import dataclass

from cerone import CeroneConfigurationError


@dataclass(frozen=True)
class AgentPreset:
    name: str
    purpose: str
    capabilities: tuple[str, ...]
    description: str


PRESETS: dict[str, AgentPreset] = {
    "coding_agent": AgentPreset(
        name="coding_agent",
        purpose=(
            "Inspect repositories, read and modify project files, run engineering "
            "workflows, and answer software engineering questions."
        ),
        capabilities=("file_read", "file_write", "network_access", "api_call"),
        description="For code assistants, repository agents, and engineering workflows.",
    ),
    "research_agent": AgentPreset(
        name="research_agent",
        purpose=(
            "Gather external information, inspect reference material, and synthesize "
            "research findings using approved network and API access."
        ),
        capabilities=("network_access", "api_call", "file_read"),
        description="For web research, fact gathering, and reference-driven agents.",
    ),
    "support_agent": AgentPreset(
        name="support_agent",
        purpose=(
            "Review customer support context, retrieve operational data, and update "
            "support systems to resolve approved service requests."
        ),
        capabilities=("db_read", "db_write", "api_call"),
        description="For support, service, and operational ticket workflows.",
    ),
}


def get_preset(name: str) -> AgentPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PRESETS))
        raise CeroneConfigurationError(
            f"unknown preset '{name}'. Available presets: {available}"
        ) from exc


def resolve_profile(
    *,
    preset: str | AgentPreset | None,
    purpose: str | None,
    capabilities: list[str] | None,
) -> tuple[str, list[str]]:
    if preset is None:
        if not purpose or not purpose.strip():
            raise CeroneConfigurationError("purpose is required")
        if not capabilities:
            raise CeroneConfigurationError(
                "capabilities must include at least one entry"
            )
        return purpose.strip(), list(capabilities)

    preset_obj = get_preset(preset) if isinstance(preset, str) else preset
    resolved_purpose = purpose.strip() if purpose and purpose.strip() else preset_obj.purpose

    merged_capabilities: list[str] = []
    for capability in [*preset_obj.capabilities, *(capabilities or [])]:
        if capability not in merged_capabilities:
            merged_capabilities.append(capability)

    return resolved_purpose, merged_capabilities
