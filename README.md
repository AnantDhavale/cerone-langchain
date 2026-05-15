# langchain-runtime-governance

LangChain runtime governance powered by Cerone.

Cerone is a runtime governance layer for AI agents. You declare what an agent is
supposed to do, what capabilities it should have, and then Cerone validates each
tool call before execution.

For LangChain users, that means Cerone can sit in front of your tools and answer:

- is this tool call aligned with the agent's declared purpose?
- is the agent using a capability it was actually granted?
- should this tool call be allowed, flagged for review, or blocked?

This package contains two real layers:

- `cerone`: a reusable Python client for Cerone/AZTP
- `cerone_langchain`: a LangChain adapter that governs tool-executing agents and workflows before execution

## Why use it with LangChain?

LangChain makes it easy to give LLM-powered agents access to tools. The missing
piece for many teams is runtime control over what those tool calls are actually
allowed to do.

`cerone-langchain` is for cases like:

- coding agents that can read or modify files
- research agents that can call external APIs
- workflow agents that can touch customer or business systems
- multi-tool assistants that need a review or block step before risky actions
- multi-step or multi-agent workflows where child agents need scoped delegated authority

Instead of trusting the model prompt alone, you can put Cerone in the execution
path and get explicit runtime decisions:

- `approved` means the tool runs
- `flagged` means your app can escalate or review the action
- `rejected` means the tool is blocked before execution

## How it fits

At runtime, `cerone-langchain` does four things:

1. Creates or reuses a Cerone agent identity
2. Sends the tool name and parameters to Cerone before the tool runs
3. Receives a governance decision from Cerone
4. Allows, flags, or blocks the LangChain tool call based on that decision

If you do not provide an API key, the shared Cerone client can bootstrap a
hosted trial token automatically.

## Agent-oriented features

The package supports more than one-off tool wrapping:

- agent-level toolset setup with `govern_agent_tools(...)`
- runtime binding for `agent_name`, `session_id`, `workflow_id`, `workflow_step`, tags, and metadata
- built-in agent presets such as `coding_agent`, `research_agent`, and `support_agent`
- child-agent spawning for multi-step workflows using Cerone certificate lineage
- delegated access tokens for child workflow steps when a step needs scoped authority

## Install

```bash
pip install langchain-runtime-governance
```

Python imports stay:

```python
from cerone import CeroneClient
from cerone_langchain import ToolGovernor, govern_tool
```

## What it does

For each governed tool call, the adapter:

1. Bootstraps a Cerone trial token automatically if no `apiKey` is provided
2. Registers a Cerone agent if no `agent_id` is set
3. Sends the LangChain tool name and parameters to `/v1/validate`
4. Applies the Cerone result:
   - `approved` → run the tool
   - `flagged` → raise `CeroneApprovalRequiredError` by default
   - `rejected` → raise `CeroneActionRejectedError`

`flagged` handling is programmable. You can:

- set `flagged_behavior="allow"` to continue by default
- set `flagged_behavior="reject"` to hard-block flagged actions
- provide `on_flagged` to route flagged decisions into your own approval or policy workflow

## Data Handling

`cerone-langchain` sends tool invocation data to the Cerone API at runtime for
validation.

Depending on how your LangChain tools are defined, that runtime data may include:

- tool names
- tool arguments and structured parameters
- file paths
- URLs
- prompts, queries, or other operational text
- run tags or metadata passed through LangChain config

Do not use Cerone with sensitive or regulated data unless your organization has
approved that data flow. Avoid placing secrets or unnecessary sensitive content
directly in tool arguments where possible.

See [PRIVACY.md](./PRIVACY.md) for the free-trial data handling policy.

## Quick example

```python
from langchain_core.tools import tool

from cerone import CeroneClient
from cerone_langchain import govern_agent_tools


@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


client = CeroneClient(
    api_key=None,
    base_url="https://aztp-homer-semantics.onrender.com",
)

governed_tools, governor = govern_agent_tools(
    tools=[read_file],
    client=client,
    preset="coding_agent",
    capabilities=["file_read"],
    environment="development",
    flagged_behavior="raise",
    agent_name="repo-inspector",
    session_id="sess_123",
    workflow_id="wf_repo_review",
    workflow_step="inspect_repository",
    default_tags=["code-review"],
    default_metadata={"team": "eng"},
)

result = governed_tools[0].invoke({"path": "README.md"})
print(result)
```

In that example:

- the LangChain tool list is governed as an agent toolset
- Cerone sees the agent purpose, capabilities, and bound runtime context
- every invocation of `read_file` is validated before execution
- if Cerone returns `approved`, the tool runs
- if Cerone returns `flagged` or `rejected`, the adapter stops normal execution unless your policy allows otherwise

## Workflow and child-agent support

For multi-step workflows, you can spawn a child governor from a parent governor.
This aligns with Cerone's certificate lineage and delegated-token model.

```python
child_governor = governor.spawn_child_governor(
    preset="coding_agent",
    capabilities=["file_read"],
    workflow_step="inspect_repository",
    agent_name="repo-inspector",
    delegate_scope="write:validate",
)
```

That child governor:

- creates a child Cerone agent linked to the parent agent
- can request a delegated access token for the child step
- carries the parent agent ID into validation context
- lets you model multi-step LangChain workflows with explicit Cerone lineage

## Custom flagged handling

```python
from cerone_langchain import ToolGovernor


def on_flagged(validation, tool_name, parameters):
    print(f"Cerone flagged {tool_name}: {validation.primary_reason}")

    # Route into your own approval system here.
    if tool_name == "read_file":
        return "allow"
    return "reject"


governor = ToolGovernor(
    client=client,
    purpose="Perform file_read operations to inspect repository files and answer software engineering questions.",
    capabilities=["file_read"],
    environment="development",
    flagged_behavior="raise",
    on_flagged=on_flagged,
)
```

The callback may return:

- `"allow"` to proceed with the tool call
- `"reject"` to raise `CeroneActionRejectedError`
- `"raise"` to raise `CeroneApprovalRequiredError`

If the callback returns `None`, `flagged_behavior` is used as the fallback policy.

## Core client

The shared `cerone` client supports:

- hosted trial bootstrap via `/trial/session`
- certificate creation via `/v1/certificates`
- child certificate creation via `/v1/certificates/spawn`
- delegated token issuance via `/v1/token/delegate`
- validation via `/v1/validate`
- sync and async flows

This shared layer is intended to be reused by future adapters such as CrewAI or LlamaIndex.

## Notes

- The adapter wraps LangChain tools directly rather than relying on passive callbacks, so it can stop execution before the tool runs.
- The package governs LangChain agents and workflows at the tool-execution layer, where real external actions happen.
- The default `flagged_behavior="raise"` is deliberate because LangChain does not provide a built-in cross-runtime human approval UI, but `on_flagged` gives you a clean hook into your own review or escalation path.
- The Cerone trial token is cached locally under `~/.cerone/trial_token` by default.

## License

MIT
