# cerone-langchain

`cerone-langchain` adds Cerone governance to LangChain tools before they run.

The package contains two real layers:

- `cerone`: a reusable Python client for Cerone/AZTP
- `cerone_langchain`: a LangChain adapter that wraps `BaseTool` instances and validates them before execution

## Install

```bash
pip install cerone-langchain
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

## Example

```python
from langchain_core.tools import tool

from cerone import CeroneClient
from cerone_langchain import ToolGovernor, govern_tool


@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


client = CeroneClient(
    api_key=None,
    base_url="https://aztp-homer-semantics.onrender.com",
)

governor = ToolGovernor(
    client=client,
    purpose="Perform file_read operations to inspect repository files and answer software engineering questions.",
    capabilities=["file_read"],
    environment="development",
    flagged_behavior="raise",
)

governed_read_file = govern_tool(read_file, governor)

result = governed_read_file.invoke({"path": "README.md"})
print(result)
```

## Core client

The shared `cerone` client supports:

- hosted trial bootstrap via `/trial/session`
- certificate creation via `/v1/certificates`
- validation via `/v1/validate`
- sync and async flows

This shared layer is intended to be reused by future adapters such as CrewAI or LlamaIndex.

## Notes

- The adapter wraps LangChain tools directly rather than relying on passive callbacks, so it can stop execution before the tool runs.
- The default `flagged_behavior="raise"` is deliberate because LangChain does not provide a built-in cross-runtime human approval UI.
- The Cerone trial token is cached locally under `~/.cerone/trial_token` by default.

## License

MIT
