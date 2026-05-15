from langchain_core.tools import tool

from cerone import CeroneClient
from cerone_langchain import govern_agent_tools


@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


client = CeroneClient()

governed_tools, governor = govern_agent_tools(
    tools=[read_file],
    client=client,
    preset="coding_agent",
    capabilities=["file_read"],
    environment="development",
    agent_name="repo-inspector",
    workflow_id="wf_repo_review",
    workflow_step="inspect_repository",
    default_tags=["code-review"],
)

if __name__ == "__main__":
    print(governed_tools[0].invoke({"path": "README.md"}))
