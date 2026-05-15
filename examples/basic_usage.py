from langchain_core.tools import tool

from cerone import CeroneClient
from cerone_langchain import ToolGovernor, govern_tool


@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


client = CeroneClient()

governor = ToolGovernor(
    client=client,
    purpose="Perform file_read operations to inspect repository files and answer software engineering questions.",
    capabilities=["file_read"],
    environment="development",
)

governed_tool = govern_tool(read_file, governor)

if __name__ == "__main__":
    print(governed_tool.invoke({"path": "README.md"}))
