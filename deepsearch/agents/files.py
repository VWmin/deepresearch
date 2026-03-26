from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deepsearch.utils.state import ResearchState
from deepsearch.utils.files import STORE_DIR

import os

LS_DESCRIPTION = """List all files in the filesystem stored in agent state.

Shows what files currently exist in agent memory. Use this to orient yourself before other file operations and maintain awareness of your file organization.

No parameters required - simply call ls() to see all available files."""

READ_FILE_DESCRIPTION = """Read content from a file in the virtual filesystem with optional pagination.

This tool returns file content with line numbers (like `cat -n`) and supports reading large files in chunks to avoid context overflow.

Parameters:
- file_path (required): Path to the file you want to read
- offset (optional, default=0): Line number to start reading from  
- limit (optional, default=2000): Maximum number of lines to read

Essential before making any edits to understand existing content. Always read a file before editing it."""

WRITE_FILE_DESCRIPTION = """Create a new file or completely overwrite an existing file in the virtual filesystem.

This tool creates new files or replaces entire file contents. Use for initial file creation or complete rewrites. Files are stored persistently in agent state.

Parameters:
- file_path (required): Path where the file should be created/overwritten
- content (required): The complete content to write to the file

Important: This replaces the entire file content."""

FILE_USAGE_INSTRUCTIONS = """You have access to a virtual file system to help you retain and save context.

## Workflow Process
1. **Orient**: Use ls() to see existing files before starting work
2. **Save**: Use write_file() to store the user's request so that we can keep it for later 
3. **Research**: Proceed with research. The search tool will write files.  
4. **Read**: Once you are satisfied with the collected sources, read the files and use them to answer the user's question directly.
"""


@tool(description=LS_DESCRIPTION)
def ls(state: Annotated[ResearchState, InjectedState]) -> list[str]:
    return state.get("files", [])


@tool(description=READ_FILE_DESCRIPTION, parse_docstring=True)
def read_file(
        filename: str,
        state: Annotated[ResearchState, InjectedState],
        offset: int = 0,
        limit: int = 2000,
) -> str:
    """从文件系统阅读文件内容，可选行偏移和行数限制

    Args:
        filename: 要阅读的文件名
        state: agent 状态
        offset: 从哪行开始读，默认0
        limit: 最多读多少行，默认2000

    Returns:
        带行号的文件内容，如果有错误的话返回错误信息
    """
    files = state.get("files", [])
    if filename not in files:
        return f"Error: File '{filename}' not found."

    file_path = os.path.join(STORE_DIR, filename)
    if not os.path.exists(file_path):
        return f"Error: File '{filename}' does not exist on disk."

    with open(file_path, "r") as f:
        content = f.read()
    if not content:
        return "System reminder: File exists but has empty content."

    lines = content.splitlines()
    start_index = offset
    end_index = min(len(lines), start_index + limit)
    if start_index > len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    result_lines = []
    for i in range(start_index, end_index):
        line_content = lines[i][:2000]  # 避免行太长了
        result_lines.append(f'{i + 1: 6d}\t{line_content}')
    return "\n".join(result_lines)


@tool(description=WRITE_FILE_DESCRIPTION, parse_docstring=True)
def write_file(
        file_path: str,
        content: str,
        state: Annotated[ResearchState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """将内容写到指定文件路径中

    Args:
        file_path: 写入文件路径
        content: 写入内容
        state: agent 状态
        tool_call_id: 工具调用标识

    Returns:
        更新 agent 状态
    """
    return Command(update={
        "files": [(file_path, content)],
        "messages": [
            ToolMessage(f"Updated file '{file_path}'", tool_call_id=tool_call_id),
        ]
    })
