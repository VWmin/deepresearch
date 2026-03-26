from typing import TypedDict, Literal, Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deepsearch.utils.state import ResearchState, Todo

WRITE_TODOS_DESCRIPTION = """Create and manage structured task lists for tracking progress through complex workflows.

## When to Use
- Multi-step or non-trivial tasks requiring coordination
- When user provides multiple tasks or explicitly requests todo list  
- Avoid for single, trivial actions unless directed otherwise

## Structure
- Maintain one list containing multiple todo objects (content, status, id)
- Use clear, actionable content descriptions
- Status must be: pending, in_progress, or completed

## Best Practices  
- Only one in_progress task at a time
- Mark completed immediately when task is fully done
- Always send the full updated list when making changes
- Prune irrelevant items to keep list focused

## Progress Updates
- Call TodoWrite again to change task status or edit content
- Reflect real-time progress; don't batch completions  
- If blocked, keep in_progress and add new task describing blocker

## Parameters
- todos: List of TODO items with content and status fields

## Returns
Updates agent state with new todo list."""

TODO_USAGE_INSTRUCTIONS = """Based upon the user's request:
1. Use the write_todos tool to create TODO at the start of a user request, per the tool description.
2. After you accomplish a TODO, use the read_todos to read the TODOs in order to remind yourself of the plan. 
3. Reflect on what you've done and the TODO.
4. Mark you task as completed, and proceed to the next TODO.
5. Continue this process until you have completed all TODOs.

IMPORTANT: Always create a research plan of TODOs and conduct research following the above guidelines for ANY user request.
IMPORTANT: Aim to batch research tasks into a *single TODO* in order to minimize the number of TODOs you have to keep track of.
"""

@tool(description=WRITE_TODOS_DESCRIPTION, parse_docstring=True)
def write_todos(
        todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """ 创建或更新 agent 的待办列表

    Args:
        todos: 待办项列表
        tool_call_id: 工具调用的标识

    Returns:
        更新 agent 状态
    """
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f'Updated todo list to {todos}', tool_call_id=tool_call_id),
            ]
        }
    )


@tool(parse_docstring=True)
def read_todos(
        state: Annotated[ResearchState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
):
    """从 agent 状态中读取当前的待办列表，从而保持对剩余任务的追踪

    Args:
        state: 包含待办列表的 agent 状态
        tool_call_id: 工具调用的标识

    Returns:
        对于当前待办列表的 str 格式化输出
    """
    todos = state.get("todos", [])
    if not todos:
        return "No todos currently in the list."

    result = "Current TODO List:\n"
    for i, todo in enumerate(todos, 1):
        status_emoji = {"pending": "⏳", "processing": "🔄", "done": "✅"}
        emoji = status_emoji.get(todo["status"], "❓")
        result += f"{i}. {emoji} {todo['content']} ({todo['status']})\n"

    return result.strip()
