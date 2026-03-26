from typing import Any, NotRequired

from langchain.agents import AgentState
from langgraph.graph import add_messages
from typing_extensions import TypedDict, Literal, Annotated

from deepsearch.utils.files import file_reducer


class Todo(TypedDict):
    content: str
    status: Literal["pending", "processing", "done"]


class ResearchState(AgentState):
    """研究助手状态，扩展 AgentState 添加自定义字段"""

    # 自定义可选字段
    input: NotRequired[str]
    output: NotRequired[str]

    request_type: NotRequired[Literal["simple", "research"]]
    topic: NotRequired[str]
    needs_clarification: NotRequired[bool]
    clarification_questions: NotRequired[list[str]]
    todos: NotRequired[list[Todo]]
    files: NotRequired[Annotated[list[str], file_reducer]]
    web_resources: NotRequired[str]
    next_action: NotRequired[Literal["simple_answer", "clarify", "end"]]
