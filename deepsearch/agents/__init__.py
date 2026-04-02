from typing import TypedDict, NotRequired, Annotated

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.errors import GraphRecursionError

from deepsearch.agents.files import write_file, read_file, ls, FILE_USAGE_INSTRUCTIONS
from deepsearch.agents.planer import write_todos, read_todos, TODO_USAGE_INSTRUCTIONS
from deepsearch.agents.research import tavily_search, think_tool, RESEARCHER_INSTRUCTIONS, get_today_str
from deepsearch.utils.llm import create_llm_from_env
from deepsearch.utils.state import ResearchState

TASK_DESCRIPTION_PREFIX = """Delegate a task to a specialized sub-agent with isolated context. Available agents for delegation are:
{other_agents}
"""

SUBAGENT_USAGE_INSTRUCTIONS = """You can delegate tasks to sub-agents.

<Task>
Your role is to coordinate research by delegating specific research tasks to sub-agents.
</Task>

<Available Tools>
1. **task(description, subagent_type)**: Delegate research tasks to specialized sub-agents
   - description: Clear, specific research question or task
   - subagent_type: Type of agent to use (e.g., "research-agent")
2. **think_tool(reflection)**: Reflect on the results of each delegated task and plan next steps.
   - reflection: Your detailed reflection on the results of the task and next steps.

**PARALLEL RESEARCH**: When you identify multiple independent research directions, make multiple **task** tool calls in a single response to enable parallel execution. Use at most {max_concurrent_research_units} parallel agents per iteration.
</Available Tools>

<Hard Limits>
**Task Delegation Budgets** (Prevent excessive delegation):
- **Bias towards focused research** - Use single agent for simple questions, multiple only when clearly beneficial or when you have multiple independent research directions based on the user's request.
- **Stop when adequate** - Don't over-research; stop when you have sufficient information
- **Limit iterations** - Stop after {max_researcher_iterations} task delegations if you haven't found adequate sources
</Hard Limits>

<Scaling Rules>
**Simple fact-finding, lists, and rankings** can use a single sub-agent:
- *Example*: "List the top 10 coffee shops in San Francisco" → Use 1 sub-agent, store in `findings_coffee_shops.md`

**Comparisons** can use a sub-agent for each element of the comparison:
- *Example*: "Compare OpenAI vs. Anthropic vs. DeepMind approaches to AI safety" → Use 3 sub-agents
- Store findings in separate files: `findings_openai_safety.md`, `findings_anthropic_safety.md`, `findings_deepmind_safety.md`

**Multi-faceted research** can use parallel agents for different aspects:
- *Example*: "Research renewable energy: costs, environmental impact, and adoption rates" → Use 3 sub-agents
- Organize findings by aspect in separate files

**Important Reminders:**
- Each **task** call creates a dedicated research agent with isolated context
- Sub-agents can't see each other's work - provide complete standalone instructions
- Use clear, specific language - avoid acronyms or abbreviations in task descriptions
</Scaling Rules>

Today's date: {date}
"""


class SubAgent(TypedDict):
    """定制 sub-agent 配置"""

    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]


def _create_task_tool(tools, subagents: list[SubAgent], model, state_schema):
    """创建一个工具（@tool），这个工具的作用是能够将任务分派到不同的subagent中，并且拥有各自隔离的上下文

    Args:
        tools: 总共有哪些可用的工具
        subagents: subagent 配置
        model: 语言模型
        state_schema: 状态

    Returns:
        一个分派任务到subagent的工具
    """
    # tools
    tools_mapping = {}
    for e in tools:
        if not isinstance(e, BaseTool):
            e = tool(e)
        tools_mapping[e.name] = e

    # subagents - 使用 langchain.agents.create_agent
    agents_mapping = {}
    for e in subagents:
        if "tools" in e:
            _tools = [tools_mapping[t] for t in e["tools"]]
        else:
            _tools = list(tools_mapping.values())
        agents_mapping[e['name']] = create_agent(
            model, tools=_tools, state_schema=state_schema, system_prompt=e["prompt"]
        )

    # 描述有哪些 subagents，各自有啥用
    all_subagents_description = [
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ]

    # 分派器
    @tool(description=TASK_DESCRIPTION_PREFIX.format(other_agents=all_subagents_description))
    def task(
            description: str,
            subagent_name: str,
            state: Annotated[ResearchState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command | str:
        """在隔离的上下文中分派任务到某个 sub-agent"""
        if subagent_name not in agents_mapping:
            return (f"Error: invoked agent {subagent_name}, "
                    f"the only allowed agents are {[f'`{n}`' for n in agents_mapping]}")

        subagent = agents_mapping[subagent_name]

        # 创建新的上下文
        state["messages"] = [HumanMessage(content=description)]

        # 使用 stream 模式，逐步收集状态
        current_state = state.copy()
        try:
            for event in subagent.stream(state):
                # event 结构: {"agent": {"messages": [...], ...}} 或 {"tools": {...}}
                for node_name, node_state in event.items():
                    current_state = node_state
            result = current_state
        except GraphRecursionError:
            # 达到递归上限，使用当前上下文生成总结
            llm = model
            summary_prompt = """你的研究任务已达到最大迭代次数限制。
请基于目前收集到的信息，提供一个总结性的回复。
如果已有部分研究结果，请总结并呈现。
如果研究未完成，请说明目前进展和发现。"""

            current_messages = current_state.get("messages", [])
            summary_response = llm.invoke(current_messages + [HumanMessage(content=summary_prompt)])
            current_state["messages"] = current_messages + [summary_response]
            result = current_state

        return Command(update={
            "files": result.get("files", []),  # Merge any file changes
            "messages": [ToolMessage(result["messages"][-1].content, tool_call_id=tool_call_id)],
        })

    return task


_model = create_llm_from_env(temperature=0.0)
# 描述 sub-agent 依赖哪些工具
sub_agent_tools = [tavily_search, think_tool]
# agent 可以使用的工具
built_in_tools = [ls, read_file, write_file, write_todos, read_todos, think_tool]

# 目前 sub-agent 只有 researcher 一个，其主要依赖 tavily 搜索工具和反思工具来进行 web 内容引用和总结
# 使用 sub-agent 的一大目的是隔离上下文
research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    "tools": ["tavily_search", "think_tool"],  # 单独指定某个 sub-agent 需要哪些工具
}

task_tool = _create_task_tool(
    sub_agent_tools, [research_sub_agent], _model, ResearchState
)

max_concurrent_research_units = 3
max_researcher_iterations = 3
SUBAGENT_INSTRUCTIONS = SUBAGENT_USAGE_INSTRUCTIONS.format(
    max_concurrent_research_units=max_concurrent_research_units,
    max_researcher_iterations=max_researcher_iterations,
    date=get_today_str(),
)

INSTRUCTIONS = (
        "# TODO MANAGEMENT\n"
        + TODO_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# FILE SYSTEM USAGE\n"
        + FILE_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# SUB-AGENT DELEGATION\n"
        + SUBAGENT_INSTRUCTIONS
)

agent = create_agent(
    _model,
    built_in_tools + [task_tool],
    state_schema=ResearchState,
    system_prompt=INSTRUCTIONS,
)

# 到这里有一个 agent，和一个 sub-agent
# agent 有工具：[ls, read_file, write_file, write_todos, read_todos, think_tool] + [task_tool]
#   通过 task_tool，agent 知道可以委派研究任务给 sub-agent。
# agent 通过 system-prompt 了解了编写待办的能力，使用文件系统的能力，使用 sub-agent 的能力
