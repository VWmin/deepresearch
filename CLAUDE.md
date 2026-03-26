# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research assistant system built with LangGraph that processes user requests through a state machine workflow:
1. Classifies requests as "simple" (direct answer) or "research" (multi-step investigation)
2. Clarifies vague research topics through up to 3 interactive questions
3. Plans and executes research steps
4. Generates research reports

**Current implementation status**: orchestrator → simple_answer/clarifier → END. The plan/researcher/reporter nodes are planned but not yet implemented.

## Architecture

```
orchestrator → clarifier → plan → researcher → reporter
     ↓
   simple_answer (for simple questions)
```

**Two LangGraph API patterns are used:**
- `main.py` uses `StateGraph`: nodes are functions that take/return state, edges defined explicitly with `add_edge()` and `add_conditional_edges()`
- `subgraph/clarifier.py` uses functional API (`@task`, `@entrypoint`): tasks return `.result()` to get values, uses Python `input()` for user interaction

**Key files:**
- `deepsearch/utils/state.py` - `ResearchState` inherits from `AgentState` (from `langchain.agents`); `messages: Annotated[list, add_messages]` inherited, `files: Annotated[list, file_reducer]` for state aggregation
- `deepsearch/utils/files.py` - Custom `file_reducer` that stores files to `store/` directory and tracks filenames in state
- `deepsearch/utils/llm.py` - `create_llm_from_env()` uses `init_chat_model(model="deepseek:deepseek-chat")`
- `deepsearch/nodes/*.py` - Node functions receive `ResearchState`, return modified state via `state.update()`
- `deepsearch/agents/__init__.py` - Main agent using `langchain.agents.create_agent` with sub-agent delegation pattern
- `deepsearch/agents/research.py` - `tavily_search` tool with `InjectedState` and `Command` for state updates
- `deepsearch/agents/planer.py` - `write_todos`/`read_todos` tools for task tracking
- `deepsearch/agents/files.py` - Virtual filesystem tools (`ls`, `read_file`, `write_file`)

**Agent pattern (langchain.agents.create_agent):**
```python
from langchain.agents import create_agent, AgentState

# ResearchState must inherit from AgentState for create_agent compatibility
class ResearchState(AgentState):
    files: NotRequired[Annotated[list[str], file_reducer]]

agent = create_agent(model, tools, state_schema=ResearchState, system_prompt=INSTRUCTIONS)
```

**Tool pattern with state injection:**
```python
@tool
def my_tool(
    arg: str,
    state: Annotated[ResearchState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    return Command(update={"field": value, "messages": [ToolMessage(..., tool_call_id=tool_call_id)]})
```

**Structured output pattern:**
```python
structured_llm = llm.with_structured_output(PydanticModel)
chain = PROMPT | structured_llm
result = chain.invoke({"input": state["input"]})
```

**Routing:**
- `next_action` field values: `"simple_answer"`, `"clarify"`, `"end"`
- Currently both `simple_answer` and `clarifier` nodes route directly to END

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# .env file required:
# DEEPSEEK_API_KEY=your_api_key
# TAVILY_API_KEY=your_api_key  # for web search
```

## Running

```bash
python -m deepsearch.main          # Main agent test (test_agent())
python -m deepsearch.subgraph.clarifier  # Clarifier subgraph (interactive, uses input())
```

## Dependencies

- `langgraph` - State machine workflow with `StateGraph` and functional API (`@task`, `@entrypoint`)
- `langchain` / `langchain-core` - Core LangChain utilities, `init_chat_model`, tools
- `langchain.agents.create_agent` - Agent creation (replaces deprecated `langgraph.prebuilt.create_react_agent`)
- `langchain-deepseek` - DeepSeek LLM integration
- `tavily-python` - Web search tool
- `python-dotenv` - Loads `.env` with `load_dotenv(override=True)`
- `httpx` - HTTP client for fetching web content (use `follow_redirects=True`)
- `markdownify` - HTML to Markdown conversion

## File Storage

Research materials are stored in `store/` directory via the `file_reducer` function. Each `tavily_search` result creates a markdown file with URL, query, summary, and raw content.