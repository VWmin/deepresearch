import base64
import os
import uuid
from datetime import datetime
from typing import Literal, Annotated

import httpx
from langchain_core.tools import tool, InjectedToolCallId, InjectedToolArg
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from markdownify import markdownify
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from tavily import TavilyClient

from deepsearch.utils.llm import create_llm_from_env
from deepsearch.utils.state import ResearchState

tavily_client = TavilyClient()

SUMMARIZE_WEB_SEARCH = """You are creating a minimal summary for research steering - your goal is to help an agent know what information it has collected, NOT to preserve all details.

<webpage_content>
{webpage_content}
</webpage_content>

Create a VERY CONCISE summary focusing on:
1. Main topic/subject in 1-2 sentences
2. Key information type (facts, tutorial, news, analysis, etc.)  
3. Most significant 1-2 findings or points

Keep the summary under 150 words total. The agent needs to know what's in this file to decide if it should search for more information or use this source.

Generate a descriptive filename that indicates the content type and topic (e.g., "mcp_protocol_overview.md", "ai_safety_research_2024.md").

Output format:
```json
{{
   "filename": "descriptive_filename.md",
   "summary": "Very brief summary under 150 words focusing on main topic and key findings"
}}
```

Today's date: {date}
"""

FILE_CONTENT_FORMAT = """# Search Result: {title}

**URL:** {url}
**Query:** {query}
**Date:** {date}

## Summary
{summary}

## Raw Content
{raw_content}
"""

THINK_TOOL_DESCRIPTION = """Tool for strategic reflection on research progress and decision-making.

Use this tool after each search to analyze results and plan next steps systematically.
This creates a deliberate pause in the research workflow for quality decision-making.

When to use:
- After receiving search results: What key information did I find?
- Before deciding next steps: Do I have enough to answer comprehensively?
- When assessing research gaps: What specific information am I still missing?
- Before concluding research: Can I provide a complete answer now?
- How complex is the question: Have I reached the number of search limits?

Reflection should address:
1. Analysis of current findings - What concrete information have I gathered?
2. Gap assessment - What crucial information is still missing?
3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
4. Strategic decision - Should I continue searching or provide my answer?

Args:
    reflection: Your detailed reflection on research progress, findings, gaps, and next steps

Returns:
    Confirmation that reflection was recorded for decision-making
"""

RESEARCHER_INSTRUCTIONS = """You are a research assistant conducting research on the user's input topic. For context, today's date is {date}.

<Task>
Your job is to use tools to gather information about the user's input topic.
You can use any of the tools provided to you to find resources that can help answer the research question. You can call these tools in series or in parallel, your research is conducted in a tool-calling loop.
</Task>

<Available Tools>
You have access to two main tools:
1. **tavily_search**: For conducting web searches to gather information
2. **think_tool**: For reflection and strategic planning during research

**CRITICAL: Use think_tool after each search to reflect on results and plan next steps**
</Available Tools>

<Instructions>
Think like a human researcher with limited time. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Start with broader searches** - Use broad, comprehensive queries first
3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
4. **Execute narrower searches as you gather information** - Fill in the gaps
5. **Stop when you can answer confidently** - Don't keep searching for perfection
</Instructions>

<Hard Limits>
**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: Use 1-2 search tool calls maximum
- **Normal queries**: Use 2-3 search tool calls maximum
- **Very Complex queries**: Use up to 5 search tool calls maximum
- **Always stop**: After 5 search tool calls if you cannot find the right sources

**Stop Immediately When**:
- You can answer the user's question comprehensively
- You have 3+ relevant examples/sources for the question
- Your last 2 searches returned similar information
</Hard Limits>

<Show Your Thinking>
After each search tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I search more or provide my answer?
</Show Your Thinking>
"""


class Summary(BaseModel):
    """Schema for webpage content summarization"""
    filename: str = Field(description="Name of the file to store.")
    summary: str = Field(description="Key learnings from the webpage.")


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %d, %Y")


def run_tavily_search(
        search_query: str,
        max_results: int = 1,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True,
) -> dict:
    result = tavily_client.search(
        search_query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )
    return result


def summarize_webpage_content(webpage_content: str) -> Summary:
    """Summarize webpage content using the configured summarization model."""
    try:
        model = create_llm_from_env()
        structured_model = model.with_structured_output(Summary)

        summary = structured_model.invoke([
            HumanMessage(content=SUMMARIZE_WEB_SEARCH.format(
                webpage_content=webpage_content,
                date=get_today_str(),
            ))
        ])

        return summary

    except Exception as e:
        print(e)
        # fallback
        return Summary(
            filename="search_result.md",
            summary=webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content,
        )


def process_search_results(results: dict) -> list[dict]:
    """解析 tavily 的搜索结果，生成页面总结并存储"""
    processed_results = []

    # 使用上下文管理器确保资源正确释放
    with httpx.Client(timeout=30.0, follow_redirects=True) as httpx_client:
        for result in results.get('results', []):

            # Get url
            url = result['url']

            # Read url with timeout and error handling
            try:
                response = httpx_client.get(url)

                if response.status_code == 200:
                    # Convert HTML to Markdown
                    raw_content = markdownify(response.text)
                    summary_obj = summarize_webpage_content(raw_content)
                else:
                    # Use Tavily's generated summary
                    raw_content = result.get('raw_content', '')
                    summary_obj = Summary(
                        filename="URL_error.md",
                        summary=result.get('content', 'Error reading URL; try another search.')
                    )
            except (httpx.TimeoutException, httpx.RequestError) as e:
                # Handle timeout or connection errors gracefully
                raw_content = result.get('raw_content', '')
                summary_obj = Summary(
                    filename="connection_error.md",
                    summary=result.get('content', 'Could not fetch URL (timeout/connection error). Try another search.')
                )

            # uniquify file names
            uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
            name, ext = os.path.splitext(summary_obj.filename)
            summary_obj.filename = f"{name}_{uid}{ext}"

            processed_results.append({
                'url': result['url'],
                'title': result['title'],
                'summary': summary_obj.summary,
                'filename': summary_obj.filename,
                'raw_content': raw_content,
            })

    return processed_results


@tool(parse_docstring=True)
def tavily_search(
        query: str,
        state: Annotated[ResearchState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        max_results: Annotated[int, InjectedToolArg],
        topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> Command:
    """搜索web内容并存储详细结果到文件，state中存储最小化的上下文

    Args:
        query: 要执行的搜索询问
        state: agent 状态
        tool_call_id: 工具调用标识
        max_results: 最多返回多少搜索结果，默认1
        topic: 搜索问题的分类 - 'general', 'news', 或者 'finance' (默认: 'general')
    """
    search_results = run_tavily_search(query, max_results, topic)
    processed_results = process_search_results(search_results)

    files = state.get("files", [])
    saved_files = []
    summaries = []

    for i, result in enumerate(processed_results):
        filename = result['filename']
        file_content = FILE_CONTENT_FORMAT.format(
            title=result['title'],
            url=result['url'],
            query=query,
            date=get_today_str(),
            summary=result['summary'],
            raw_content=result['raw_content'] if result['raw_content'] else 'No raw content available.',
        )
        files.append((filename, file_content))
        saved_files.append(filename)
        summaries.append(f"- {filename}: {result['summary']}...")

    # 返回简化过的信息给 调用方，详细的内容存起来
    summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':

{chr(10).join(summaries)}

Files: {', '.join(saved_files)}
💡 Use read_file() to access full details when needed."""

    return Command(update={
        "files": files,
        "messages": [ToolMessage(summary_text, tool_call_id=tool_call_id)],
    })


@tool(description=THINK_TOOL_DESCRIPTION)
def think_tool(reflection: str) -> str:
    return f"Reflection recorded: {reflection}"
