from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from deepsearch.nodes.clarifier import clarifier
from deepsearch.nodes.orchestrator import orchestrator
from deepsearch.nodes.simple_answer import simple_answer
from deepsearch.utils.state import ResearchState


def route_by_action(state: ResearchState) -> str:
    """根据 next_action 决定下一个节点"""
    return state.get("next_action", "end")


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("orchestrator", orchestrator)
    graph.add_node("simple_answer", simple_answer)
    graph.add_node("clarify", clarifier)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_by_action,
        {
            "simple_answer": "simple_answer",
            "clarify": "clarify",
        }
    )
    graph.add_edge("simple_answer", END)
    graph.add_edge("clarify", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    return app


def test_graph():
    app = build_graph()

    init_state = {"input": "AI发展现状"}
    config = {"configurable": {"thread_id": 1}}
    result = app.invoke(init_state, config)

    print(result)


def test_agent():
    from deepsearch.agents import agent
    from deepsearch.utils.formatter import format_messages

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Give me an overview of Model Context Protocol (MCP).",
        }],
    })
    format_messages(result["messages"])


if __name__ == '__main__':
    # test_graph()
    test_agent()
