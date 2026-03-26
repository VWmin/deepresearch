from deepsearch.subgraph import clarifier as subgraph_clarifier
from deepsearch.utils.state import ResearchState


def clarifier(state: ResearchState) -> ResearchState:
    """用最多三轮，来澄清用户要研究的主题"""
    config = state.get("configurable", {})
    topic = subgraph_clarifier.clarify.invoke(state["input"], config)

    state.update({
        "topic": topic,
        "next_action": "end"
    })
    return state