from langchain_core.messages import HumanMessage

from deepsearch.agents import agent
from deepsearch.utils.state import ResearchState


def researcher(state: ResearchState) -> ResearchState:
    messages = state.get("messages", [])
    clarified_topic = state.get("topic", "Research topic is None.")
    messages.append(HumanMessage(content=clarified_topic))

    result = agent.invoke(state)
    state.update(**result)

    return state