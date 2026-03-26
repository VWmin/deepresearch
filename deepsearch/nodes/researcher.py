from deepsearch.agents import agent
from deepsearch.utils.state import ResearchState


def researcher(state: ResearchState) -> ResearchState:
    result = agent.invoke(state)

    state.update(**result)

    return state