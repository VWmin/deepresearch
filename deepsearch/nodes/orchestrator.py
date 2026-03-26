from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal
from deepsearch.utils.llm import create_llm_from_env
from deepsearch.utils.state import ResearchState

class RequestClassification(BaseModel):
    """分类用户的请求"""
    request_type: Literal["simple", "research"] = Field(
        description="Type of request: 'simple' for straightforward questions, 'research' for complex topics"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        description="Brief explanation for the classification"
    )

# Prompt for request classification
CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a request classifier for a research assistant system.
Classify user requests as either:
- **simple**: Direct questions that can be answered from general knowledge (e.g., "What is Python?", "Who is the president of USA?")
- **research**: Complex topics requiring investigation, multiple sources, or in-depth analysis (e.g., "Analyze the impact of AI on healthcare", "Research recent developments in quantum computing")

Consider a request as 'research' if it:
- Requires gathering information from multiple sources
- Involves analysis or synthesis of information
- Is about current events or specialized topics
- Requires comparison or evaluation of multiple options

Respond in JSON format with: request_type, confidence, reasoning"""),
    ("human", "{input}")
])


def orchestrator(state: ResearchState) -> ResearchState:
    """对用户请求进行分类，决定是简单问题直接回答还是复杂问题开始研究"""
    llm = create_llm_from_env(temperature=0)
    structured_llm = llm.with_structured_output(RequestClassification)
    chain = CLASSIFICATION_PROMPT | structured_llm
    result = chain.invoke({"input": state["input"]})
    next_action = "simple_answer" if result.request_type == "simple" else "clarify"

    state.update({
        "request_type": result.request_type,
        "next_action": next_action, # type: ignore
        "messages": state["messages"] + [{
            "role": "system", "content": f"User request is classified as {result.request_type}",
        }],
    })
    return state

