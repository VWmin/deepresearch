from langchain_core.prompts import ChatPromptTemplate

from deepsearch.utils.llm import create_llm_from_env
from deepsearch.utils.state import ResearchState



SIMPLE_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question concisely and accurately."),
    ("human", "{input}")
])


def simple_answer(state: ResearchState) -> ResearchState:
    """回答简单问题"""
    llm = create_llm_from_env(temperature=0.3)
    chain = SIMPLE_ANSWER_PROMPT | llm
    result = chain.invoke({"input": state["input"]})

    state.update({
        "output": result.content,
        "next_action": "end", # type: ignore
        "messages": state["messages"] + [{
            "role": "assistant", "content": result.content
        }]
    })
    return state