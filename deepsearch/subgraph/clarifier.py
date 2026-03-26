from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langgraph.func import task, entrypoint
from pydantic import BaseModel, Field

from deepsearch.utils.llm import create_llm_from_env


class NeedClassification(BaseModel):
    """判定是否需要澄清"""
    need_clarification: Literal["yes", "no"] = Field(description="Does user input need clarification")


class ClarificationQuestion(BaseModel):
    """用提问澄清用户的主题"""
    question: str = Field(description="clarification question")
    reasoning: str = Field(description="Explanation of why this question will help clarify the research topic")


class ClarifiedTopic(BaseModel):
    """澄清后的主题"""
    topic: str = Field(description="clarified topic")


NEED_CLARIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant helping clarify research topics.

Task:
- Determine whether the user's research topic is too vague or broad and needs clarification.
- Respond with **exactly one word**: `yes` if clarification is needed, `no` if not.

Consider a topic vague if:
- It lacks specifics about scope (time period, geography, industry)
- It does not indicate purpose (academic, business, personal)
- It is very general or broad

Examples:
- User input: "Tell me about AI" → yes
- User input: "Analyze the market size of US cloud computing in 2023" → no
- User input: "Research the market" → yes"""),
    ("human", "{input}")
])

# Prompt for generating clarification questions
CLARIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant clarifying vague research topics.

Rules:
1. Ask **only one clarification question at a time**.
2. Wait for the user's answer before asking the next question.
3. Ask at most **three questions** in total per topic.
4. Your goal is to:
   - Narrow the scope (time period, geography, industry)
   - Clarify the purpose (academic, business, personal)
   - Identify specific aspects to focus on
   - Determine the desired output format

Examples:

Vague topic: "Tell me about AI"
- Round 1: "What specific aspect of AI interests you?"
- Round 2 (after user answers): "Are you looking for technical details or business applications?"
- Round 3 (after user answers): "What time period should we focus on?"

Vague topic: "Research the market"
- Round 1: "Which industry or market segment?"
- Round 2 (after user answers): "What geographic region?"
- Round 3 (after user answers): "What specific information do you need (size, trends, competitors)?"
"""),
    ("human", """Research Topic: {topic}

Current Context:
{context}

Generate up to 3 clarification questions to better understand this research topic.""")
])

CLARIFIED_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant.

Task:
- The user has provided an initial research request: {user_request}
- You also have some clarification questions and the user's answers, contained in the following context:
{context}

Your goal:
- Combine the original request and the user's answers to produce a **single, concise, clarified research topic**.
- Only output the clarified topic as a string. Do not include explanations, notes, or extra text.

Example:

user_request: "Tell me about AI"
context: "Q1: What specific aspect of AI interests you? A1: Natural language processing
          Q2: Are you looking for technical details or business applications? A2: Technical details
          Q3: What time period should we focus on? A3: 2020-2023"

Clarified topic: "Natural language processing in AI: technical details from 2020-2023"
"""),
    ("human", "What is the clarified topic?")
])


@task
def need_clarification(user_request: str) -> bool:
    llm = create_llm_from_env(temperature=0.3)
    structured_llm = llm.with_structured_output(NeedClassification)
    chain = NEED_CLARIFICATION_PROMPT | structured_llm
    result = chain.invoke({"input": user_request})
    return result.need_clarification == "yes"


@task
def get_clarification_question(user_request: str, context: str):
    llm = create_llm_from_env(temperature=0.3)
    structured_llm = llm.with_structured_output(ClarificationQuestion)
    chain = CLARIFY_PROMPT | structured_llm
    result = chain.invoke({"topic": user_request, "context": context})
    return result.question


@task
def clarified_topic(user_request: str, context: str):
    llm = create_llm_from_env(temperature=0.3)
    structured_llm = llm.with_structured_output(ClarifiedTopic)
    chain = CLARIFIED_PROMPT | structured_llm
    result = chain.invoke({"user_request": user_request, "context": context})
    return result.topic


@entrypoint()
def clarify(user_request: str) -> str:
    need = need_clarification(user_request).result()
    if need:
        context, counter = "", 0
        while True:
            question = get_clarification_question(user_request, context).result()
            counter += 1
            if question == "" or question is None or counter > 3:
                break
            answer = input(f"{question}\n> ")
            context += f"{question}\n{answer}\n\n"
        topic = clarified_topic(user_request, context).result()
    else:
        topic = user_request

    return topic
