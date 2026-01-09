"""
Minimal LangGraph Example - Bare Bones Version

This is the simplest possible working example.
See writing_assistant.py for the full-featured version.
"""
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# lets start clean in the terminal, clear previous runs/errors
os.system("cls" if os.name == "nt" else "clear")

load_dotenv()


class State(TypedDict):
    topic: str
    outline: str
    article: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def outline_agent(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content="Create a brief outline for the given topic."),
        HumanMessage(content=state["topic"])
    ])
    return {"outline": response.content}


def writer_agent(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content="Write a short article based on this outline."),
        HumanMessage(content=state["outline"])
    ])
    return {"article": response.content}


# Build graph
graph = StateGraph(State)
graph.add_node("outline", outline_agent)
graph.add_node("writer", writer_agent)
graph.add_edge(START, "outline")
graph.add_edge("outline", "writer")
graph.add_edge("writer", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "topic": "Why dogs cats make great pets, and the difference between them.",
        "outline": "",
        "article": ""
    })
    print(result["article"])
