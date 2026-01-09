import os
from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# State definition
class AgentState(TypedDict):
    topic: str
    outline: str
    final_draft: str

# Initialize LLM (fix model name - "gpt-4o-mini" is the correct format)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Define the agents
def outline_agent(state: AgentState) -> dict:
    """Takes a topic, creates an outline"""
    print(f"Outline agent working on: {state['topic']}")
    
    messages = [
        SystemMessage(content="You are an expert outline creator. Create a detailed outline for the given topic."),
        HumanMessage(content=f"Create a detailed outline for:\n\n{state['topic']}")
    ]
    
    # Use .invoke() instead of calling llm directly
    response = llm.invoke(messages)
    
    print(f"\nOutline created:\n{response.content}\n{'='*50}\n")
    
    # Return only the fields being updated
    return {"outline": response.content}


def writing_agent(state: AgentState) -> dict:
    """Takes an outline, writes a final draft"""
    print(f"Writing agent working on outline...")
    
    messages = [
        SystemMessage(content="You are an expert writer. Write a detailed article based on the given outline."),
        HumanMessage(content=f"Write a detailed article based on this outline:\n\n{state['outline']}")
    ]
    
    response = llm.invoke(messages)
    
    return {"final_draft": response.content}


# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("outline_agent", outline_agent)
workflow.add_node("writing_agent", writing_agent)

workflow.add_edge(START, "outline_agent")
workflow.add_edge("outline_agent", "writing_agent")
workflow.add_edge("writing_agent", END)

app = workflow.compile()


if __name__ == "__main__":
    initial_state: AgentState = {
        "topic": "The impact of artificial intelligence on modern education",
        "outline": "",
        "final_draft": ""
    }

    final_state = app.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL DRAFT:")
    print("="*50 + "\n")
    print(final_state["final_draft"])