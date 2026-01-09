"""
Advanced LangGraph Writing Assistant with Multi-Agent Workflow

Features:
- Error handling with retries
- Streaming output
- Three agents: Outliner → Writer → Editor
- Typed state with Pydantic
- Configurable LLM settings
- Progress callbacks
"""

import os
import sys
from typing import Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError


load_dotenv()

# =============================================================================
# Clear out previous run results/errors from terminal
# =============================================================================

os.system("cls" if os.name == "nt" else "clear")


# =============================================================================
# State Definition (using Pydantic for validation)
# =============================================================================

class WritingState(BaseModel):
    """State that flows through the writing pipeline."""
    topic: str = Field(description="The topic to write about")
    outline: str = Field(default="", description="Generated outline")
    draft: str = Field(default="", description="First draft from writer")
    final_draft: str = Field(default="", description="Edited final version")
    revision_notes: str = Field(default="", description="Editor's revision notes")
    error: str | None = Field(default=None, description="Error message if something failed")
    
    class Config:
        # Allow mutation for state updates
        frozen = False


# =============================================================================
# LLM Configuration
# =============================================================================

def create_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    streaming: bool = True
) -> ChatOpenAI:
    """Create a configured LLM instance."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
    )


# Default LLM - can be overridden
llm = create_llm()


# =============================================================================
# Retry Decorator for API Resilience
# =============================================================================

def with_retries(func):
    """Decorator that adds retry logic for transient API failures."""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError)),
        before_sleep=lambda retry_state: print(
            f"  ⚠️  API error, retrying in {retry_state.next_action.sleep} seconds..."
        )
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# =============================================================================
# Agent Definitions
# =============================================================================

@with_retries
def outline_agent(state: WritingState) -> dict:
    """
    Creates a structured outline from the topic.
    
    Returns only the fields being updated (LangGraph merges them).
    """
    print("\n📋 OUTLINE AGENT")
    print(f"   Topic: {state.topic}")
    print("   Working...", end="", flush=True)
    
    try:
        messages = [
            SystemMessage(content="""You are an expert outline creator. 
Create a clear, hierarchical outline with:
- A compelling title
- 3-5 main sections with Roman numerals
- 2-3 subsections under each main section
- Brief notes on key points to cover

Keep the outline focused and well-organized."""),
            HumanMessage(content=f"Create a detailed outline for: {state.topic}")
        ]
        
        # Stream the response
        full_response = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                print(".", end="", flush=True)
        
        print(" Done!")
        print(f"\n   📄 Outline Preview: {full_response[:100]}...")
        
        return {"outline": full_response}
    
    except Exception as e:
        print(f" Failed: {e}")
        return {"error": f"Outline agent failed: {str(e)}"}


@with_retries
def writing_agent(state: WritingState) -> dict:
    """
    Writes a first draft based on the outline.
    """
    # Skip if previous agent errored
    if state.error:
        return {}
    
    print("\n✍️  WRITING AGENT")
    print(f"   Using outline ({len(state.outline)} chars)")
    print("   Writing...", end="", flush=True)
    
    try:
        messages = [
            SystemMessage(content="""You are an expert writer. 
Write a well-structured article based on the provided outline.

Guidelines:
- Follow the outline structure closely
- Use clear, engaging prose
- Include smooth transitions between sections
- Aim for ~500-800 words
- Write in a professional but accessible tone"""),
            HumanMessage(content=f"Write an article based on this outline:\n\n{state.outline}")
        ]
        
        full_response = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                print(".", end="", flush=True)
        
        print(" Done!")
        print(f"   📝 Draft length: {len(full_response.split())} words")
        
        return {"draft": full_response}
    
    except Exception as e:
        print(f" Failed: {e}")
        return {"error": f"Writing agent failed: {str(e)}"}


@with_retries  
def editor_agent(state: WritingState) -> dict:
    """
    Reviews and improves the draft.
    """
    if state.error:
        return {}
    
    print("\n🔍 EDITOR AGENT")
    print(f"   Reviewing draft ({len(state.draft.split())} words)")
    print("   Editing...", end="", flush=True)
    
    try:
        messages = [
            SystemMessage(content="""You are an expert editor. 
Review and improve the draft article.

Your tasks:
1. Fix any grammatical or spelling errors
2. Improve clarity and flow
3. Strengthen weak sentences
4. Ensure consistent tone
5. Add a brief "Revision Notes" section at the end listing your key changes

Return the improved article followed by your revision notes."""),
            HumanMessage(content=f"Edit and improve this draft:\n\n{state.draft}")
        ]
        
        full_response = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                print(".", end="", flush=True)
        
        print(" Done!")
        
        # Try to separate revision notes if present
        if "Revision Notes" in full_response:
            parts = full_response.split("Revision Notes", 1)
            final_draft = parts[0].strip()
            revision_notes = "Revision Notes" + parts[1] if len(parts) > 1 else ""
        else:
            final_draft = full_response
            revision_notes = ""
        
        return {
            "final_draft": final_draft,
            "revision_notes": revision_notes
        }
    
    except Exception as e:
        print(f" Failed: {e}")
        return {"error": f"Editor agent failed: {str(e)}"}


# =============================================================================
# Conditional Edge (example: skip editor if draft is short)
# =============================================================================

def should_edit(state: WritingState) -> str:
    """Decide whether to run the editor or go straight to END."""
    if state.error:
        return "end"
    
    # Example condition: skip editing for very short drafts
    word_count = len(state.draft.split())
    if word_count < 50:
        print("\n⏭️  Skipping editor (draft too short)")
        return "end"
    
    return "editor"


# =============================================================================
# Build the Graph
# =============================================================================

def create_workflow() -> StateGraph:
    """Build and compile the writing workflow graph."""
    
    # Create graph with Pydantic state
    workflow = StateGraph(WritingState)
    
    # Add nodes
    workflow.add_node("outline", outline_agent)
    workflow.add_node("writer", writing_agent)
    workflow.add_node("editor", editor_agent)
    
    # Add edges
    workflow.add_edge(START, "outline")
    workflow.add_edge("outline", "writer")
    
    # Conditional edge: writer → editor OR end
    workflow.add_conditional_edges(
        "writer",
        should_edit,
        {
            "editor": "editor",
            "end": END
        }
    )
    
    workflow.add_edge("editor", END)
    
    return workflow.compile()


# =============================================================================
# Main Execution
# =============================================================================

def run_writing_assistant(
    topic: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> dict:
    """
    Run the complete writing assistant pipeline.
    
    Args:
        topic: The topic to write about
        model: OpenAI model to use
        temperature: Creativity level (0-1)
    
    Returns:
        Dict with all generated content (topic, outline, draft, final_draft, etc.)
    """
    global llm
    llm = create_llm(model=model, temperature=temperature)
    
    print("=" * 60)
    print("🚀 WRITING ASSISTANT STARTING")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Topic: {topic}")
    
    # Create initial state
    initial_state = WritingState(topic=topic)
    
    # Build and run workflow
    app = create_workflow()
    
    # Run workflow
    final_state = app.invoke(initial_state)
    
    return final_state


def main():
    """CLI entry point."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("   Set it in a .env file or export it directly")
        sys.exit(1)
    
    # Default topic or from command line
    topic = (
        " ".join(sys.argv[1:]) 
        if len(sys.argv) > 1 
        else "Being an older man dating younger women in today's society"
    )
    
    try:
        result = run_writing_assistant(
            topic=topic,
            model="gpt-4o-mini",  # Use "gpt-4o" for better quality
            temperature=0.7
        )
        
        # Output results
        print("\n" + "=" * 60)
        print("✅ COMPLETE!")
        print("=" * 60)
        
        if result.get("error"):
            print(f"\n❌ Error occurred: {result['error']}")
        else:
            print("\n📄 FINAL ARTICLE:")
            print("-" * 40)
            print(result.get("final_draft") or result.get("draft", ""))
            
            if result.get("revision_notes"):
                print("\n" + "-" * 40)
                print(result["revision_notes"])
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()