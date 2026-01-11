"""
Advanced LangGraph Writing Assistant with Multi-Agent Workflow

Enhanced Features:
- Error handling with retries
- Streaming output with continuation for long responses
- Four agents: Extra Context → Outliner → Writer → Editor
- Automatic web search via Tavily for current/niche topics
- Typed state with Pydantic
- Configurable LLM settings
- Progress callbacks
- File system I/O for intermediate and final outputs
- Job ID tracking for file organization
- Interactive prompting when run in IDE
- Agents read context from files (not state) for modularity
"""

import os
import sys
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError
from tavily import TavilyClient


load_dotenv()

# =============================================================================
# Clear out previous run results/errors from terminal
# =============================================================================

os.system("cls" if os.name == "nt" else "clear")


# =============================================================================
# Directory Configuration
# =============================================================================

def get_script_directory() -> Path:
    """Get the directory where this script is located."""
    # __file__ gives the path to this script
    # If running interactively or __file__ isn't available, use current working directory
    try:
        return Path(__file__).parent.resolve()
    except NameError:
        return Path.cwd()


llm_model = "gpt-5-mini"

# Directories relative to where the script lives
SCRIPT_DIR = get_script_directory()
WORKING_DIR = SCRIPT_DIR / "wip"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Create directories if they don't exist
WORKING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Script directory: {SCRIPT_DIR}")
print(f"📁 Working directory: {WORKING_DIR}")
print(f"📁 Output directory: {OUTPUT_DIR}")


# =============================================================================
# Job ID Management
# =============================================================================

def generate_job_id() -> str:
    """Generate a unique job ID for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"


# Global job ID - set at runtime
JOB_ID: str = ""


# =============================================================================
# File System Utilities
# =============================================================================

def get_working_filepath(agent_name: str) -> Path:
    """Get the filepath for an agent's output in the working directory."""
    return WORKING_DIR / f"{JOB_ID}_{agent_name}.md"


def get_output_filepath(filename: str) -> Path:
    """Get the filepath for final output."""
    return OUTPUT_DIR / f"{JOB_ID}_{filename}"


def save_to_working(agent_name: str, content: str) -> Path:
    """
    Save content to the working directory.
    
    Args:
        agent_name: Name of the agent (used in filename)
        content: Content to save
        
    Returns:
        Path to the saved file
    """
    filepath = get_working_filepath(agent_name)
    filepath.write_text(content, encoding="utf-8")
    print(f"   💾 Saved to: {filepath}")
    return filepath


def load_from_working(agent_name: str) -> str:
    """
    Load content from the working directory.
    
    Args:
        agent_name: Name of the agent whose output to load
        
    Returns:
        Content from the file, or empty string if not found
    """
    filepath = get_working_filepath(agent_name)
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return ""


def save_final_output(content: str, filename: str = "final_article.md") -> Path:
    """
    Save final output to the output directory.
    
    Args:
        content: Content to save
        filename: Output filename
        
    Returns:
        Path to the saved file
    """
    filepath = get_output_filepath(filename)
    filepath.write_text(content, encoding="utf-8")
    return filepath


def append_to_file(filepath: Path, content: str) -> None:
    """Append content to an existing file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(content)


# =============================================================================
# State Definition (using Pydantic for validation)
# Minimal state - agents read content from files
# =============================================================================

class WritingState(BaseModel):
    """
    State that flows through the writing pipeline.
    
    Content is stored in files rather than state for modularity.
    State tracks job metadata and completion status.
    """
    job_id: str = Field(default="", description="Unique job identifier")
    topic: str = Field(description="The topic to write about")
    context_complete: bool = Field(default=False, description="Whether extra context gathering is complete")
    outline_complete: bool = Field(default=False, description="Whether outline has been generated")
    draft_complete: bool = Field(default=False, description="Whether draft has been written")
    edit_complete: bool = Field(default=False, description="Whether editing is complete")
    error: str | None = Field(default=None, description="Error message if something failed")
    
    class Config:
        # Allow mutation for state updates
        frozen = False


# =============================================================================
# LLM Configuration
# =============================================================================

def create_llm(
    model: str = llm_model,
    temperature: float = 0.7,
    streaming: bool = True,
    max_tokens: int = 4096
) -> ChatOpenAI:
    """Create a configured LLM instance."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
        max_tokens=max_tokens,
    )


# Default LLM - can be overridden
llm = create_llm()


# =============================================================================
# Continuation Logic for Long Responses
# =============================================================================

def stream_with_continuation(
    messages: list,
    max_continuations: int = 5,
    continuation_prompt: str = "Continue from where you left off. Do not repeat what you've already written."
) -> tuple[str, str]:
    """
    Stream LLM response with automatic continuation if response is truncated.
    
    Args:
        messages: Initial messages to send
        max_continuations: Maximum number of continuation calls
        continuation_prompt: Prompt to use for continuations
        
    Returns:
        Tuple of (full_response, finish_reason)
    """
    full_response = ""
    finish_reason = "unknown"
    conversation = list(messages)  # Copy to avoid mutation
    
    for iteration in range(max_continuations + 1):
        chunk_text = ""
        current_finish_reason = None
        
        # Stream the response
        for chunk in llm.stream(conversation):
            if chunk.content:
                chunk_text += chunk.content
                print(".", end="", flush=True)
            
            # Try to get finish reason from chunk metadata
            if hasattr(chunk, 'response_metadata'):
                metadata = chunk.response_metadata
                if isinstance(metadata, dict) and 'finish_reason' in metadata:
                    current_finish_reason = metadata['finish_reason']
        
        full_response += chunk_text
        
        # Check if we got a complete response
        # OpenAI returns 'stop' when complete, 'length' when truncated
        if current_finish_reason:
            finish_reason = current_finish_reason
        
        # Heuristic checks for incomplete response
        response_seems_incomplete = (
            finish_reason == "length" or
            chunk_text.rstrip().endswith(("...", "—", "-", ",")) or
            (len(chunk_text) > 100 and not chunk_text.rstrip().endswith((".", "!", "?", '"', "'")))
        )
        
        if not response_seems_incomplete:
            # Response appears complete
            break
        
        if iteration < max_continuations:
            print(f"\n   🔄 Response may be incomplete, continuing (attempt {iteration + 2})...", end="", flush=True)
            
            # Add the partial response and continuation request to conversation
            conversation.append(AIMessage(content=chunk_text))
            conversation.append(HumanMessage(content=continuation_prompt))
        else:
            print(f"\n   ⚠️  Max continuations reached")
    
    return full_response, finish_reason


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
# Web Context Fetching (Tavily)
# =============================================================================

def call_web_for_context(query: str) -> str:
    """
    Fetch relevant web context using Tavily search.
    
    Args:
        query: The search query
        
    Returns:
        Summarized answer from web sources
    """
    client = TavilyClient()
    response = client.search(
        query="query: " + query + "\nProvide detailed and relevant information from recent web sources to answer the query.",
        include_answer="advanced",
        search_depth="advanced"
    )   
    return response["answer"]


# =============================================================================
# Agent Definitions
# Agents read context from files and write output to files
# =============================================================================

@with_retries
def extra_context_agent(state: WritingState) -> dict:
    """
    Analyzes the topic to determine if web search is needed.
    
    Triggers web search if:
    - User wants latest/current information
    - Topic is likely not in a smaller LLM's training data
    
    For time-sensitive queries, includes the current UTC date in the search.
    
    Reads: topic from state
    Writes: web_context to file (if search performed)
    Returns: completion status
    """
    from datetime import timezone
    
    # Get current UTC date for time-sensitive queries
    utc_now = datetime.now(timezone.utc)
    current_date_str = utc_now.strftime("%B %d, %Y")  # e.g., "January 10, 2025"
    current_year = utc_now.strftime("%Y")
    
    print("\n🔎 EXTRA CONTEXT AGENT")
    print(f"   Job ID: {JOB_ID}")
    print(f"   Topic: {state.topic}")
    print(f"   📅 Current UTC date: {current_date_str}")
    print("   Analyzing if web search needed...", end="", flush=True)
    
    try:
        # Ask LLM to analyze if web search is needed
        analysis_messages = [
            SystemMessage(content=f"""You are an assistant that determines whether a topic requires web search for current information.

IMPORTANT: Today's date is {current_date_str} (UTC). Use this as your reference for what counts as "recent" or "current".

Analyze the given topic and respond with ONLY a JSON object (no markdown, no explanation):
{{
    "needs_web_search": true/false,
    "is_time_sensitive": true/false,
    "reason": "brief explanation",
    "search_query": "optimized search query if needed, otherwise empty string"
}}

Set needs_web_search to TRUE if ANY of these apply:
- Topic asks about current events, recent developments, or "latest" information
- Topic involves statistics, data, or facts that change over time (prices, populations, rankings)
- Topic is about recent technology, products, or services released in the last 2-3 years
- Topic involves current policies, regulations, or laws
- Topic is about specific people's current roles, positions, or recent activities
- Topic is niche or specialized enough that a general LLM may have limited knowledge
- Topic involves recent scientific discoveries or research

Set is_time_sensitive to TRUE if the topic involves:
- Any temporal language (latest, recent, current, now, today, this year, etc.)
- Events, news, or developments
- Statistics or data that change over time
- Current state of anything (who is president, current prices, etc.)

Set needs_web_search to FALSE if:
- Topic is about well-established historical facts
- Topic is about fundamental concepts, theories, or principles
- Topic is clearly creative/fictional writing
- Topic is about timeless skills or general how-to guides

For search_query: Create an optimized search query. Do NOT include dates in the search_query - dates will be added automatically if needed."""),
            HumanMessage(content=f"Analyze this topic: {state.topic}")
        ]
        
        # Get analysis (no streaming needed for short response)
        analysis_response = llm.invoke(analysis_messages)
        analysis_text = analysis_response.content.strip()
        
        # Parse JSON response
        # Clean up response if it has markdown code blocks
        if "```" in analysis_text:
            analysis_text = analysis_text.split("```")[1]
            if analysis_text.startswith("json"):
                analysis_text = analysis_text[4:]
            analysis_text = analysis_text.strip()
        
        analysis = json.loads(analysis_text)
        
        needs_search = analysis.get("needs_web_search", False)
        is_time_sensitive = analysis.get("is_time_sensitive", False)
        reason = analysis.get("reason", "")
        search_query = analysis.get("search_query", state.topic)
        
        print(" Done!")
        print(f"   📊 Needs web search: {needs_search}")
        print(f"   ⏰ Time sensitive: {is_time_sensitive}")
        print(f"   💭 Reason: {reason}")
        
        if needs_search:
            # Add date context to time-sensitive queries
            if is_time_sensitive:
                # Prepend date context to the search query
                search_query = f"{search_query} as of {current_date_str}"
                print(f"   📅 Added date context to query")
            
            print(f"   🌐 Fetching web context...")
            print(f"   🔍 Query: {search_query}")
            
            try:
                web_context = call_web_for_context(search_query)
                
                # Format the context for use by other agents
                formatted_context = f"""# Web Research Context

**Search Query:** {search_query}
**Search Date:** {current_date_str} (UTC)
**Reason for Search:** {reason}
**Time Sensitive:** {"Yes" if is_time_sensitive else "No"}

## Research Findings

{web_context}

---
*This context was automatically gathered on {current_date_str} to supplement the article with current information.*
"""
                save_to_working("web_context", formatted_context)
                print(f"   ✅ Web context retrieved ({len(web_context)} chars)")
                
            except Exception as e:
                print(f"   ⚠️  Web search failed: {e}")
                # Don't fail the whole pipeline, just note the error
                save_to_working("web_context", f"# Web Search Failed\n\nError: {str(e)}\n\nProceeding without web context.")
        else:
            print("   ⏭️  Skipping web search (not needed)")
            # Save empty context file to indicate analysis was done
            save_to_working("web_context", "# No Web Context Needed\n\nThe topic does not require current web information.")
        
        return {"context_complete": True}
    
    except json.JSONDecodeError as e:
        print(f" Warning: Could not parse analysis response: {e}")
        print("   ⏭️  Skipping web search (analysis failed)")
        save_to_working("web_context", "# Analysis Failed\n\nCould not determine if web search was needed. Proceeding without web context.")
        return {"context_complete": True}
    
    except Exception as e:
        print(f" Failed: {e}")
        return {"error": f"Extra context agent failed: {str(e)}"}

@with_retries
def outline_agent(state: WritingState) -> dict:
    """
    Creates a structured outline from the topic.
    
    Reads: topic from state, web_context from file (if available)
    Writes: outline to file
    Returns: completion status
    """
    if state.error:
        return {}
    
    print("\n📋 OUTLINE AGENT")
    print(f"   Job ID: {JOB_ID}")
    print(f"   Topic: {state.topic}")
    
    # Check for web context
    web_context = load_from_working("web_context")
    has_web_context = web_context and "No Web Context Needed" not in web_context and "Failed" not in web_context
    
    if has_web_context:
        print(f"   📚 Using web context ({len(web_context)} chars)")
    
    print("   Working...", end="", flush=True)
    
    try:
        # Build the prompt with optional web context
        system_prompt = """You are an expert outline creator. 
Create a clear, hierarchical outline with:
- A compelling title
- 5-7 main sections with Roman numerals
- 3-4 subsections under each main section
- Brief notes on key points to cover
- Estimated word count targets for each section

Keep the outline focused and well-organized. This outline will be used to write a comprehensive, long-form article."""

        if has_web_context:
            system_prompt += """

IMPORTANT: You have been provided with current web research on this topic. 
Incorporate relevant facts, statistics, and recent developments from this research into your outline.
Ensure the outline reflects the most up-to-date information available."""

        user_content = f"Create a detailed outline for a comprehensive article about: {state.topic}"
        
        if has_web_context:
            user_content += f"\n\n## Research Context\n\n{web_context}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        
        # Stream with continuation support
        full_response, finish_reason = stream_with_continuation(messages)
        
        print(" Done!")
        print(f"   📄 Outline length: {len(full_response)} chars")
        print(f"   🏁 Finish reason: {finish_reason}")
        
        # Save to working directory
        save_to_working("outline", full_response)
        
        return {"outline_complete": True}
    
    except Exception as e:
        print(f" Failed: {e}")
        return {"error": f"Outline agent failed: {str(e)}"}


@with_retries
def writing_agent(state: WritingState) -> dict:
    """
    Writes a first draft based on the outline.
    Uses continuation for long articles.
    
    Reads: outline from file, web_context from file (if available)
    Writes: draft to file
    Returns: completion status
    """
    # Skip if previous agent errored
    if state.error:
        return {}
    
    # Load outline from file
    outline = load_from_working("outline")
    if not outline:
        return {"error": "Writing agent failed: Could not load outline from file"}
    
    # Check for web context
    web_context = load_from_working("web_context")
    has_web_context = web_context and "No Web Context Needed" not in web_context and "Failed" not in web_context
    
    print("\n✍️  WRITING AGENT")
    print(f"   Job ID: {JOB_ID}")
    print(f"   Loaded outline ({len(outline)} chars) from file")
    if has_web_context:
        print(f"   📚 Using web context ({len(web_context)} chars)")
    print("   Writing...", end="", flush=True)
    
    try:
        system_prompt = """You are an expert long-form writer. 
Write a comprehensive, well-structured article based on the provided outline.

Guidelines:
- Follow the outline structure closely
- Use clear, engaging prose with smooth transitions
- Include relevant examples and explanations
- Aim for 1500-2500 words total
- Write in a professional but accessible tone
- Complete all sections - do not truncate or summarize
- If you run out of space, end at a natural paragraph break so you can continue"""

        if has_web_context:
            system_prompt += """

IMPORTANT: You have been provided with current web research on this topic.
Use specific facts, statistics, and recent information from this research in your article.
Cite sources naturally where appropriate (e.g., "According to recent studies..." or "As of 2024...")."""

        user_content = f"Write a comprehensive article based on this outline:\n\n{outline}"
        
        if has_web_context:
            user_content += f"\n\n## Research Context (use this for accurate, current information)\n\n{web_context}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        
        # Stream with continuation support for long articles
        full_response, finish_reason = stream_with_continuation(
            messages, 
            max_continuations=8,
            continuation_prompt="Continue writing the article from exactly where you stopped. Do not repeat any content. Pick up mid-section if needed."
        )
        
        print(" Done!")
        word_count = len(full_response.split())
        print(f"   📝 Draft length: {word_count} words")
        print(f"   🏁 Finish reason: {finish_reason}")
        
        # Save to working directory
        save_to_working("draft", full_response)
        
        return {"draft_complete": True}
    
    except Exception as e:
        print(f" Failed: {e}")
        return {"error": f"Writing agent failed: {str(e)}"}


@with_retries  
def editor_agent(state: WritingState) -> dict:
    """
    Reviews and improves the draft.
    Handles long articles with continuation.
    
    Reads: draft from file
    Writes: final_draft and revision_notes to files
    Returns: completion status
    """
    if state.error:
        return {}
    
    # Load draft from file
    draft = load_from_working("draft")
    if not draft:
        return {"error": "Editor agent failed: Could not load draft from file"}
    
    word_count = len(draft.split())
    print("\n🔍 EDITOR AGENT")
    print(f"   Job ID: {JOB_ID}")
    print(f"   Loaded draft ({word_count} words) from file")
    print("   Editing...", end="", flush=True)
    
    try:
        messages = [
            SystemMessage(content="""You are an expert editor working on long-form content.
Review and improve the draft article comprehensively.

Your tasks:
1. Fix any grammatical or spelling errors
2. Improve clarity and flow throughout
3. Strengthen weak sentences and transitions
4. Ensure consistent tone and style
5. Enhance readability without changing the meaning
6. Preserve the full length - do not truncate or summarize
7. At the very end, add a "---REVISION NOTES---" section listing your key changes

Return the COMPLETE improved article followed by your revision notes.
If the article is long, maintain its full length in your edited version."""),
            HumanMessage(content=f"Edit and improve this draft (preserve full length):\n\n{draft}")
        ]
        
        # Stream with continuation support
        full_response, finish_reason = stream_with_continuation(
            messages,
            max_continuations=8,
            continuation_prompt="Continue the edited article from exactly where you stopped. Do not repeat content. Complete all remaining sections and then add the revision notes at the end."
        )
        
        print(" Done!")
        print(f"   🏁 Finish reason: {finish_reason}")
        
        # Try to separate revision notes if present
        if "---REVISION NOTES---" in full_response:
            parts = full_response.split("---REVISION NOTES---", 1)
            final_draft = parts[0].strip()
            revision_notes = "REVISION NOTES\n" + parts[1].strip() if len(parts) > 1 else ""
        elif "Revision Notes" in full_response:
            parts = full_response.split("Revision Notes", 1)
            final_draft = parts[0].strip()
            revision_notes = "Revision Notes" + parts[1] if len(parts) > 1 else ""
        else:
            final_draft = full_response
            revision_notes = ""
        
        final_word_count = len(final_draft.split())
        print(f"   📝 Final draft: {final_word_count} words")
        
        # Save to working directory
        save_to_working("editor_output", full_response)
        save_to_working("final_draft", final_draft)
        if revision_notes:
            save_to_working("revision_notes", revision_notes)
        
        return {"edit_complete": True}
    
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
    
    # Load draft from file to check word count
    draft = load_from_working("draft")
    word_count = len(draft.split()) if draft else 0
    
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
    workflow.add_node("extra_context", extra_context_agent)
    workflow.add_node("outline", outline_agent)
    workflow.add_node("writer", writing_agent)
    workflow.add_node("editor", editor_agent)
    
    # Add edges
    # Flow: START → extra_context → outline → writer → editor/end → END
    workflow.add_edge(START, "extra_context")
    workflow.add_edge("extra_context", "outline")
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
# Interactive Mode Detection
# =============================================================================

def is_interactive_mode() -> bool:
    """
    Detect if the script is running in an interactive environment (IDE).
    
    Returns True if:
    - Running in an IDE (PyCharm, VS Code, etc.)
    - Running in an interactive Python shell
    - stdin is a terminal
    """
    # Check for common IDE indicators
    ide_indicators = [
        "PYCHARM_HOSTED" in os.environ,
        "VSCODE_PID" in os.environ,
        "TERM_PROGRAM" in os.environ and "vscode" in os.environ.get("TERM_PROGRAM", "").lower(),
        "JUPYTER_RUNTIME_DIR" in os.environ,
        "JPY_PARENT_PID" in os.environ,
        "SPYDER" in os.environ,
    ]
    
    if any(ide_indicators):
        return True
    
    # Check if running interactively (no command line args and stdin is a terminal)
    try:
        if len(sys.argv) <= 1 and sys.stdin.isatty():
            return True
    except:
        pass
    
    return False


def get_topic_interactively() -> str:
    """Prompt the user for a topic interactively."""
    print("\n" + "=" * 60)
    print("📝 WRITING ASSISTANT - Interactive Mode")
    print("=" * 60)
    print("\nEnter the topic you'd like to write about.")
    print("(Press Enter for default topic)\n")
    
    default_topic = "The Future of Artificial Intelligence in Healthcare: Opportunities, Challenges, and Ethical Considerations"
    
    try:
        user_input = input(f"Topic [{default_topic[:50]}...]: ").strip()
        return user_input if user_input else default_topic
    except (EOFError, KeyboardInterrupt):
        print("\nUsing default topic...")
        return default_topic


# =============================================================================
# Main Execution
# =============================================================================

def run_writing_assistant(
    topic: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 4096
) -> dict:
    """
    Run the complete writing assistant pipeline.
    
    Args:
        topic: The topic to write about
        model: OpenAI model to use
        temperature: Creativity level (0-1)
        max_tokens: Maximum tokens per LLM call
    
    Returns:
        Dict with completion status and file paths
    """
    global llm, JOB_ID
    
    # Generate unique job ID for this run
    JOB_ID = generate_job_id()
    
    llm = create_llm(model=model, temperature=temperature, max_tokens=max_tokens)
    
    print("=" * 60)
    print("🚀 WRITING ASSISTANT STARTING")
    print("=" * 60)
    print(f"Job ID: {JOB_ID}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Max Tokens: {max_tokens}")
    print(f"Working Dir: {WORKING_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Topic: {topic}")
    
    # Create initial state
    initial_state = WritingState(topic=topic, job_id=JOB_ID)
    
    # Save initial state info
    job_info = f"""# Job Info

- **Job ID:** {JOB_ID}
- **Started:** {datetime.now().isoformat()}
- **Model:** {model}
- **Temperature:** {temperature}
- **Topic:** {topic}
"""
    save_to_working("job_info", job_info)
    
    # Build and run workflow
    app = create_workflow()
    
    # Run workflow
    final_state = app.invoke(initial_state)
    
    # Convert final_state to dict if needed
    if hasattr(final_state, 'model_dump'):
        return final_state.model_dump()
    return dict(final_state)


def main():
    """CLI entry point."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("   Set it in a .env file or export it directly")
        sys.exit(1)
    
    # Determine topic based on run mode
    if is_interactive_mode():
        topic = get_topic_interactively()
    elif len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = "The Future of Artificial Intelligence in Healthcare: Opportunities, Challenges, and Ethical Considerations"
    
    try:
        result = run_writing_assistant(
            topic=topic,
            model="gpt-4o-mini",  # Use "gpt-4o" for better quality
            temperature=0.7,
            max_tokens=4096
        )
        
        # Output results
        print("\n" + "=" * 60)
        print("✅ COMPLETE!")
        print("=" * 60)
        
        if result.get("error"):
            print(f"\n❌ Error occurred: {result['error']}")
            
            # Save error info
            error_path = save_final_output(
                f"# Error\n\n{result['error']}\n\nPartial results may be in working directory.",
                "error.md"
            )
            print(f"   Error log saved to: {error_path}")
        else:
            # Load final content from files
            final_content = load_from_working("final_draft") or load_from_working("draft")
            revision_notes = load_from_working("revision_notes")
            
            # Print to screen
            word_count = len(final_content.split()) if final_content else 0
            print(f"\n📄 FINAL ARTICLE ({word_count} words):")
            print("-" * 40)
            print(final_content)
            
            if revision_notes:
                print("\n" + "-" * 40)
                print(revision_notes)
            
            # Save to output directory
            full_output = f"""# {topic}

**Job ID:** {JOB_ID}  
**Generated:** {datetime.now().isoformat()}

---

{final_content}

"""
            if revision_notes:
                full_output += f"""
---

{revision_notes}
"""
            
            output_path = save_final_output(full_output, "final_article.md")
            print(f"\n   📁 Output saved to: {output_path}")
            
            # Also save just the article without metadata
            draft_output_path = save_final_output(final_content, "article_draft.md")
            print(f"   📁 Article draft saved to: {draft_output_path}")
            
            # Print summary of all files
            print(f"\n📂 Working files in {WORKING_DIR}:")
            for f in sorted(WORKING_DIR.glob(f"{JOB_ID}_*")):
                print(f"   - {f.name}")
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()