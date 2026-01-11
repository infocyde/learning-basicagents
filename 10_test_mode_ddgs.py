"""
Advanced LangGraph Writing Assistant with Multi-Agent Workflow

Enhanced Features:
- Error handling with retries
- Streaming output with continuation for long responses
- Four agents: Extra Context → Outliner → Writer → Editor
- Automatic web search via Tavily for current/niche topics
- MULTI-PART TAVILY: Breaks complex queries into up to 10 sub-queries
- EXTERNALIZED PROMPTS: All agent prompts loaded from /prompts directory
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
from ddgs import DDGS


load_dotenv()

# =============================================================================
# Hard Coded Variables Configuration
# =============================================================================
# When TEST_MODE is True, uses free DuckDuckGo search instead of Tavily API
# Useful for development/testing without consuming Tavily API credits
TEST_MODE = True  # Set to True to use DDGS instead of Tavily
llm_model = "gpt-5-mini"  # Default LLM model to use

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


# Directories relative to where the script lives
SCRIPT_DIR = get_script_directory()
WORKING_DIR = SCRIPT_DIR / "wip"
OUTPUT_DIR = SCRIPT_DIR / "output"
PROMPTS_DIR = SCRIPT_DIR / "prompts"

# Create directories if they don't exist
WORKING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Script directory: {SCRIPT_DIR}")
print(f"📁 Working directory: {WORKING_DIR}")
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"📁 Prompts directory: {PROMPTS_DIR}")


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
# Prompt Loading Utilities
# =============================================================================

def load_prompt(prompt_name: str, **kwargs) -> str:
    """
    Load a prompt from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
                     Format: prod_[agent_name].txt
        **kwargs: Variables to substitute in the prompt using .format()
        
    Returns:
        The prompt text with any variables substituted
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    filepath = PROMPTS_DIR / f"{prompt_name}.txt"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    
    prompt_text = filepath.read_text(encoding="utf-8")
    
    # Substitute any provided variables
    if kwargs:
        prompt_text = prompt_text.format(**kwargs)
    
    return prompt_text


def load_prompt_safe(prompt_name: str, fallback: str = "", **kwargs) -> str:
    """
    Load a prompt from the prompts directory with a fallback.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        fallback: Default text to return if file not found
        **kwargs: Variables to substitute in the prompt
        
    Returns:
        The prompt text, or fallback if file not found
    """
    try:
        return load_prompt(prompt_name, **kwargs)
    except FileNotFoundError as e:
        print(f"   ⚠️  {e} - using fallback")
        return fallback


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


def append_user_prompt(topic: str) -> Path:
    """
    Append the user's prompt/topic to a history file in the working directory.
    
    Args:
        topic: The user's topic/prompt
        
    Returns:
        Path to the prompts file
    """
    filepath = WORKING_DIR / "user_prompts.txt"
    timestamp = datetime.now().isoformat()
    
    entry = f"""
================================================================================
Job ID: {JOB_ID}
Timestamp: {timestamp}
--------------------------------------------------------------------------------
{topic}
================================================================================
"""
    append_to_file(filepath, entry)
    print(f"   📝 Appended prompt to: {filepath}")
    return filepath


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
# Multi-Part Query Breaking (NEW)
# =============================================================================

def break_query_into_parts(topic: str, current_date_str: str) -> list[dict]:
    """
    Use LLM to break a complex query into multiple focused sub-queries.
    
    Each sub-query targets a different aspect of the topic to maximize
    the information gathered from Tavily's ~2000 word limit per call.
    
    Args:
        topic: The main topic/query to break down
        current_date_str: Current date string for context
        
    Returns:
        List of dicts with 'part_name' and 'query' keys
    """
    print("   🔧 Breaking query into focused sub-parts...")
    
    # Load prompt from file
    system_prompt = load_prompt("prod_query_breakdown", current_date_str=current_date_str)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Break down this topic into focused research sub-queries:\n\n{topic}")
    ]
    
    try:
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Clean up response if it has markdown code blocks
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        parts = json.loads(response_text)
        
        # Validate and limit to 10 parts
        if not isinstance(parts, list):
            raise ValueError("Response is not a list")
        
        parts = parts[:10]  # Limit to max 10 parts
        
        # Ensure each part has required fields
        validated_parts = []
        for i, part in enumerate(parts):
            if isinstance(part, dict) and 'query' in part:
                validated_parts.append({
                    'part_name': part.get('part_name', f'Part {i+1}'),
                    'query': part['query']
                })
        
        if not validated_parts:
            # Fallback: use original topic as single query
            validated_parts = [{'part_name': 'Main Query', 'query': topic}]
        
        print(f"   ✅ Created {len(validated_parts)} sub-queries")
        return validated_parts
        
    except Exception as e:
        print(f"   ⚠️  Query breakdown failed ({e}), using single query")
        return [{'part_name': 'Main Query', 'query': topic}]


def call_tavily_single(query: str) -> str:
    """
    Make a single Tavily API call.
    
    Args:
        query: The search query
        
    Returns:
        The answer from Tavily
    """
    client = TavilyClient()
    response = client.search(
        query="query: " + query + "\nProvide detailed and relevant information from recent web sources to answer the query.",
        include_answer="advanced",
        search_depth="advanced"
    )   
    return response.get("answer", "No answer returned")


def call_ddgs_single(query: str, max_results: int = 8) -> str:
    """
    Make a single DuckDuckGo search API call (unofficial, free).
    
    Args:
        query: The search query
        max_results: Number of results to fetch
        
    Returns:
        Combined text from search results
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        
        if not results:
            return "No results found"
        
        # Combine results into a formatted string
        combined = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No content')
            href = result.get('href', '')
            combined.append(f"**{title}**\n{body}\nSource: {href}")
        
        return "\n\n---\n\n".join(combined)
        
    except Exception as e:
        return f"DDGS search error: {str(e)}"


def call_web_single(query: str) -> str:
    """
    Make a single web search call using either Tavily or DDGS based on TEST_MODE.
    
    Args:
        query: The search query
        
    Returns:
        Search results/answer
    """
    if TEST_MODE:
        return call_ddgs_single(query)
    else:
        return call_tavily_single(query)


def call_web_for_context_multipart(topic: str, is_time_sensitive: bool, current_date_str: str) -> tuple[str, str]:
    """
    Fetch web context using multiple search calls for comprehensive coverage.
    
    Breaks the topic into multiple sub-queries, executes each separately,
    and combines the results. Uses Tavily or DDGS based on TEST_MODE.
    
    Args:
        topic: The main topic/query
        is_time_sensitive: Whether to add date context to queries
        current_date_str: Current date string
        
    Returns:
        Tuple of (combined_context, breakdown_summary)
    """
    # Step 1: Break query into parts
    query_parts = break_query_into_parts(topic, current_date_str)
    
    # Determine search provider
    search_provider = "DDGS (Test Mode)" if TEST_MODE else "Tavily"
    
    # Step 2: Save the breakdown plan
    breakdown_summary = f"""# Query Breakdown Plan

**Original Topic:** {topic}
**Date:** {current_date_str}
**Number of Sub-Queries:** {len(query_parts)}
**Search Provider:** {search_provider}

## Sub-Queries:
"""
    for i, part in enumerate(query_parts, 1):
        breakdown_summary += f"\n### Part {i}: {part['part_name']}\n"
        breakdown_summary += f"**Query:** {part['query']}\n"
    
    # Step 3: Execute each query and collect results
    all_results = []
    combined_context = ""
    
    print(f"\n   📡 Executing {len(query_parts)} {search_provider} API calls...")
    print("   " + "=" * 50)
    
    for i, part in enumerate(query_parts, 1):
        part_name = part['part_name']
        query = part['query']
        
        # Add date context if time-sensitive
        if is_time_sensitive:
            query = f"{query} as of {current_date_str}"
        
        print(f"\n   [{i}/{len(query_parts)}] 🔍 {part_name}")
        print(f"         Query: {query[:80]}{'...' if len(query) > 80 else ''}")
        
        try:
            result = call_web_single(query)
            result_preview = result[:200] + "..." if len(result) > 200 else result
            print(f"         ✅ Retrieved {len(result)} chars")
            
            # Store result
            all_results.append({
                'part_name': part_name,
                'query': query,
                'result': result,
                'char_count': len(result)
            })
            
            # Add to combined context
            combined_context += f"\n\n## {part_name}\n\n"
            combined_context += f"**Query:** {query}\n\n"
            combined_context += result
            
            # Update breakdown summary with result info
            breakdown_summary += f"**Result:** {len(result)} chars retrieved\n"
            
        except Exception as e:
            print(f"         ❌ Failed: {e}")
            breakdown_summary += f"**Result:** FAILED - {str(e)}\n"
            all_results.append({
                'part_name': part_name,
                'query': query,
                'result': f"Error: {str(e)}",
                'char_count': 0
            })
    
    print("\n   " + "=" * 50)
    
    # Step 4: Summary statistics
    total_chars = sum(r['char_count'] for r in all_results)
    successful = sum(1 for r in all_results if r['char_count'] > 0)
    
    print(f"   📊 Summary: {successful}/{len(query_parts)} queries successful")
    print(f"   📊 Total context gathered: {total_chars} chars (~{total_chars // 4} tokens)")
    
    breakdown_summary += f"""
---

## Execution Summary

- **Successful Queries:** {successful}/{len(query_parts)}
- **Total Characters Retrieved:** {total_chars}
- **Approximate Tokens:** {total_chars // 4}
"""
    
    # Step 5: Save individual results to separate file
    individual_results_content = "# Individual Tavily Results\n\n"
    for i, r in enumerate(all_results, 1):
        individual_results_content += f"---\n\n## Part {i}: {r['part_name']}\n\n"
        individual_results_content += f"**Query:** {r['query']}\n\n"
        individual_results_content += f"**Characters:** {r['char_count']}\n\n"
        individual_results_content += f"### Result:\n\n{r['result']}\n\n"
    
    save_to_working("tavily_individual_results", individual_results_content)
    
    return combined_context, breakdown_summary


# Legacy single-call function (kept for reference)
def call_web_for_context(query: str) -> str:
    """
    Fetch relevant web context using Tavily search (single call).
    
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
    Uses MULTI-PART Tavily calls for comprehensive coverage.
    
    Reads: topic from state
    Writes: web_context, tavily_breakdown, tavily_individual_results to files
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
        # Load analysis prompt from file
        system_prompt = load_prompt("prod_extra_context_analysis", current_date_str=current_date_str)
        
        analysis_messages = [
            SystemMessage(content=system_prompt),
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
        complexity = analysis.get("complexity", "moderate")
        reason = analysis.get("reason", "")
        search_query = analysis.get("search_query", state.topic)
        
        print(" Done!")
        print(f"   📊 Needs web search: {needs_search}")
        print(f"   ⏰ Time sensitive: {is_time_sensitive}")
        print(f"   📈 Complexity: {complexity}")
        print(f"   💭 Reason: {reason}")
        
        if needs_search:
            print(f"   🌐 Initiating MULTI-PART web context gathering...")
            print(f"   🔍 Base topic: {state.topic[:80]}{'...' if len(state.topic) > 80 else ''}")
            
            try:
                # Use multi-part Tavily calls
                combined_context, breakdown_summary = call_web_for_context_multipart(
                    topic=state.topic,
                    is_time_sensitive=is_time_sensitive,
                    current_date_str=current_date_str
                )
                
                # Save the breakdown plan
                save_to_working("tavily_breakdown", breakdown_summary)
                
                # Format the combined context for use by other agents
                formatted_context = f"""# Web Research Context (Multi-Part)

**Original Topic:** {state.topic}
**Search Date:** {current_date_str} (UTC)
**Reason for Search:** {reason}
**Time Sensitive:** {"Yes" if is_time_sensitive else "No"}
**Complexity Level:** {complexity}

---

{combined_context}

---
*This context was automatically gathered on {current_date_str} using multiple focused queries to ensure comprehensive coverage.*
"""
                save_to_working("web_context", formatted_context)
                print(f"\n   ✅ Web context retrieved ({len(combined_context)} chars total)")
                
            except Exception as e:
                print(f"   ⚠️  Web search failed: {e}")
                # Don't fail the whole pipeline, just note the error
                save_to_working("web_context", f"# Web Search Failed\n\nError: {str(e)}\n\nProceeding without web context.")
                save_to_working("tavily_breakdown", f"# Breakdown Failed\n\nError: {str(e)}")
        else:
            print("   ⏭️  Skipping web search (not needed)")
            # Save empty context file to indicate analysis was done
            save_to_working("web_context", "# No Web Context Needed\n\nThe topic does not require current web information.")
            save_to_working("tavily_breakdown", "# No Breakdown Needed\n\nWeb search was not required for this topic.")
        
        return {"context_complete": True}
    
    except json.JSONDecodeError as e:
        print(f" Warning: Could not parse analysis response: {e}")
        print("   ⏭️  Skipping web search (analysis failed)")
        save_to_working("web_context", "# Analysis Failed\n\nCould not determine if web search was needed. Proceeding without web context.")
        save_to_working("tavily_breakdown", "# Analysis Failed\n\nCould not parse LLM response.")
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
    
    # Also load the breakdown info for reference
    tavily_breakdown = load_from_working("tavily_breakdown")
    has_breakdown = tavily_breakdown and "No Breakdown Needed" not in tavily_breakdown and "Failed" not in tavily_breakdown
    
    if has_web_context:
        print(f"   📚 Using web context ({len(web_context)} chars)")
    if has_breakdown:
        print(f"   📋 Query breakdown available for reference")
    
    print("   Working...", end="", flush=True)
    
    try:
        # Load base prompt from file
        system_prompt = load_prompt("prod_outline")

        if has_web_context:
            # Append web context instructions
            web_context_prompt = load_prompt_safe("prod_outline_web_context", fallback="""

IMPORTANT: You have been provided with current web research on this topic, gathered from multiple focused queries.
Incorporate relevant facts, statistics, and recent developments from this research into your outline.
Ensure the outline reflects the most up-to-date information available.
The research covers multiple aspects of the topic - make sure to address all major areas covered.""")
            system_prompt += web_context_prompt

        user_content = f"Create a detailed outline for a comprehensive article about: {state.topic}"
        
        if has_web_context:
            user_content += f"\n\n## Research Context (from multiple web queries)\n\n{web_context}"
        
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
    
    # Check for web context - NOW ALSO READING FROM FILE
    web_context = load_from_working("web_context")
    has_web_context = web_context and "No Web Context Needed" not in web_context and "Failed" not in web_context
    
    # Also load individual Tavily results for more detailed reference
    individual_results = load_from_working("tavily_individual_results")
    has_individual_results = individual_results and len(individual_results) > 100
    
    print("\n✍️  WRITING AGENT")
    print(f"   Job ID: {JOB_ID}")
    print(f"   Loaded outline ({len(outline)} chars) from file")
    if has_web_context:
        print(f"   📚 Loading web context from file ({len(web_context)} chars)")
    if has_individual_results:
        print(f"   📑 Loading individual Tavily results ({len(individual_results)} chars)")
    print("   Writing...", end="", flush=True)
    
    try:
        # Load base prompt from file
        system_prompt = load_prompt("prod_writer")

        if has_web_context:
            # Append web context instructions
            web_context_prompt = load_prompt_safe("prod_writer_web_context", fallback="""

IMPORTANT: You have been provided with current web research gathered from multiple focused queries.
Use specific facts, statistics, and recent information from this research in your article.
The research covers various aspects of the topic - incorporate relevant details from each section.
Cite sources naturally where appropriate (e.g., "According to recent studies..." or "As of 2024...").""")
            system_prompt += web_context_prompt

        user_content = f"Write a comprehensive article based on this outline:\n\n{outline}"
        
        if has_web_context:
            user_content += f"\n\n## Web Research Context (use this for accurate, current information)\n\n{web_context}"
        
        # If we have detailed individual results and they're not too long, include them too
        if has_individual_results and len(individual_results) < 50000:
            user_content += f"\n\n## Detailed Research Results (additional context)\n\n{individual_results}"

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
        # Load prompt from file
        system_prompt = load_prompt("prod_editor")
        
        messages = [
            SystemMessage(content=system_prompt),
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
    
    # Append user's prompt to history file
    append_user_prompt(topic)
    
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
    print(f"Prompts Dir: {PROMPTS_DIR}")
    print(f"Topic: {topic}")
    print(f"🆕 Multi-Part Web Search: ENABLED (up to 10 sub-queries)")
    print(f"🆕 External Prompts: ENABLED")
    if TEST_MODE:
        print(f"🧪 TEST MODE: ENABLED (using DDGS instead of Tavily)")
    else:
        print(f"🌐 Search Provider: Tavily")
    
    # Verify prompt files exist
    required_prompts = [
        "prod_extra_context_analysis",
        "prod_query_breakdown", 
        "prod_outline",
        "prod_writer",
        "prod_editor"
    ]
    missing_prompts = []
    for prompt_name in required_prompts:
        filepath = PROMPTS_DIR / f"{prompt_name}.txt"
        if not filepath.exists():
            missing_prompts.append(prompt_name)
    
    if missing_prompts:
        print(f"\n⚠️  Missing prompt files: {missing_prompts}")
        print(f"   Please ensure all prompt files exist in: {PROMPTS_DIR}")
    else:
        print(f"✅ All {len(required_prompts)} required prompt files found")
    
    # Create initial state
    initial_state = WritingState(topic=topic, job_id=JOB_ID)
    
    # Save initial state info
    search_provider = "DDGS (Test Mode)" if TEST_MODE else "Tavily"
    job_info = f"""# Job Info

- **Job ID:** {JOB_ID}
- **Started:** {datetime.now().isoformat()}
- **Model:** {model}
- **Temperature:** {temperature}
- **Topic:** {topic}
- **Multi-Part Web Search:** Enabled
- **Search Provider:** {search_provider}
- **External Prompts:** Enabled
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
    
    if not os.getenv("TAVILY_API_KEY") and not TEST_MODE:
        print("⚠️  Warning: TAVILY_API_KEY not found in environment")
        print("   Web search features will not work without it")
        print("   (Set TEST_MODE = True to use free DDGS search instead)")
    
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
            tavily_breakdown = load_from_working("tavily_breakdown")
            
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
            
            # Highlight web search-related files
            print(f"\n📊 Web Search Multi-Part Files:")
            search_files = ["tavily_breakdown", "tavily_individual_results", "web_context"]
            for fname in search_files:
                fpath = get_working_filepath(fname)
                if fpath.exists():
                    size = fpath.stat().st_size
                    print(f"   - {fname}.md ({size} bytes)")
            
            # List prompt files
            print(f"\n📝 Prompt files in {PROMPTS_DIR}:")
            for f in sorted(PROMPTS_DIR.glob("prod_*.txt")):
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
