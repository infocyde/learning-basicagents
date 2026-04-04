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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError
from tavily import TavilyClient  # API client for Tavily web search which uses Bing under the hood and is optimized for LLMs
from ddgs import DDGS # free unofficial DuckDuckGo Search API, brings back list of urls to be used with Trafilatura
import trafilatura # Web page content extraction library 

# Load environment variables from a .env file if present in a directory tree parent to this script
load_dotenv() 

# Clear out previous run results/errors from terminal
os.system("cls" if os.name == "nt" else "clear")


# =============================================================================
# This the base of a SYNCOUS multi-agent writing assistant with advanced features
# =============================================================================
# also expects OpenAI API key in environment for LLM calls
# and optionally a TAVILY_API_KEY for Tavily usage, to use TAVILY you need to sign up for an API key at https://tavily.com and set TEST_MODE to False below


# Note, both tarfilatura and Tavily do not support headless browsing with JavaScript rendering for modern web pages. A problem for both.
# When TEST_MODE is True, uses free DuckDuckGo search + trafilatura instead of Tavily API
# you abuse duckduckgo you may get blocked temporarily. So use responsibly.
# both ddgs and trafilatura have a get images option but since this is a text only agent we disable image downloading since without vision capabilities they are not useful.
# some of the new smaller LLMs are now vision, so there could be another subagent, maybe even an async one, that swaps images out with what it sees in the image
# god level (little g there is only one God) would be to have an agent that does vision, mp3, video, etc analysis and summarization to feed into the writing pipeline


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

# =============================================================================
# User-Configurable Constants
# =============================================================================
TEST_MODE = True  # Set to True to use DDGS + trafilatura instead of Tavily
llm_model = "google_gemma-4-e4b-it"  # Default LLM model to use
llm_temperature = 0.9  # Creativity level (0.0 = deterministic, 1.0 = creative)
context_window_tokens = 32000  # Your model's context window size in tokens (check model spec)
chars_per_token = 3  # Conservative estimate (3 is safer for structured/markdown content)
use_draft_as_final = True  # Set to True to skip the editor agent and use the draft as the final article
force_web_search = False  # Set to True to always search the web, overriding the LLM's analysis (useful for smaller models)
max_parallel_summarizations = 1  # Number of chunk summarization calls to run in parallel. Recommend 1 unless you have GPU headroom or use a remote LLM.
max_parallel_web_calls = 3  # Number of web search/scrape calls to run in parallel. I/O-bound so can go higher than LLM calls.

# =============================================================================
# Research Depth & Source Count Settings
# =============================================================================
RESEARCH_LEVELS = {
    "1": {"name": "Light",  "max_chars_per_source": 6000,  "description": "Quick research — core content only, fast processing"},
    "2": {"name": "Medium", "max_chars_per_source": 12000, "description": "Balanced — deeper content from data-heavy pages"},
    "3": {"name": "Deep",   "max_chars_per_source": 20000, "description": "Thorough — pulls extensive detail, slower processing"},
}
DEFAULT_RESEARCH_LEVEL = "1"

SOURCE_LEVELS = {
    "1": {"name": "Few",     "results_per_query": 2, "description": "2 sources per sub-query — fast, focused"},
    "2": {"name": "Average", "results_per_query": 4, "description": "4 sources per sub-query — balanced coverage"},
    "3": {"name": "Many",    "results_per_query": 7, "description": "7 sources per sub-query — comprehensive, slower"},
}
DEFAULT_SOURCE_LEVEL = "2"

# Active settings (set from defaults, overridden by interactive prompt)
max_chars_per_agent = RESEARCH_LEVELS[DEFAULT_RESEARCH_LEVEL]["max_chars_per_source"]
max_ddg_results = SOURCE_LEVELS[DEFAULT_SOURCE_LEVEL]["results_per_query"]

# =============================================================================
# Derived Constants (auto-calculated from context_window_tokens)
# =============================================================================
context_window_chars = context_window_tokens * chars_per_token
max_tokens = context_window_tokens * 2 // 3
prompt_reserve_chars = context_window_chars // 6
max_individual_results_chars = context_window_chars * 2  # Drop threshold for detailed results in writer

# Status markers written to working files — used to check if content is usable
STATUS_NO_CONTEXT = "No Web Context Needed"
STATUS_NO_BREAKDOWN = "No Breakdown Needed"
STATUS_FAILED = "Failed"

JOB_ID: str = "" # Global job ID - set at runtime

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


def generate_job_id() -> str:
    """Generate a unique job ID for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"


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

def load_working_if_usable(agent_name: str, skip_markers: list[str] = None) -> tuple[str, bool]:
    """
    Load content from working directory and check if it's usable (not a skip/failure marker).

    Args:
        agent_name: Name of the agent whose output to load
        skip_markers: Strings that indicate content should be skipped. Defaults to [STATUS_FAILED].

    Returns:
        Tuple of (content, is_usable)
    """
    if skip_markers is None:
        skip_markers = [STATUS_FAILED]
    content = load_from_working(agent_name)
    if not content:
        return "", False
    for marker in skip_markers:
        if marker in content:
            return content, False
    return content, True

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

# Global references collector — populated during chunk_and_summarize, appended to final output
_collected_references: list[dict] = []

def extract_references(text: str) -> tuple[str, list[dict]]:
    """
    Extract source references from content before it gets summarized.

    Looks for patterns like:
        ## Title
        **Source:** https://example.com

    Returns the text with source lines removed, and a list of reference dicts.
    """
    refs = []
    # Match "## Title\n**Source:** URL" blocks from DDGS/trafilatura output
    pattern = re.compile(
        r'^## (.+)\n\*\*Source:\*\* (https?://\S+)',
        re.MULTILINE
    )
    for match in pattern.finditer(text):
        title = match.group(1).strip()
        url = match.group(2).strip()
        if not any(r['url'] == url for r in refs):
            refs.append({'title': title, 'url': url})

    # Remove the **Source:** lines so the summarizer doesn't mangle URLs
    cleaned = re.sub(r'\*\*Source:\*\* https?://\S+\n?', '', text)
    return cleaned, refs


def collect_references(refs: list[dict]) -> None:
    """Add references to the global collector, deduplicating by URL."""
    for ref in refs:
        if not any(r['url'] == ref['url'] for r in _collected_references):
            _collected_references.append(ref)


def format_references() -> str:
    """Format collected references as a markdown bibliography."""
    if not _collected_references:
        return ""
    lines = ["## References", ""]
    for i, ref in enumerate(_collected_references, 1):
        lines.append(f"{i}. [{ref['title']}]({ref['url']})")
    return "\n".join(lines)


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
# Writing Style Options
# =============================================================================

WRITING_STYLES = {
    "1": {
        "name": "Technical / Factual",
        "description": "Precise, factual, well-structured. Focuses on accuracy and clarity without bureaucratic language.",
        "prompt_instruction": (
            "Write in a precise, factual style. Use clear structured sections, specific data and numbers, "
            "and an objective tone. Be direct — state facts plainly without inflating language. "
            "Say 'two US aircraft were shot down' not 'kinetic asset neutralization events were documented.' "
            "Cite sources where available. Target audience: informed readers who want accurate, well-organized information."
        )
    },
    "2": {
        "name": "Blog Post",
        "description": "Conversational, engaging, accessible. Uses a friendly tone with practical examples.",
        "prompt_instruction": (
            "Write in a conversational blog style. Use an engaging, accessible tone with practical examples, "
            "short paragraphs, and relatable analogies. Include a hook introduction and actionable takeaways. "
            "Target audience: general readers interested in the topic."
        )
    },
    "3": {
        "name": "Narrative / Story",
        "description": "Creative, immersive, story-driven. Weaves facts into compelling narrative arcs.",
        "prompt_instruction": (
            "Write in a narrative, story-driven style. Weave facts and information into compelling narrative arcs "
            "with vivid descriptions, character perspectives where appropriate, and dramatic structure. "
            "Use creative language and immersive storytelling techniques. Target audience: readers who enjoy long-form narrative."
        )
    },
    "4": {
        "name": "News Briefing",
        "description": "Concise, factual, inverted pyramid. Leads with the biggest developments, no fluff.",
        "prompt_instruction": (
            "Write as a news briefing. Use inverted pyramid structure — lead with the most significant developments first, "
            "then provide supporting details and context. Be concise and factual. Every sentence should deliver information. "
            "No filler introductions, no dramatic buildup, no conclusions that just restate what was said. "
            "Use short paragraphs. State numbers, names, and specifics directly. "
            "Target audience: someone who wants to get caught up quickly."
        )
    },
}
DEFAULT_WRITING_STYLE = "2"  # Blog Post


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
    writing_style: str = Field(default=DEFAULT_WRITING_STYLE, description="Writing style key from WRITING_STYLES")
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
    temperature: float = llm_temperature,
    streaming: bool = True,
    max_tokens: int = max_tokens
) -> ChatOpenAI:
    """Create a configured LLM instance."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
        max_tokens=max_tokens,
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
    )


# Default LLM - can be overridden
llm = create_llm()


# =============================================================================
# Continuation Logic for Long Responses
# =============================================================================

def stream_with_continuation(
    messages: list,
    max_continuations: int = 3,
    continuation_prompt: str = "Continue from where you left off. Do not repeat what you've already written."
) -> tuple[str, str]:
    """
    Stream LLM response with automatic continuation if response is truncated.

    Only continues when the server explicitly signals truncation via finish_reason='length'.
    Uses a sliding window approach to prevent context explosion on continuations.

    Args:
        messages: Initial messages to send
        max_continuations: Maximum number of continuation calls (default 3)
        continuation_prompt: Prompt to use for continuations

    Returns:
        Tuple of (full_response, finish_reason)
    """
    full_response = ""
    finish_reason = "unknown"
    # Keep only the original messages for continuations (system + user)
    original_messages = list(messages)

    for iteration in range(max_continuations + 1):
        chunk_text = ""
        current_finish_reason = None

        # Build conversation for this iteration
        if iteration == 0:
            conversation = list(original_messages)
        else:
            # Sliding window: original messages + summary of what's written + continue prompt
            # Only include the tail of what we've written to avoid bloating context
            tail_chars = min(len(full_response), context_window_chars // 6)
            written_so_far = full_response[-tail_chars:]
            conversation = list(original_messages)
            conversation.append(AIMessage(content=f"[Partial response so far, showing last portion:]\n\n{written_so_far}"))
            conversation.append(HumanMessage(content=continuation_prompt))

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
        # Only continue if the server explicitly says it was truncated due to length
        if current_finish_reason:
            finish_reason = current_finish_reason

        if finish_reason != "length":
            # Response is complete (got 'stop', 'unknown', or anything else)
            break

        if iteration < max_continuations:
            print(f"\n   🔄 Response truncated by token limit, continuing (attempt {iteration + 2})...", end="", flush=True)
        else:
            print(f"\n   ⚠️  Max continuations reached ({max_continuations})")

    return full_response, finish_reason


# =============================================================================
# Chunk and Summarize for Large Content
# =============================================================================

def chunk_and_summarize(text: str, purpose: str, max_chars: int = None, max_depth: int = 3) -> str:
    """
    Recursively chunk and summarize text that exceeds the context window.

    If text fits within max_chars, returns it as-is. Otherwise splits into
    chunks, summarizes each via LLM, combines summaries, and recurses if
    the combined result is still too large.

    Args:
        text: The text to potentially chunk and summarize
        purpose: Description of what this content is for (e.g., "web research context for outline generation")
        max_chars: Character threshold. Defaults to context_window_chars - prompt_reserve_chars.
        max_depth: Max recursion depth to prevent runaway loops

    Returns:
        Original text if under threshold, or a condensed summary
    """
    if max_chars is None:
        max_chars = context_window_chars - prompt_reserve_chars

    # Extract and preserve references before any summarization
    text, refs = extract_references(text)
    if refs:
        collect_references(refs)
        print(f"   📎 Extracted {len(refs)} source references (preserved separately)")

    # Base case: text fits
    if len(text) <= max_chars:
        return text

    # Safety: stop recursing
    if max_depth <= 0:
        print(f"   ⚠️  Max summarization depth reached, hard-truncating to {max_chars} chars")
        return text[:max_chars] + "\n\n[Content truncated due to length...]"

    print(f"   📐 Content too long ({len(text)} chars > {max_chars} limit). Chunking and summarizing (depth remaining: {max_depth})...")

    # Split into chunks at section boundaries
    chunks = _split_into_chunks(text, max_chars)
    print(f"   📦 Split into {len(chunks)} chunks")

    # Summarize chunks in parallel
    total = len(chunks)
    print(f"   🔄 Summarizing {total} chunks (max {max_parallel_summarizations} in parallel)...")
    summaries = [None] * total

    with ThreadPoolExecutor(max_workers=max_parallel_summarizations) as executor:
        future_to_idx = {
            executor.submit(_summarize_chunk, chunk, purpose, i + 1, total): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            summary = future.result()
            summaries[idx] = summary
            print(f"   ✅ Chunk {idx + 1}/{total} done ({len(chunks[idx])} → {len(summary)} chars)")

    combined = "\n\n---\n\n".join(summaries)
    print(f"   📊 Combined summaries: {len(combined)} chars")

    # Recurse if still too long
    if len(combined) > max_chars:
        print(f"   🔁 Combined summaries still exceed limit, recursing...")
        return chunk_and_summarize(combined, purpose, max_chars, max_depth - 1)

    return combined


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks that each fit under max_chars.
    Tries to split at section boundaries (## headings, --- dividers, double newlines).
    Oversized sections are further split at paragraph boundaries to ensure no chunk exceeds max_chars.
    """
    # Step 1: Try splitting at markdown section boundaries
    pieces = [text]
    section_separators = ["\n## ", "\n---\n", "\n\n"]

    for separator in section_separators:
        new_pieces = []
        for piece in pieces:
            if len(piece) <= max_chars:
                new_pieces.append(piece)
            else:
                # Try to split this oversized piece at the current separator
                sections = piece.split(separator)
                if len(sections) > 1:
                    for j, section in enumerate(sections):
                        restored = (separator + section) if j > 0 else section
                        new_pieces.append(restored)
                else:
                    new_pieces.append(piece)
        pieces = new_pieces

    # Step 2: Group small pieces together, split remaining oversized pieces by character
    chunks = []
    current_chunk = ""

    for piece in pieces:
        if len(piece) <= max_chars:
            # Try to group with current chunk
            if len(current_chunk) + len(piece) <= max_chars:
                current_chunk += piece
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = piece
        else:
            # Piece is still oversized after all separator splits — force-split by character at paragraph boundaries
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            start = 0
            while start < len(piece):
                end = start + max_chars
                if end >= len(piece):
                    current_chunk = piece[start:]
                    break
                # Try to break at a paragraph boundary
                newline_pos = piece.rfind("\n\n", start, end)
                if newline_pos > start + (max_chars // 2):
                    end = newline_pos
                chunks.append(piece[start:end])
                start = end

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if len(chunks) > 1 else [text[:max_chars], text[max_chars:]]


def _summarize_chunk(chunk: str, purpose: str, chunk_num: int, total_chunks: int) -> str:
    """
    Send a single chunk to the LLM for concise summarization.
    """
    system_prompt = (
        "You are a precise summarizer. Condense the following content into a concise summary "
        "that preserves all key facts, statistics, names, dates, and important details. "
        "Remove redundancy and filler but do NOT omit important information.\n\n"
        f"Purpose of this content: {purpose}\n"
        f"This is chunk {chunk_num} of {total_chunks}."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Summarize this content concisely:\n\n{chunk}")
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f" ⚠️ Summarization failed: {e}")
        # Fallback: hard truncate
        truncated_len = (context_window_chars - prompt_reserve_chars) // 3
        return chunk[:truncated_len] + "\n\n[Summarization failed, content truncated...]"


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

def break_query_into_parts(topic: str, current_date_str: str, complexity: str = "moderate") -> list[dict]:
    """
    Use LLM to break a complex query into multiple focused sub-queries.

    Each sub-query targets a different aspect of the topic to maximize
    the information gathered from Tavily's ~2000 word limit per call.

    Args:
        topic: The main topic/query to break down
        current_date_str: Current date string for context
        complexity: Topic complexity from analysis step ("simple", "moderate", "complex")

    Returns:
        List of dicts with 'part_name' and 'query' keys
    """
    print("   🔧 Breaking query into focused sub-parts...")

    # Load prompt from file
    system_prompt = load_prompt("prod_query_breakdown", current_date_str=current_date_str, complexity=complexity)
    
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
def call_ddgs_single(query: str, max_results: int = max_ddg_results) -> str:
    """
    Search with DDGS for URLs, then extract full content using trafilatura.
    
    Args:
        query: The search query
        max_results: Number of results to fetch and extract
        
    Returns:
        Combined extracted content from search result pages
    """
    try:
        # Step 1: Get URLs from DDGS
        results = DDGS().text(query, max_results=max_results)
        
        if not results:
            return "No search results found"
        
        # Step 2: Extract full content from each URL using trafilatura
        combined = []
        successful_extractions = 0
        
        for result in results:
            title = result.get('title', 'No title')
            href = result.get('href', '')
            
            if not href:
                continue
                
            try:
                # Fetch the page
                downloaded = trafilatura.fetch_url(href)
                
                if downloaded:
                    # Extract content with tables included
                    content = trafilatura.extract(
                        downloaded,
                        include_tables=True,
                        include_comments=False,
                        include_images=False,  # Just URLs, not useful without vision
                        favor_recall=True,  # Get more content
                        deduplicate=True
                    )
                    
                    if content and len(content.strip()) > 100:
                        # Truncate if necessary
                        if len(content) > max_chars_per_agent:
                            content = content[:max_chars_per_agent] + "\n\n[Content truncated...]"
                        
                        combined.append(f"## {title}\n**Source:** {href}\n\n{content}")
                        successful_extractions += 1
                        
            except Exception as e:
                # Skip failed extractions silently, continue with others
                continue
        
        if not combined:
            return "No content could be extracted from search results"
        
        return f"*Extracted content from {successful_extractions} sources:*\n\n" + "\n\n---\n\n".join(combined)
        
    except Exception as e:
        return f"DDGS/trafilatura error: {str(e)}"


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


def call_web_for_context_multipart(topic: str, is_time_sensitive: bool, current_date_str: str, complexity: str = "moderate") -> tuple[str, str]:
    """
    Fetch web context using multiple search calls for comprehensive coverage.

    Breaks the topic into multiple sub-queries, executes each separately,
    and combines the results. Uses Tavily or DDGS based on TEST_MODE.

    Args:
        topic: The main topic/query
        is_time_sensitive: Whether to add date context to queries
        current_date_str: Current date string
        complexity: Topic complexity from analysis step

    Returns:
        Tuple of (combined_context, breakdown_summary)
    """
    # Step 1: Break query into parts
    query_parts = break_query_into_parts(topic, current_date_str, complexity=complexity)
    
    # Determine search provider
    search_provider = "DDGS + trafilatura (Test Mode)" if TEST_MODE else "Tavily"
    
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
    
    # Step 3: Prepare queries (add date context if needed)
    prepared_queries = []
    for part in query_parts:
        query = part['query']
        if is_time_sensitive:
            query = f"{query} as of {current_date_str}"
        prepared_queries.append({'part_name': part['part_name'], 'query': query})

    # Step 4: Execute queries in parallel
    print(f"\n   📡 Executing {len(prepared_queries)} {search_provider} calls (max {max_parallel_web_calls} in parallel)...")
    print("   " + "=" * 50)

    def _fetch_one(idx_and_query):
        idx, pq = idx_and_query
        part_name = pq['part_name']
        query = pq['query']
        try:
            result = call_web_single(query)
            return {'index': idx, 'part_name': part_name, 'query': query, 'result': result, 'char_count': len(result), 'error': None}
        except Exception as e:
            return {'index': idx, 'part_name': part_name, 'query': query, 'result': f"Error: {str(e)}", 'char_count': 0, 'error': str(e)}

    all_results = [None] * len(prepared_queries)
    with ThreadPoolExecutor(max_workers=max_parallel_web_calls) as executor:
        futures = {executor.submit(_fetch_one, (i, pq)): i for i, pq in enumerate(prepared_queries)}
        for future in as_completed(futures):
            r = future.result()
            idx = r['index']
            all_results[idx] = r
            if r['error']:
                print(f"   [{idx + 1}/{len(prepared_queries)}] ❌ {r['part_name']}: {r['error']}")
            else:
                print(f"   [{idx + 1}/{len(prepared_queries)}] ✅ {r['part_name']}: {r['char_count']} chars")

    # Build combined context and update breakdown summary (in original order)
    combined_context = ""
    for r in all_results:
        if r['error']:
            breakdown_summary += f"**Result:** FAILED - {r['error']}\n"
        else:
            combined_context += f"\n\n## {r['part_name']}\n\n"
            combined_context += f"**Query:** {r['query']}\n\n"
            combined_context += r['result']
            breakdown_summary += f"**Result:** {r['char_count']} chars retrieved\n"
    
    print("\n   " + "=" * 50)
    
    # Step 5: Summary statistics
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
    individual_results_content = f"# Individual {search_provider} Results\n\n"
    for i, r in enumerate(all_results, 1):
        individual_results_content += f"---\n\n## Part {i}: {r['part_name']}\n\n"
        individual_results_content += f"**Query:** {r['query']}\n\n"
        individual_results_content += f"**Characters:** {r['char_count']}\n\n"
        individual_results_content += f"### Result:\n\n{r['result']}\n\n"
    
    save_to_working("web_individual_results", individual_results_content)
    
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

        # Override if force flag is set
        if force_web_search and not needs_search:
            needs_search = True
            reason = f"Forced by force_web_search flag (LLM said no: {reason})"

        print(" Done!")
        print(f"   📊 Needs web search: {needs_search}{' (forced)' if force_web_search else ''}")
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
                    current_date_str=current_date_str,
                    complexity=complexity
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
                save_to_working("web_context", f"# Web Search {STATUS_FAILED}\n\nError: {str(e)}\n\nProceeding without web context.")
                save_to_working("tavily_breakdown", f"# Breakdown Failed\n\nError: {str(e)}")
        else:
            print("   ⏭️  Skipping web search (not needed)")
            # Save empty context file to indicate analysis was done
            save_to_working("web_context", f"# {STATUS_NO_CONTEXT}\n\nThe topic does not require current web information.")
            save_to_working("tavily_breakdown", f"# {STATUS_NO_BREAKDOWN}\n\nWeb search was not required for this topic.")
        
        return {"context_complete": True}
    
    except json.JSONDecodeError as e:
        print(f" Warning: Could not parse analysis response: {e}")
        print("   ⏭️  Skipping web search (analysis failed)")
        save_to_working("web_context", f"# Analysis {STATUS_FAILED}\n\nCould not determine if web search was needed. Proceeding without web context.")
        save_to_working("tavily_breakdown", f"# Analysis {STATUS_FAILED}\n\nCould not parse LLM response.")
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
    web_context, has_web_context = load_working_if_usable("web_context", [STATUS_NO_CONTEXT, STATUS_FAILED])

    # Also load the breakdown info for reference
    tavily_breakdown, has_breakdown = load_working_if_usable("tavily_breakdown", [STATUS_NO_BREAKDOWN, STATUS_FAILED])
    
    if has_web_context:
        print(f"   📚 Using web context ({len(web_context)} chars)")
        # Chunk and summarize if web context exceeds context window
        web_context = chunk_and_summarize(web_context, "web research context for outline generation")
        print(f"   📚 Web context after processing: {len(web_context)} chars")
    if has_breakdown:
        print(f"   📋 Query breakdown available for reference")

    print("   Working...", end="", flush=True)

    try:
        # Load base prompt from file
        system_prompt = load_prompt("prod_outline")

        # Inject writing style instruction
        style_info = WRITING_STYLES.get(state.writing_style, WRITING_STYLES[DEFAULT_WRITING_STYLE])
        system_prompt += f"\n\nWRITING STYLE: {style_info['name']}\n{style_info['prompt_instruction']}\n"

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
    
    # Load web research — prefer individual results (more structured) over combined web_context
    # Both contain the same underlying data, so we only use one to avoid redundant summarization
    web_context, has_web_context = load_working_if_usable("web_individual_results", [STATUS_FAILED])
    if not has_web_context:
        # Fall back to combined web_context if individual results aren't available
        web_context, has_web_context = load_working_if_usable("web_context", [STATUS_NO_CONTEXT, STATUS_FAILED])

    print("\n✍️  WRITING AGENT")
    print(f"   Job ID: {JOB_ID}")
    print(f"   Loaded outline ({len(outline)} chars) from file")

    # Budget: reserve half the context window for output + system prompt + continuations
    content_budget = context_window_chars // 2

    if has_web_context:
        # Split budget: outline 30%, web research 70%
        outline_budget = content_budget * 3 // 10
        web_budget = content_budget * 7 // 10
        print(f"   📊 Content budget: {content_budget} chars (outline: {outline_budget}, web: {web_budget})")
    else:
        # No web research — outline gets the full budget
        outline_budget = content_budget
        print(f"   📊 Content budget: {content_budget} chars (all for outline, no web research)")

    # Chunk and summarize each content piece to fit its budget
    outline = chunk_and_summarize(outline, "article outline for writing the draft", max_chars=outline_budget)
    print(f"   📋 Outline after processing: {len(outline)} chars")

    if has_web_context:
        search_label = "DDGS/trafilatura" if TEST_MODE else "Tavily"
        print(f"   📚 Loading web research ({len(web_context)} chars)")
        web_context = chunk_and_summarize(web_context, "web research context for article writing", max_chars=web_budget)
        print(f"   📚 Web research after processing: {len(web_context)} chars")

    print("   Writing...", end="", flush=True)
    
    try:
        # Load base prompt from file
        system_prompt = load_prompt("prod_writer")

        # Inject writing style instruction
        style_info = WRITING_STYLES.get(state.writing_style, WRITING_STYLES[DEFAULT_WRITING_STYLE])
        system_prompt += f"\n\nWRITING STYLE: {style_info['name']}\n{style_info['prompt_instruction']}\n"

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

        # Include reference list so the writer can attribute facts to sources
        if _collected_references:
            ref_list = "\n".join(f"- {ref['title']} — {ref['url']}" for ref in _collected_references)
            user_content += f"\n\n## Source Reference List (for attribution of high-impact facts)\n\n{ref_list}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        
        # Stream with continuation support for long articles
        full_response, finish_reason = stream_with_continuation(
            messages, 
            max_continuations=3,
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
    print(f"   Loaded draft ({word_count} words, {len(draft)} chars) from file")

    # Chunk and summarize if draft exceeds context window
    draft = chunk_and_summarize(draft, "article draft for editing and improvement")
    print(f"   📝 Draft after processing: {len(draft)} chars")

    print("   Editing...", end="", flush=True)
    
    try:
        # Load prompt from file
        system_prompt = load_prompt("prod_editor")

        # Inject writing style instruction
        style_info = WRITING_STYLES.get(state.writing_style, WRITING_STYLES[DEFAULT_WRITING_STYLE])
        system_prompt += f"\n\nWRITING STYLE: {style_info['name']}\n{style_info['prompt_instruction']}\n"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Edit and improve this draft (preserve full length):\n\n{draft}")
        ]
        
        # Stream with continuation support
        full_response, finish_reason = stream_with_continuation(
            messages,
            max_continuations=3,
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

    if use_draft_as_final:
        print("\n⏭️  Skipping editor (use_draft_as_final = True)")
        # Copy draft to final_draft so downstream output logic finds it
        draft = load_from_working("draft")
        if draft:
            save_to_working("final_draft", draft)
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


def _interactive_choice(label: str, options: dict, default_key: str) -> str:
    """Generic interactive chooser for style/level/source options."""
    print(f"\n" + "-" * 60)
    print(f"📝 {label}:")
    print("-" * 60)
    for key, opt in options.items():
        print(f"   {key}) {opt['name']} — {opt.get('description', '')}")
    print()

    default_name = options[default_key]['name']
    try:
        choice = input(f"Choice [{default_key} = {default_name}]: ").strip()
        if choice in options:
            print(f"   ✅ Selected: {options[choice]['name']}")
            return choice
        elif not choice:
            print(f"   ✅ Using default: {default_name}")
            return default_key
        else:
            print(f"   ⚠️  Invalid choice '{choice}', using default: {default_name}")
            return default_key
    except (EOFError, KeyboardInterrupt):
        print(f"\n   Using default: {default_name}")
        return default_key


def get_topic_interactively() -> tuple[str, str, str, str]:
    """Prompt the user for topic, writing style, research level, and source level."""
    print("\n" + "=" * 60)
    print("📝 WRITING ASSISTANT - Interactive Mode")
    print("=" * 60)
    print("\nEnter the topic you'd like to write about.")
    print("(Press Enter for default topic)\n")

    default_topic = "The Future of Artificial Intelligence in Healthcare: Opportunities, Challenges, and Ethical Considerations"

    try:
        user_input = input(f"Topic [{default_topic[:50]}...]: ").strip()
        topic = user_input if user_input else default_topic
    except (EOFError, KeyboardInterrupt):
        print("\nUsing default topic...")
        topic = default_topic

    style = _interactive_choice("Select a writing style", WRITING_STYLES, DEFAULT_WRITING_STYLE)
    research = _interactive_choice("Select research depth", RESEARCH_LEVELS, DEFAULT_RESEARCH_LEVEL)
    sources = _interactive_choice("Select source count", SOURCE_LEVELS, DEFAULT_SOURCE_LEVEL)
    return topic, style, research, sources


# =============================================================================
# Main Execution
# =============================================================================

def run_writing_assistant(
    topic: str,
    model: str = llm_model,
    temperature: float = llm_temperature,
    writing_style: str = DEFAULT_WRITING_STYLE,
    research_level: str = DEFAULT_RESEARCH_LEVEL,
    source_level: str = DEFAULT_SOURCE_LEVEL
) -> dict:
    """
    Run the complete writing assistant pipeline.

    Args:
        topic: The topic to write about
        model: OpenAI model to use
        temperature: Creativity level (0-1)
        writing_style: Key from WRITING_STYLES dict
        research_level: Key from RESEARCH_LEVELS dict
        source_level: Key from SOURCE_LEVELS dict

    Returns:
        Dict with completion status and file paths
    """
    global llm, JOB_ID, max_chars_per_agent, max_ddg_results

    # Apply research and source level settings
    max_chars_per_agent = RESEARCH_LEVELS.get(research_level, RESEARCH_LEVELS[DEFAULT_RESEARCH_LEVEL])["max_chars_per_source"]
    max_ddg_results = SOURCE_LEVELS.get(source_level, SOURCE_LEVELS[DEFAULT_SOURCE_LEVEL])["results_per_query"]

    # Generate unique job ID for this run
    JOB_ID = generate_job_id()

    # Reset references collector for this run
    _collected_references.clear()
    
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
    style_info = WRITING_STYLES.get(writing_style, WRITING_STYLES[DEFAULT_WRITING_STYLE])
    research_info = RESEARCH_LEVELS.get(research_level, RESEARCH_LEVELS[DEFAULT_RESEARCH_LEVEL])
    source_info = SOURCE_LEVELS.get(source_level, SOURCE_LEVELS[DEFAULT_SOURCE_LEVEL])
    print(f"Topic: {topic}")
    print(f"✍️  Writing Style: {style_info['name']}")
    print(f"🔬 Research Depth: {research_info['name']} ({max_chars_per_agent} chars/source)")
    print(f"📚 Source Count: {source_info['name']} ({max_ddg_results} per sub-query)")
    if use_draft_as_final:
        print(f"⏩ Draft as Final: ENABLED (editor agent will be skipped)")
    if force_web_search:
        print(f"🔍 Force Web Search: ENABLED (will always search regardless of LLM analysis)")
    print(f"🆕 Multi-Part Web Search: ENABLED (up to 10 sub-queries)")
    print(f"🆕 External Prompts: ENABLED")
    if TEST_MODE:
        print(f"🧪 TEST MODE: ENABLED (using DDGS + trafilatura instead of Tavily)")
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
    initial_state = WritingState(topic=topic, job_id=JOB_ID, writing_style=writing_style)
    
    # Save initial state info
    search_provider = "DDGS + trafilatura (Test Mode)" if TEST_MODE else "Tavily"
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
    
    # Determine topic and settings based on run mode
    writing_style = DEFAULT_WRITING_STYLE
    research_level = DEFAULT_RESEARCH_LEVEL
    source_level = DEFAULT_SOURCE_LEVEL
    if is_interactive_mode():
        topic, writing_style, research_level, source_level = get_topic_interactively()
    elif len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = "The Future of Artificial Intelligence in Healthcare: Opportunities, Challenges, and Ethical Considerations"

    try:
        result = run_writing_assistant(
            topic=topic,
            model=llm_model,
            temperature=llm_temperature,
            writing_style=writing_style,
            research_level=research_level,
            source_level=source_level
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
            
            # Append collected references
            references_section = format_references()
            if references_section:
                print(f"\n📎 Appending {len(_collected_references)} references to output")
                save_to_working("references", references_section)

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
            if references_section:
                full_output += f"""
---

{references_section}
"""

            output_path = save_final_output(full_output, "final_article.md")
            print(f"\n   📁 Output saved to: {output_path}")

            # Save a separate clean draft only if the editor ran (otherwise it's the same content)
            if not use_draft_as_final:
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
