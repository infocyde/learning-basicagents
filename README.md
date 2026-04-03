# learning-basicagents

## What is this

This demonstrates the use of some really basic langgraph "agents". The advanced version does web pulling to add context and more relevant information (see below). It runs in your IDE's terminal window or can be called by another process with the prompt being passed in as the first parameter.

A video explaining this Repo and a demo running the advanced features can be found here-
<https://www.youtube.com/watch?v=G-Lrn0Li60U>

## Scripts

### hello_world_langgraph.py

A simple two-agent setup that runs one at a time. This could easily be handled without langgraph — it's just a learning example.

### local_agent.py

The main writing assistant. A multi-agent pipeline that researches a topic, outlines an article, writes a draft, and optionally edits it into a final version. All agents read/write intermediate files from a `wip/` directory and final output goes to `output/`.

**Pipeline flow:** Extra Context (web research) -> Outliner -> Writer -> Editor (optional)

**Key features:**

- **Web research:** Breaks your topic into sub-queries and searches via DuckDuckGo + trafilatura (free, `TEST_MODE = True`) or Tavily API (`TEST_MODE = False`). Web calls run in parallel for speed.
- **Research depth control:** Choose Light (6K chars/source), Medium (12K), or Deep (20K) to balance quality vs. processing time. Source count is also configurable (Few/Average/Many results per sub-query).
- **Chunk & summarize:** If content exceeds the model's context window, it's automatically chunked at section/paragraph boundaries and recursively summarized. Chunk summarization can run in parallel.
- **Citation preservation:** Source URLs are extracted before summarization and appended as a references section to the final article, so citations survive the summarization process.
- **Writing style selection:** Choose between Technical Report, Blog Post, or Narrative/Story — the style is injected into outline, writer, and editor agent prompts.
- **Externalized prompts:** All agent system prompts live in `prompts/prod_*.txt` files so you can tune behavior without touching code.
- **Continuation logic:** Long LLM responses are automatically detected and continued across multiple calls.
- **Draft-as-final mode:** Set `use_draft_as_final = True` to skip the editor agent and save tokens/time.
- **Force web search:** Set `force_web_search = True` to always search the web, useful for smaller models that incorrectly decide they don't need to search.

**Interactive mode prompts:**

When run interactively, the script asks four questions before starting:

1. **Topic** — what to write about
2. **Writing style** — Technical Report / Blog Post / Narrative Story
3. **Research depth** — Light / Medium / Deep (controls how much content is pulled per web source)
4. **Source count** — Few / Average / Many (controls how many web pages are scraped per sub-query)

**Configuration (constants at top of file):**

| Constant | What it controls |
| --- | --- |
| `context_window_tokens` | Your model's context window in tokens — derived constants calculate from this |
| `chars_per_token` | Character-to-token ratio (default 4, conservative) |
| `llm_model` | Model name passed to the OpenAI-compatible API |
| `llm_temperature` | Creativity level (0.0-1.0) |
| `TEST_MODE` | `True` = free DDGS search, `False` = Tavily API |
| `use_draft_as_final` | `True` = skip editor agent, use draft as final output |
| `force_web_search` | `True` = always search the web, override LLM analysis |
| `max_parallel_summarizations` | Parallel chunk summarization calls (1 recommended for local LLMs) |
| `max_parallel_web_calls` | Parallel web scrape calls (3 default, I/O-bound so safe to go higher) |
| `DEFAULT_RESEARCH_LEVEL` | Default research depth: "1" Light, "2" Medium, "3" Deep |
| `DEFAULT_SOURCE_LEVEL` | Default source count: "1" Few, "2" Average, "3" Many |

**Usage:**

```bash
# Interactive mode (prompts for topic, style, research depth, source count)
python local_agent.py

# Pass topic as argument (uses default settings for style/research/sources)
python local_agent.py "Your topic here"
```

**Output files:**

- `output/{job_id}_final_article.md` — The final article with metadata and references
- `output/{job_id}_article_draft.md` — Separate draft (only when editor runs, i.e. `use_draft_as_final = False`)
- `wip/{job_id}_*.md` — Intermediate files (outline, draft, web context, references, etc.)

## Requirements

In order to use this-

- You must have Python installed, and know how to use pip and/or uv.
- You must have an OpenAI key, or be advanced enough to use a compatible OpenAI API with the model of your choice. `local_agent.py` defaults to `base_url="http://localhost:1234/v1"` for local LLM servers (e.g., LM Studio).
- Optionally you can use Tavily.com's API to pull content from the web for context. Set `TEST_MODE = False` in the script and make sure you have a `TAVILY_API_KEY` in your `.env` file.

**Setup:**

1. Clone the repo
2. Create a venv: `uv venv --python 3.11`
3. Activate it: `.venv\scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac)
4. Install dependencies: `uv pip install -r requirements.txt`
5. Run from IDE or terminal

## Future

- Async sub-agent demos
- Additional output format options
- Database/RAG integration examples

## Disclosures

I vibe coded much of this based on an outdated tutorial for the hello world version. Then I took past LLM workflows as a template that I had coded without langgraph. I directed Claude to do most of the grunt work in recreating the patterns I had worked out before. In production code I'd probably break out the file saves/reads and the LLM call into separate modules to promote reuse.

So here you are, base code. Add RAG, async, database support, whatever floats your boat. Have fun.

To be honest you can replicate this quickly with GUI tools like n8n. But what you learn here will apply to scenarios where n8n will have trouble following. YMMV.

And as always, remember LLM-based libraries and Python itself change all the time. Everything is working as of the time of this post, but if you pull this repo later you might have to tweak a few lines — but by then the AI will have your back :)
