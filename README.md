# learning-basicagents

## What is this

This demonstrates the user of some really basic langgraph "agents".  The advanced version does web pulling to add context, more relevant information (see below).  It runs in your IDE's terminal window or can be called by another process with the prompt being passed in as the first parameter.

## Requirements

In order to use this-

* You must have python installed, and now how to use pip / and or uv.
* You must have an OpenAI key, or be advanced enough to use a compatible OpenAI API with the model of your choice and be able to hack a url in the LLM call in the code. You can use any OpenAI model that you wish.
* Optionally you can use Tavity.com's API to pull content from the web for context in the advanced example. Set the TEST_MODE value to FALSE in the script and make sure you have a Tavity API key in your .env file

1) Clone the repo
2) Go into the folder of the clone, create a venv virtual enviornment (uv gives you control over which python interpreter to use)
    example command: uv venv --python 3.11
3) When done (fast in uv, slow if you use pip)
    .venv\scripts\activate (or do / if linux/mac)
4) With your virtual environment active, install the requirements (if you don't have the .venv active, you will install all the requirements globally, don't do that)
    uv pip install -r requirements.txt
5) You can then run the script out of the IDE or typeing run and the script name into the termainal.

## Two langgraph examples included-

* hello_world_langgraph.py - A simple two agent setup that run one at a time.  This could easily be handled without langgraph.
* advanced_langgraph_sync_agent - this one uses prompts that are read dynamically out of files instead of in state, has more agents, has retries and better error handling, and has the option to add additional near live into via web scraping via DuckDuckGo + trafilatura (if hardcoded TEST_MODE = True) of Tavily if TEST_MODE = False and you have a Tavily API key in your .env with the standard naming conventions.

## Future

Nothing rocket science here, future plans are-

* adding a few options for what type of activity you are doing (research, blog post, indepth summary, bullet point quick hits), temperature (creative, accurate, balanced) and writing styles of some sort.
* will do some sub agent async demo in the near future.  I actually have a paying gig that I'm about to work on for that so it will be back burnered till I can get to it

## Disclosures

I vibe coded much of this based on an outdated tutorial for the hello world version. Then I took past LLM work flows as a template that I had coded without langgraph that I'd done on the past. I just directed Claude Opus 4.5 to do most of the grunt work in recreating the patterns I had worked out before.  In production code I'd probably break out the file saves and reads and the LLM call into a seperate script to promote that reuse in similar projects.  

So here you are, base code.  Add rag, async, database support, whatever floats your boat.  Have fun.

To be honest you can replicate this quickly with GUI tools like n8n.  But what you learn here will apply to scenarios where n8n will have trouble following.  YMMV.

And as always, remember LLM based libraries and Python itself change all the time.  Everything is working as of the time of this post, but you pull this repo in 2027 and beyond you might have to tweak a few lines, but by then the AI will have your back :)
