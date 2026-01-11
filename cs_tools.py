# To install: pip install tavily-python

from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv() # Load environment variables from a .env file if present

def call_web_for_context(query: str) -> str:
    client = TavilyClient()
    response = client.search(
        query="query: " + query + "\nProvide detailed and relevant information from recent web sources to answer the query.",
        include_answer="advanced",
        search_depth="advanced"
    )   
    return response["answer"]



test_query = "Give me a bullet list with sub bullet points if necessary of common nlp manipulation techniques and a quick brief description of each."
print(call_web_for_context(test_query))