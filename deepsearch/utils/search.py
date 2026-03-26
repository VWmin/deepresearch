import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv(verbose=True)

def create_search_client() -> TavilyClient:
    return TavilyClient(api_key=os.getenv("TAIL_API_KEY"))
