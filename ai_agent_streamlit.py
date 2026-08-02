import os
import requests

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Load Environment Variables
load_dotenv()

# Read API Key from .env
JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY")
# JSEARCH_API_KEY = "JSEARCH_API_KEY"


# -----------------------------
# Custom Tool
# -----------------------------
@tool
def search_jobs(query: str):
    """
    Search for job postings based on the user's query.
    """

    response = requests.get(
        "https://api.openwebninja.com/jsearch/search-v2",
        headers={"X-API-Key": JSEARCH_API_KEY},
        params={
            "query": query,
            "country": "in",
            "date_posted": "today"
        }
    )

    response.raise_for_status()

    return response.json()

# For Testing the Function
# print(search_jobs.invoke({"query": "Python Developer"}))



# Create LLM
# -----------------------------
llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct"
)


# -----------------------------
# Register Tools
# -----------------------------
tools = [search_jobs]


# -----------------------------
# Create AI Job Search Agent
# -----------------------------
agent = create_agent(
    model=llm,
    tools=tools
)


# -----------------------------
# User Query
# -----------------------------
result = agent.invoke(
    {
        "messages": [
            HumanMessage(content=""" Find Data Analyst jobs in India. For each job provide:

- Job Title
- Company Name
- Location
- Job Type
- Posted Date
- Apply Link
"""
            )
        ]
    }
)


# -----------------------------
# Print Final Response
# -----------------------------
print(result["messages"][-1].content)