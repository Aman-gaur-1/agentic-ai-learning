import os
from http.client import responses

import requests

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY")


@tool
def search_job(query: str):

    """
    Search for job posting based on the user's query.
    """
    response = requests.get(


        url=  "https://api.openwebninja.com/jsearch/search-v2",

        headers={"X-API-KEY": JSEARCH_API_KEY},
        params = {
            "query": query,
            "country": 'in',
            "date_posted": "today"

    })


    response.raise_for_status()

    return response.json()


# print(search_job.invoke({"query":"Python Developer"}))

llm = ChatNVIDIA(

    model= "meta/llama-3.1-8b-instruct"
)


tools = [search_job]

agent  = create_agent(
    model= llm,
    tools= tools
)

user_query = agent.invoke(
    {

        "messages": [
            HumanMessage(content= """Find data analyst jobs in india. For each job provide:
                
            - Job Title
            - Company Name
            - Location
            - Job Type
            - Posted Time
            - Apply Link"""
        )
    ]
}
)



print(user_query["messages"][-1].content)







