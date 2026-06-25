from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

response = llm.invoke("Explain agentic AI in one simple line")

print(response.content)


