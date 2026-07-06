# from dotenv import load_dotenv
#
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
#
# load_dotenv()
#
# llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
#
# response = llm.invoke("Explain agentic AI in one simple line")
#
# print(response.content)


from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

prompt_value = template.invoke(
    {
        "name": "Bob",
        "user_input": "What is your name?",
    }
)
# Output:
# ChatPromptValue(
#    messages=[
#        SystemMessage(content='You are a helpful AI bot. Your name is Bob.'),
#        HumanMessage(content='Hello, how are you doing?'),
#        AIMessage(content="I'm doing well, thanks!"),
#        HumanMessage(content='What is your name?')
#    ]
# )