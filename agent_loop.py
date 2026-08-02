from dotenv import load_dotenv
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable


load_dotenv()

max_iteration = 10
model = "meta/llama-3.1-70b-instruct"


catalog_df = pd.read_csv("catalog.csv")


product_df = catalog_df[catalog_df["record_type"] == "product"][
    ["product","price","category"]
].reset_index(drop =True)

# print(product_df)

discount_df = catalog_df[catalog_df["record_type"]  == "discount"][
    ["tier","percentage"]
    ].reset_index(drop= True)

# print(discount_df)




@tool
def get_product_price(product:str)-> float:
    """Look up the price of a product in the catalog (read from catalog.csv)"""
    print(f"---->  Executing get_product_price( product = {product})")

    row = product_df[product_df["product"].str.lower() == product.lower()]

    if row.empty:
        return 0.0

    return float(row.iloc[0]["price"])


# # print(get_product_price("Laptop"))

@tool
def apply_discount(price:float,discount_tier:str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers come from catalog.csv (e.g. bronze, silver, gold)"""

    print(f"Executing appy_discount(price = {price}, discount_tier= {discount_tier})")

    row = discount_df[discount_df["tier"].str.lower() == discount_tier.lower()]

    if row.empty:
        discount = 0

    else:
        discount = float(row.iloc[0]["percentage"])

    return round(price *  (1 - discount /100),2)
#
#
# # print(apply_discount(1299.99,"gold"))
#

@traceable(name = "agentic_ai_learning")
def run_agent(question:str):
    tools = [get_product_price,apply_discount]

    tool_dict = {t.name: t for t in tools}

    llm = init_chat_model(f'nvidia:{model}',temperature = 0,timeout = 120,model_kwargs={'max_retries': 2})


    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content= (
            "you are a helpful shopping assistant with two tools:"
            "get_product_price and apply_discount.\n\n"
            "RULES:\n"
            "1. Never guess a price - always call get_product_price first.\n"
            "2. Only call apply_discount after get_product_price return a value."
            "pass that exact number, naver a made-up or truncated one.\n"
            "3. Never do discount math yourself - always use apply_discount\n"
            "4. If no discount tier is given, ask the user which tier to use.\n"
            "5. Always use real tool calls - never write a tool call as plain"
            ""
            "text in your reply.\n\n"
            "Example: get_product_price(\"Laptop\") -> 1299.99"
            "--> apply_discount(price =1299.99, discount_tier= \"gold\") --> final price"
            "reply with that number"
        )

        ),HumanMessage(content=question),
    ]

    for iteration in range(1, max_iteration + 1):
        print(f'\n --- Iteration {iteration}---')

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            return ai_message.content


        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Selected] {tool_name}: with args: {tool_args}")

        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f'Tool {tool_name} Not Found')


        try:
            observation = tool_to_use.invoke(tool_args)
            print(f'[Tool Result]: {observation}')

        except Exception as e:
            observation = (
                f'Error: {e}'
                f'Reminder: call get_product_price first to get a real numeric.'
                f'price, then pass that excat number into apply_discount'

            )
            print(f'[Tool Error] {observation}')


        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id= tool_call_id)
        )


# # print(run_agent('"What’s the price of a laptop, after a gold discount?" '))

    print("ERROR: Max iteration reached without a final answer")
    return None


if __name__ == "__main__":
    print('Hello langchain agent')
    print()

    result = run_agent("What is the the price of a laptop after applying a gold discount?")
    print(result)





