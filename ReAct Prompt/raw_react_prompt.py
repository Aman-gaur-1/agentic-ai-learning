from dotenv import load_dotenv
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.messages import  HumanMessage
from langsmith import traceable
import inspect
import re


load_dotenv()

max_iteration = 10
# model = "meta/llama-3.1-70b-instruct"
model = "meta/llama-3.1-8b-instruct"


catalog_df = pd.read_csv("catalog.csv")


product_df = catalog_df[catalog_df["record_type"] == "product"][
    ["product","price","category"]
].reset_index(drop =True)

# print(product_df)

discount_df = catalog_df[catalog_df["record_type"]  == "discount"][
    ["tier","percentage"]
    ].reset_index(drop= True)

# print(discount_df)





def get_product_price(product:str)-> float:
    """Look up the price of a product in the catalog (read from catalog.csv)"""
    print(f"---->  Executing get_product_price( product = {product})")

    row = product_df[product_df["product"].str.lower() == product.lower()]

    if row.empty:
        return 0.0

    return float(row.iloc[0]["price"])





def apply_discount(price:float,discount_tier:str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers come from catalog.csv (e.g. bronze, silver, gold)"""

    price = float(price)

    print(f"Executing appy_discount(price = {price}, discount_tier= {discount_tier})")

    row = discount_df[discount_df["tier"].str.lower() == discount_tier.lower()]

    if row.empty:
        discount = 0

    else:
        discount = float(row.iloc[0]["percentage"])

    return round(price *  (1 - discount /100),2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}


def get_tool_descriptions(tools_dict):
    descriptions = []

    for tool_name, tool_function in tools_dict.items():
        signature = inspect.signature(tool_function)
        docstring = inspect.getdoc(tool_function)  or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)

tool_names = ", ".join(tools.keys())





llm = init_chat_model(
    f"nvidia:{model}",
    temperature = 0,
    timeout = 120,
)

llm_with_stop = llm.bind(stop = ["\nObservation","Observation:","\nQuestion:"])

react_prompt = """
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:"""



@traceable(name = "Raw ReAct Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    prompt = react_prompt.format(
        tool_descriptions = tool_descriptions,
        tool_names = tool_names,
        question = question,
    )

    scratchpad = ""

    for iteration in range(1, max_iteration + 1):
        print(f"\n---- Iteration {iteration} ----")
        full_prompt = prompt + scratchpad

        response = llm_with_stop.invoke([HumanMessage(content=full_prompt)])
        output = response.content
        print(f"LLM Output:\n{output}")


        final_answer_match = re.search(r"Final Answer:\s*(.+)",output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer


        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)",output)

        if not action_match or not action_input_match:
            print(" [Parsing] ERROR: Could not Parse Action/Action Input" )
            break


        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"   [Tool Selected] {tool_name} with args: {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=",1)[-1].strip().strip("'\'") for x in raw_args]


        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' Not Found."

        else:
            try:
                observation = str(tools[tool_name](*args))

            except Exception as e:
                observation = (f"Error {e}. Call get_product_price first to get a real numeric"
                               f"price, then pass the exact number into apply_discount.")


        print(f"  [Tool Result] {observation}")

        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("Error: Max Iterations reached without a final answer")

    return None


if __name__ == "__main__":
    print("Raw ReAct Agent - No Function Calling")
    print()
    result = run_agent("What is the price of a laptop after applying a  glod discount")
    print(result)













