
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType, Tool
from langchain.tools import AIPluginTool
from langchain import SerpAPIWrapper, LLMChain, OpenAI
from langchain import LLMChain, PromptTemplate
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

# klarna_tool = AIPluginTool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")
# shop_app_tool = AIPluginTool.from_plugin_url("https://server.shop.app/.well-known/ai-plugin.json")
# shop_app_tool = AIPluginTool.from_plugin_url("https://server.shop.app/openai/v1/api.json")
search = SerpAPIWrapper()
todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
)
todo_chain = LLMChain(
    llm=OpenAI(temperature=0), 
    prompt=todo_prompt
)

def parse_response(response):
    print(response)
    return response

# llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k')
# tools = load_tools(["requests_all"])
tools = []
tools += [
    # klarna_tool, 
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
    # shop_app_tool,
    # Tool(
    #     name="Parse Response",
    #     func=parse_response,
    #     description="useful when you need to parse the response from an API",
    # ),
]

agent_chain = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
    handle_parsing_errors=True,
)

# agent_chain.run("whats the best sweater on shopify? From a canadian company?")
# question = "whats shirts are available on klarna?"
question = "What is the best electric car on the market for under $60k?"
# prompt = """
#     Your are a shopping expert that helps customers find the best products for them.
#     When you return results / recommendations you justify them based on price, reviews, quality, and you include links so the customer can buy them.

#     Here's the question from the customer you need to answer: {question}
# """
prompt = f"""
    You are a shopping assistant that helps customers find the best products for them.

    A customer has come to you with a specific shopping-related question.
    Here it is: "{question}"

    Use the TODO tool to make a step by step plan then take actions for each item on that list.

    Use specific data / reviews to justify your final recommendation and include multiple ranked options for the customer to choose from.
"""
agent_chain.run(prompt)
# agent_chain.run("whats shirts are available? Use the Shop plugin.")