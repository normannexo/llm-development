from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from dotenv import load_dotenv
load_dotenv()

llm = ChatMistralAI(
    model="mistral-medium-latest",
    temperature=0,
    max_retries=2,
)

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

tools = [get_weather, tavily_search_tool]
model =llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def custom_tools_condition(state):  
    # Get the last message from the state
    if isinstance(state, dict) and "messages" in state:
        messages = state["messages"]
        if messages and hasattr(messages[-1], "tool_calls"):
            return "tools"
    return "model"

def model_node(state: State) -> State:
    res = model.invoke(state["messages"])
    return {"messages": res}

def orchestrator_node(state: State) -> State:
    return {"messages": state["messages"]}

def done_condition(state):
    # Check if the last message indicates we're done
    if isinstance(state, dict) and "messages" in state:
        messages = state["messages"]
        if messages and messages[-1].content.strip().lower() == "end":
            return "__end__"
    return "model"

builder = StateGraph(State)
builder.add_node("model", model_node)
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", done_condition)
builder.add_conditional_edges("model", custom_tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()
print(graph.get_graph().draw_ascii())
