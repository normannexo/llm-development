# PUBLICATION RESEARCHER - MAP REDUCE (with Tool-Calling Agent)
import operator
import json
import os # For API keys
from typing import Annotated, List, Dict, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Send
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
#from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from langfuse.callback import CallbackHandler
from ftv_langgraph_tools import VisitWebpageTool # Added import
from dotenv import load_dotenv

load_dotenv()

langfuse_handler = CallbackHandler()    

print("INFO: This script uses ChatAnthropic and Tavily Search.")
print("Ensure ANTHROPIC_API_KEY and TAVILY_API_KEY environment variables are set.")

# --- Tool-Calling Sub-Graph Definition --- 

# get_publication_subscription_info is now a mock tool, 
# consider replacing or enhancing with a real DB or API lookup tool if needed.
def get_publication_subscription_info(publication_name: str) -> dict:
    """Gets the subscription price and features for a specific known publication. Use this first for direct lookups."""
    mock_db = {
        "The New York Times": {"price": "$17/month", "features": ["Full digital access", "NYT Cooking", "NYT Games"]},
        "The Wall Street Journal": {"price": "$38.99/month", "features": ["All digital content", "WSJ Magazine", "WSJ+"]},
        "The Guardian": {"price": "$10/month (supporter)", "features": ["Ad-free reading", "Exclusive newsletter", "Early access"]},
        "Wired": {"price": "$29.99/year", "features": ["Print + Digital", "Limited-Edition T-Shirt", "Events"]},
        "Tech Chronicle": {"price": "$5/month", "features": ["Latest tech news", "Gadget reviews", "Exclusive interviews"]}
    }
    print(f"    [Sub-Graph Tool] get_publication_subscription_info called for: {publication_name}")
    if publication_name in mock_db:
        return mock_db[publication_name]
    else:
        return {"price": "N/A", "features": [f"Data not found for '{publication_name}' via direct lookup tool. Consider web search."]}

def visit_webpage(url: str) -> str:
    """Simulates visiting a webpage and returns its mock content. Use this if a search result provides a relevant URL."""
    print(f"    [Sub-Graph Tool] visit_webpage called for: {url}")
    # In a real scenario, this would use requests.get(url).text
    if "example.com/nyt" in url:
        return "Mock content for NYT: Subscription is $17/month with various digital perks."
    elif "example.com/wsj" in url:
        return "Mock content for WSJ: Offers start at $38.99 for full digital access."
    return f"Simulated content for {url}: This page discusses various news topics and subscription models."


# class VisitWebpageInput(BaseModel):
#     url: str = Field(description="The url of the webpage to visit")
    
# class VisitWebpageTool(BaseTool):
#     name: str = "visit_webpage"
#     description: str = "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
#     args_schema: Type[BaseModel] = VisitWebpageInput
    
#     max_output_length: int = 40000
    
#     def _truncate_content(self, content: str, max_length: int) -> str:
#         """Truncate content to stay within the maximum length."""
#         if len(content) <= max_length:
#             return content
#         return (
#             content[: max_length // 2]
#             + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
#             + content[-max_length // 2 :]
#         )
    
#     def _run(self, url: str) -> str:
#         """Execute the webpage visit and return the content as markdown."""
#         try:
#             import re
#             import requests
#             from markdownify import markdownify
#             from requests.exceptions import RequestException
#         except ImportError as e:
#             raise ImportError(
#                 "You must install packages `markdownify` and `requests` to run this tool: "
#                 "for instance run `pip install markdownify requests`."
#             ) from e
#         try:
#             # Send a GET request to the URL with a 20-second timeout
#             response = requests.get(url, timeout=20)
#             response.raise_for_status()  # Raise an exception for bad status codes

#             # Convert the HTML content to Markdown
#             markdown_content = markdownify(response.text).strip()

#             # Remove multiple line breaks
#             markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

#             return self._truncate_content(markdown_content, self.max_output_length)

#         except requests.exceptions.Timeout:
#             return "The request timed out. Please try again later or check the URL."
#         except RequestException as e:
#             return f"Error fetching the webpage: {str(e)}"
#         except Exception as e:
#             return f"An unexpected error occurred: {str(e)}"

# Instantiate the NEW tool from ftv_langgraph_tools
visit_webpage_tool = VisitWebpageTool() # This now uses the imported version
# Initialize tools
# Ensure TAVILY_API_KEY is set in your environment for TavilySearchResults
tavily_tool = None
if os.getenv("TAVILY_API_KEY"):
    tavily_tool = TavilySearch(max_results=2, name="tavily_web_search") # Increased to 2 results
    print("INFO: Tavily Search tool initialized.")
else:
    print("WARNING: TAVILY_API_KEY not found. Tavily Search tool will not be available.")

all_tools = [visit_webpage_tool]
if tavily_tool:
    all_tools.append(tavily_tool)

sub_tool_node = ToolNode(all_tools)

publication_research_agent_graph = None
sub_model_with_tools = None
try:
    sub_model = ChatAnthropic(model_name="claude-3-5-sonnet-latest") 
    sub_model_with_tools = sub_model.bind_tools(all_tools)
    print("INFO: ChatAnthropic model initialized and tools bound.")
except Exception as e:
    print(f"ERROR: Could not initialize ChatAnthropic (ensure ANTHROPIC_API_KEY is set): {e}")

def sub_should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

def sub_call_model(state: MessagesState):
    if not sub_model_with_tools:
        raise ValueError("Sub-model for tool calling is not initialized. ANTHROPIC_API_KEY might be missing or model binding failed.")
    print(f"    [Sub-Graph Model] Calling LLM with messages (last: {state['messages'][-1].content[:100]}...). Total messages: {len(state['messages'])}")
    messages = state["messages"]
    response = sub_model_with_tools.invoke(messages)
    print(f"    [Sub-Graph Model] LLM Response has tool calls: {bool(response.tool_calls)}")
    return {"messages": [response]}

if sub_model_with_tools:
    sub_builder = StateGraph(MessagesState)
    sub_builder.add_node("call_model", sub_call_model)
    sub_builder.add_node("tools", sub_tool_node)
    sub_builder.add_edge(START, "call_model")
    sub_builder.add_conditional_edges("call_model", sub_should_continue, {"tools": "tools", END: END})
    sub_builder.add_edge("tools", "call_model")
    publication_research_agent_graph = sub_builder.compile()
    print("INFO: Tool-calling sub-graph compiled successfully.")
else:
    print("WARNING: Tool-calling sub-graph could not be compiled due to model or tool initialization failure.")

# --- End of Tool-Calling Sub-Graph Definition ---


# Overall state for the main graph
class OverallState(TypedDict):
    publications_to_research: List[str]
    research_results: Annotated[List[Dict[str, any]], operator.add]
    final_report: str

# State for the individual publication research node (the "map" part)
class PublicationResearchState(TypedDict):
    publication_name: str

# Node: Provides the list of publications to research
def get_publications_to_research(state: OverallState) -> Dict[str, List[str]]:
    """Provides an initial list of publications to research."""
    return {
        "publications_to_research": [
            "The New York Times",
            "The Wall Street Journal",
            "The Guardian",
            "Wired",
            "Tech Chronicle", 
            "NonExistent News", # Should trigger web search
            "Future Today Magazine" # Another one for web search
        ]
    }

# Node: Researches subscription offers for a single publication using the tool-calling agent
def research_publication_offers(state: PublicationResearchState) -> Dict[str, List[Dict[str, any]]]:
    """Researches subscription offers for a given publication using a tool-calling agent."""
    publication_name = state["publication_name"]
    print(f"\n[Main Graph Node] Researching: {publication_name}")

    if not publication_research_agent_graph:
        print(f"  Skipping research for {publication_name} as agent graph is not available.")
        return {"research_results": [{"publication": publication_name, "price": "Error", "features": ["Agent not available (check API keys)"]}]}

    # Broader prompt to encourage using available tools, including search if direct lookup fails.
    initial_message_content = (
        f"Research and find the current digital subscription offers, including price and key features, "
        f"for the publication: '{publication_name}'. "
        f"First, try a direct lookup. If that fails or provides insufficient information, "
        f"use a web search to find the official website or relevant articles. "
        f"If you find a relevant URL, you can visit it to get more details. "
        f"Compile the price and features into a structured format."
    )
    initial_message = HumanMessage(content=initial_message_content)
    
    try:
        # It's good practice to set a recursion limit for agents
        sub_graph_response_state = publication_research_agent_graph.invoke(
            {"messages": [initial_message]},
            config={"recursion_limit": 10} 
        )
    except Exception as e:
        print(f"  ERROR invoking sub-graph for {publication_name}: {e}")
        return {"research_results": [{"publication": publication_name, "price": "Error", "features": [f"Sub-graph invocation failed: {e}"]}]}

    extracted_data = None
    # Try to find the output of get_publication_subscription_info first, as it's most structured
    if sub_graph_response_state and "messages" in sub_graph_response_state:
        for msg in reversed(sub_graph_response_state["messages"]):
            if isinstance(msg, ToolMessage) and msg.name == "get_publication_subscription_info":
                try:
                    tool_output_data = json.loads(msg.content)
                    # Check if the tool found data or returned its 'N/A' message
                    if not (tool_output_data.get("price") == "N/A" and "Data not found" in "".join(tool_output_data.get("features",[]))):
                        extracted_data = {"publication": publication_name, **tool_output_data}
                        print(f"  [Main Graph Node] Extracted structured tool data (get_publication_subscription_info) for {publication_name}: {tool_output_data}")
                        break # Found definitive data from the primary tool
                except Exception as e:
                    print(f"  [Main Graph Node] Error processing get_publication_subscription_info output for {publication_name}: {e}")
                    # Don't break, let it try to find other tool messages or final LLM response
    
    # If primary tool didn't yield results, look for any other tool message or final LLM response
    if not extracted_data and sub_graph_response_state and "messages" in sub_graph_response_state:
        final_llm_message = sub_graph_response_state["messages"][-1]
        if isinstance(final_llm_message, ToolMessage): # Should ideally be an AIMessage after tool use
             # This case might happen if the graph ends on a tool call without a model summarizing it.
             # For simplicity, we'll try to parse its content if it's the last message.
            try:
                content_data = json.loads(final_llm_message.content)
                extracted_data = {"publication": publication_name, "price": content_data.get("price", "From other tool"), "features": content_data.get("features", [f"Content: {final_llm_message.content[:100]}..."])}
                print(f"  [Main Graph Node] Extracted data from last ToolMessage ({final_llm_message.name}) for {publication_name}")
            except: # If not JSON, take as a feature
                extracted_data = {"publication": publication_name, "price": "N/A", "features": [f"Final tool ({final_llm_message.name}) output: {final_llm_message.content[:100]}..."]}
                print(f"  [Main Graph Node] Using content from last ToolMessage ({final_llm_message.name}) for {publication_name}")
        elif hasattr(final_llm_message, 'content'): # AIMessage
            # This is a fallback: try to parse price/features from the final LLM text response if no tool provided structured data.
            # This is brittle and would ideally be handled by the LLM itself being prompted to summarize.
            llm_text_summary = final_llm_message.content
            extracted_data = {"publication": publication_name, "price": "From LLM summary", "features": [llm_text_summary[:200] + "..."]}
            print(f"  [Main Graph Node] Using final LLM text summary for {publication_name}")

    if extracted_data:
        return {"research_results": [extracted_data]}
    else:
        print(f"  [Main Graph Node] No usable data extracted for {publication_name}.")
        return {"research_results": [{"publication": publication_name, "price": "N/A", "features": ["Agent did not provide structured data or summary"]}]}

# Node: Aggregates all research results into a final report (the "reduce" part)
def aggregate_research_results(state: OverallState) -> Dict[str, str]:
    """Aggregates research results into a formatted table."""
    results = state["research_results"]
    if not results:
        return {"final_report": "No research data available."}

    report_lines = ["\nDigital Subscription Offers (Researched by Agent):", "-" * 50]
    report_lines.append("{:<25} | {:<15} | {}".format("Publication", "Price", "Features"))
    report_lines.append("-" * 80)

    for res in results:
        # Ensure features is a list of strings before joining
        features_list = res.get("features", [])
        if isinstance(features_list, str): # Handle case where features might be a single string
            features_list = [features_list]
        features_str = ", ".join(map(str, features_list)) 
        report_lines.append("{:<25} | {:<15} | {}".format(str(res.get("publication", "N/A")), str(res.get("price", "N/A")), features_str))
    
    return {"final_report": "\n".join(report_lines)}

# Edge Logic: Determines the next step after getting the publication list
def map_publications_to_research_nodes(state: OverallState) -> List[Send]:
    """Creates a Send object for each publication to be researched."""
    return [
        Send("research_publication_node", {"publication_name": pub_name})
        for pub_name in state["publications_to_research"]
    ]

# Construct the main graph
builder = StateGraph(OverallState)
builder.add_node("get_publications_node", get_publications_to_research)
builder.add_node("research_publication_node", research_publication_offers)
builder.add_node("aggregate_results_node", aggregate_research_results)

builder.add_edge(START, "get_publications_node")
builder.add_conditional_edges(
    "get_publications_node", 
    map_publications_to_research_nodes, 
    {"research_publication_node": "research_publication_node"} 
)
builder.add_edge("research_publication_node", "aggregate_results_node")
builder.add_edge("aggregate_results_node", END)

main_graph = builder.compile()

if __name__ == "__main__":
    print("\nRunning publication research graph with Tool-Calling Agent...")
    
    if not publication_research_agent_graph:
        print("CRITICAL ERROR: Main graph cannot run because the sub-graph (publication_research_agent_graph) failed to compile.")
        print("Please ensure ANTHROPIC_API_KEY and TAVILY_API_KEY (if used) are set correctly and required packages are installed.")
    else:
        result = main_graph.invoke({}, config={"callbacks": [langfuse_handler]})
        print("\nFinal Report from Main Graph:")
        print(result.get("final_report", "No report generated."))
        # print("\nASCII representation of the main graph:")
        # print(main_graph.get_graph().draw_ascii())
