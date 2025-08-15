import os
import uuid
from datetime import datetime, timezone
import pandas as pd
from typing import List, Dict, Any, TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langfuse.callback import CallbackHandler
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langchain_core.tools import tool

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

# Setup tools
tools = [
    tavily_search_tool
]




# Load environment variables
load_dotenv()

# Initialize Langfuse callback handler
langfuse_handler = CallbackHandler()

# Initialize the LLM
llm = ChatMistralAI(
    model="mistral-medium-latest",
    temperature=0,
    max_retries=2
)

llm_with_tools = llm.bind_tools(tools)

# Define the state structure
class AgentState(TypedDict):
    publications: List[str]
    research_results: Dict[str, Dict[str, Any]]
    current_publication: str
    current_research: Dict[str, Any]

# Initialize the state
initial_state: AgentState = {
    "publications": [],
    "research_results": {},
    "current_publication": "",
    "current_research": {}
}
def custom_tools_condition(state: AgentState) -> bool:  
    # Get the last message from the state
    if isinstance(state, dict) and "messages" in state:
        messages = state["messages"]
        if messages and hasattr(messages[-1], "tool_calls"):
            return True
    return False

# Define a preparation function to initialize state
def prepare_state(state: AgentState) -> AgentState:
    """Initialize any state variables needed for the workflow."""
    # Initialize research_results if it doesn't exist
    if not state.get("research_results"):
        state["research_results"] = {}
    
    # Log the start of the process
    publications = state.get("publications", [])
    print(f"Starting workflow for {len(publications)} publications: {', '.join(publications)}")
    
    return state

def research_title(state: AgentState) -> AgentState:
    """Research a single publication title."""
    publication = state["current_publication"]
    print(f"Researching subscription plans for: {publication}")
    # Invoke the LLM with the current publication
    response = llm_with_tools.invoke([HumanMessage(content=publication)])
    # Process the response
    if response:
        result = response[0]
        if result.tool_calls:
            result = result.tool_calls[0].result
        # Update state with the research
        research_results = state.get("research_results", {})
        research_results[publication] = result
        state["research_results"] = research_results
    
    
    # Update state with the current publication for tools
    state["current_publication"] = publication
    
    # Return the state to be processed by the tool node
    return state

# Define the orchestrator agent
def orchestrator(state: AgentState) -> Dict[str, Any]:
    """Orchestrator that manages the workflow."""
    publications = state["publications"]
    research_results = state["research_results"]
    
    # If we have publications left to process
    if publications and len(research_results) < len(publications):
        # Get the next publication to process
        next_publication = publications[len(research_results)]
        state["current_publication"] = next_publication
        return {"next": "research_title", **state}
    
    # All publications processed
    return {"next": "end", **state}

# Define the function to compile results
def compile_results(state: AgentState) -> AgentState:
    """Compile all research results into a pandas DataFrame."""
    all_plans = []
    
    for publication, research in state["research_results"].items():
        if "plans" in research:
            for plan in research["plans"]:
                # Add publication name to each plan
                plan["publication"] = publication
                all_plans.append(plan)
    
    # Convert to DataFrame
    if all_plans:
        df = pd.json_normalize(
            all_plans,
            meta=["publication", "name", "price", "currency", "billing_period"],
            record_path=None
        )
        
        # Reorder columns for better readability
        columns = ["publication", "name", "price", "currency", "billing_period"]
        other_columns = [col for col in df.columns if col not in columns]
        df = df[columns + other_columns]
        
        # Save to CSV
        df.to_csv("publication_offers.csv", index=False)
        print("Results saved to publication_offers.csv")
        print("\nSummary of offers:")
        print(df[['publication', 'name', 'price', 'currency']].to_string(index=False))
    else:
        print("No research results to compile.")
    
    return state

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("research_title", research_title)
workflow.add_node("compile_results", compile_results)
workflow.add_node("tools", ToolNode(tools))

# Add the conditional edges for the orchestrator
workflow.add_conditional_edges(
    "orchestrator",
    lambda x: x["next"] if isinstance(x, dict) and "next" in x else "end",
    {
        "research_title": "research_title",
        "end": "compile_results"
    }
)

# Add edges between nodes
workflow.add_edge(START, "orchestrator")
workflow.add_edge("research_title", "orchestrator")
workflow.add_edge("compile_results", END)
workflow.add_conditional_edges("research_title", tools_condition)
workflow.add_edge("tools", "research_title")

# Compile the workflow
app = workflow.compile()

def run_research(publications: List[str]):
    """Run the research workflow for the given list of publications."""
    # Initialize the state with publications
    state = initial_state.copy()
    state["publications"] = publications
    
    # Prepare the initial state
    state = prepare_state(state)
    
    # Create a unique run ID for this execution
    run_id = str(uuid.uuid4())
    print(f"Run ID: {run_id}\n")
    
    # Add metadata to the Langfuse handler
    langfuse_handler.trace = langfuse_handler.langfuse.trace(
        id=run_id,
        name="publication_research",
        metadata={
            "publications": publications,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id
        }
    )
    
    # Store the run ID in the state
    state["run_id"] = run_id
    
    # Run the workflow with Langfuse callback
    config = {
        "callbacks": [langfuse_handler],
        "run_name": f"publication_research_{run_id}",
        "metadata": {
            "publications": publications,
            "run_id": run_id
        }
    }
    
    # Run the workflow - now uses standard START node
    for output in app.stream(state, config=config):
        for key, value in output.items():
            if key == "__end__" or key.startswith("__"):
                continue
            if key == "research_publication":
                current_pub = value.get("current_publication", "")
                if current_pub:
                    print(f"✅ Completed research for: {current_pub}")
    
    # Update the trace with completion status
    if hasattr(langfuse_handler, 'trace') and langfuse_handler.trace:
        langfuse_handler.trace.update(
            metadata={
                "status": "completed",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "publications_researched": list(state["research_results"].keys())
            }
        )
    
    print("\nResearch complete! Check 'publication_offers.csv' for the complete results.")
    if hasattr(langfuse_handler, 'trace') and langfuse_handler.trace:
        print(f"View trace: https://cloud.langfuse.com/trace/{langfuse_handler.trace.id}")

if __name__ == "__main__":
    # Example usage
    print(app.get_graph().draw_ascii())
    publications_to_research = ["The Guardian", "Linux Magazin", "Bild-Zeitung"]
   # run_research(publications_to_research)
