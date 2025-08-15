from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the state structure
class AgentState(TypedDict):
    publications: List[str]
    current_publication: str
    research_results: Dict[str, Dict[str, Any]]

# Initialize LLM
llm = ChatMistralAI(model="mistral-medium-latest")

@tool
def web_search(query: str) -> str:
    """Search the web for information about publications and their subscription plans."""
    print(f"Searching web for: {query}")
    return f"Search results for: {query}"

@tool
def visit_webpage(url: str) -> str:
    """Visit a specific webpage to get detailed information."""
    print(f"Visiting webpage: {url}")
    return f"Content from {url}"

# Create tool instances
tools = [web_search, visit_webpage]

# Create ReAct agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant that finds information about digital subscription plans for publications.
    For the given publication, find the following information:
    - Available subscription plans with their prices
    - Billing periods (monthly, annual, etc.)
    - Features included in each plan
    - Any special offers or discounts
    
    Be thorough in your research and verify information from multiple sources when possible.
    
    When you have gathered all the information, present it in a structured format.
    """),
    ("human", "Research subscription plans for: {publication}")
])

# Create the agent with the correct signature
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt
)


def research_publication(state: AgentState) -> AgentState:
    """Research subscription plans for the current publication."""
    publication = state["current_publication"]
    print(f"\nResearching: {publication}")
    
    try:
        # Execute the agent
        result = agent.invoke({
            "input": f"Research subscription plans for: {publication}"
        })
        
        # Store the result
        state["research_results"][publication] = {
            "status": "completed",
            "result": result
        }
        print(f"✅ Completed research for: {publication}")
        
    except Exception as e:
        print(f"❌ Error researching {publication}: {str(e)}")
        state["research_results"][publication] = {
            "status": "error",
            "error": str(e)
        }
    
    return state

def should_continue(state: AgentState) -> str:
    """Determine if we should continue processing more publications."""
    publications = state["publications"]
    results = state["research_results"]
    
    # If we've processed all publications, we're done
    if len(results) >= len(publications):
        return "end"
    
    # Otherwise, process the next publication
    next_pub = publications[len(results)]
    state["current_publication"] = next_pub
    return "research"

def compile_results(state: AgentState) -> AgentState:
    """Compile all research results into a pandas DataFrame."""
    data = []
    
    for pub, result in state["research_results"].items():
        if result["status"] == "completed":
            # In a real implementation, you would parse the agent's output
            # and extract structured data. This is a simplified example.
            data.append({
                "publication": pub,
                "plan_name": "Standard",  # Extract from result
                "price": "9.99",           # Extract from result
                "currency": "USD",          # Extract from result
                "billing_period": "monthly" # Extract from result
            })
    
    # Create and save DataFrame
    if data:
        df = pd.DataFrame(data)
        df.to_csv("publication_subscriptions.csv", index=False)
        print("\nResearch results saved to 'publication_subscriptions.csv'")
        print("\nSummary of subscription plans:")
        print(df.to_string(index=False))
    else:
        print("No valid research results to compile.")
    
    return state

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("research", research_publication)
workflow.add_node("compile", compile_results)

# Add edges
workflow.add_conditional_edges(
    "research",
    should_continue,
    {
        "research": "research",
        "end": "compile"
    }
)
workflow.add_edge("compile", END)

# Set the entry point
workflow.set_entry_point("research")

# Compile the workflow
app = workflow.compile()

def run_research(publications: List[str]):
    """Run the research workflow for the given list of publications."""
    # Initialize state
    state = {
        "publications": publications,
        "current_publication": publications[0] if publications else "",
        "research_results": {}
    }
    
    print(f"Starting research for {len(publications)} publications...")
    
    # Run the workflow
    for output in app.stream(state):
        for key, value in output.items():
            if key == "__end__":
                continue
    
    print("\nResearch completed!")

if __name__ == "__main__":
    # Example usage
    publications = ["The New York Times", "The Washington Post", "The Guardian"]
    run_research(publications)
