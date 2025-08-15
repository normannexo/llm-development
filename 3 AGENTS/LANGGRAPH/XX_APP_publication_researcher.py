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
from langgraph.prebuilt import create_react_agent
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

# Define a tool for visiting and reading content from webpages
from langchain_core.tools import BaseTool
from typing import Optional, Dict, Any, Type
from pydantic import BaseModel, Field

class VisitWebpageInput(BaseModel):
    url: str = Field(description="The url of the webpage to visit")
    
class VisitWebpageTool(BaseTool):
    name: str = "visit_webpage"
    description: str = "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    args_schema: Type[BaseModel] = VisitWebpageInput
    
    max_output_length: int = 40000
    
    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to stay within the maximum length."""
        if len(content) <= max_length:
            return content
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )
    
    def _run(self, url: str) -> str:
        """Execute the webpage visit and return the content as markdown."""
        try:
            import re
            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: "
                "for instance run `pip install markdownify requests`."
            ) from e
        try:
            # Send a GET request to the URL with a 20-second timeout
            response = requests.get(url, timeout=20)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert the HTML content to Markdown
            markdown_content = markdownify(response.text).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return self._truncate_content(markdown_content, self.max_output_length)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

# Instantiate the tool
visit_webpage_tool = VisitWebpageTool()

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

# Define the worker agent for researching a single publication
def research_publication(state: AgentState) -> AgentState:
    """Worker agent that researches digital subscription plans for a publication."""
    publication = state["current_publication"]
    print(f"\nResearching subscription plans for: {publication}")
    
    # Setup a fallback response structure in case of errors
    fallback_research = {
        "plans": [{
            "name": f"No data for {publication}",
            "price": 0,
            "currency": "USD",
            "billing_period": "unknown",
            "features": [f"Could not retrieve subscription data for {publication}"],
            "special_offers": []
        }]
    }
    
    try:
        print(f"  - Using a direct approach to research {publication}...")
        
        # 1. First search for information about subscription plans
        print(f"  - Searching for subscription information...")
        search_query = f"{publication} digital subscription plans pricing"        
        search_results = tavily_search_tool.invoke(search_query)
        
        # 2. Extract potential URLs from search results
        print(f"  - Extracting URLs from search results...")
        website_urls = []
        # If search_results is a string, try to extract URLs
        if isinstance(search_results, str):
            import re
            # Find URLs in the search results
            urls = re.findall(r'https?://\S+', search_results)
            if urls:
                website_urls = urls[:2]  # Take up to 2 URLs to avoid excessive processing
        
        # 3. Visit each URL and extract content
        content_collection = []
        for i, url in enumerate(website_urls):
            print(f"  - Visiting website {i+1}/{len(website_urls)}...")
            try:
                webpage_content = visit_webpage_tool.invoke(url)
                if webpage_content and isinstance(webpage_content, str):
                    content_collection.append(webpage_content[:5000])  # Limit content size
            except Exception as e:
                print(f"    - Error visiting {url}: {str(e)}")
        
        # 4. Use the LLM to analyze and extract subscription plans from all gathered information
        print(f"  - Analyzing collected data to extract subscription plans...")
        
        # Combine all information into a prompt for the LLM
        analysis_prompt = f"""Based on the following information about {publication}, extract all subscription plans.
        Return ONLY a JSON object with this exact structure: {{"plans": [{{"name": string, "price": number, "currency": string, "billing_period": string, "features": string[], "special_offers": string[]}}]}}
        
        SEARCH RESULTS:
        {search_results[:3000]}
        
        WEBPAGE CONTENT:
        {' '.join(content_collection)[:5000]}
        """
        
        # Get the LLM to analyze and structure the data
        analysis_result = llm.invoke(analysis_prompt)
        result_text = analysis_result.content if hasattr(analysis_result, 'content') else str(analysis_result)
        
        # Try to parse the JSON from the LLM's response
        print(f"  - Parsing subscription plan data...")
        try:
            # Look for JSON pattern in the response
            import re
            import json
            
            # First try to find well-formatted JSON in the text
            json_match = re.search(r'\{\s*"plans"\s*:\s*\[.+?\]\s*\}', result_text, re.DOTALL)
            if json_match:
                research_data = json.loads(json_match.group(0))
            else:
                # Try to see if the whole response is parseable JSON
                research_data = json.loads(result_text)
            
            # Validate the structure
            if "plans" not in research_data:
                if isinstance(research_data, list):
                    research_data = {"plans": research_data}
                else:
                    research_data = {"plans": []}
                    
            print(f"  ✓ Found {len(research_data.get('plans', []))} subscription plans")
                
        except Exception as parsing_error:
            print(f"  - Error parsing JSON: {str(parsing_error)}")
            # Create a structured response from the text
            research_data = {
                "plans": [{
                    "name": f"Parsing Error for {publication}",
                    "price": 0,
                    "currency": "USD",
                    "billing_period": "unknown",
                    "features": ["Error occurred when parsing subscription data"],
                    "special_offers": []
                }]
            }
    
    except Exception as e:
        print(f"Error researching {publication}: {str(e)}")
        research_data = fallback_research
    
    # Update the state with research results
    state["research_results"][publication] = research_data
    state["current_research"] = research_data
    
    return state

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
        return {"next": "research_publication", **state}
    
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
workflow.add_node("research_publication", research_publication)
workflow.add_node("compile_results", compile_results)

# Add the conditional edges for the orchestrator
workflow.add_conditional_edges(
    "orchestrator",
    lambda x: x["next"] if isinstance(x, dict) and "next" in x else "end",
    {
        "research_publication": "research_publication",
        "end": "compile_results"
    }
)

# Add edges between nodes
workflow.add_edge(START, "orchestrator")
workflow.add_edge("research_publication", "orchestrator")
workflow.add_edge("compile_results", END)

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
    publications_to_research = ["The Guardian", "Linux Magazin", "Bild-Zeitung"]
    run_research(publications_to_research)
