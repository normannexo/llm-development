import os
from dotenv import load_dotenv
from ftv_smolagent_gradio import FtvGradioUI

from ftv_smolagent_utils import create_anthropic_model, create_code_agent
from ftv_smolagent_tools import VisitWebpageTool, tavily_search
load_dotenv()

visit_webpage_tool = VisitWebpageTool()
tavily_search_tool = tavily_search

# Set up environment variables
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]
MISTRAL_KEY = os.environ["MISTRAL_API_KEY"]

# Set up telemetry
#setup_telemetry(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)

# Import the BasicAgent class from basicsmolagent.py
model = create_anthropic_model(api_key=ANTHROPIC_KEY, model_id="claude-3-5-haiku-20241022")


agent = create_code_agent(model=model, tools=[visit_webpage_tool, tavily_search_tool])
    
FtvGradioUI(agent=agent).launch()