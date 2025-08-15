from tavily import TavilyClient
from smolagents import tool, Tool
import os
from dotenv import load_dotenv

load_dotenv(override=True) # load variables from local .env file

@tool
def web_search(query: str) -> str:
    """Searches the web for your query.

    Args:
        query: Your query
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_client.search(query)
    return str(response["results"])


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def forward(self, url: str) -> str:
        try:
            import re

            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException

            from smolagents.utils import truncate_content
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: for instance run `pip install markdownify requests`."
            ) from e
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()  # Raise an exception for bad status codes
            markdown_content = markdownify(response.text).strip()
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            return truncate_content(markdown_content, 40000)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


from PIL import Image

from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel("anthropic/claude-3-5-sonnet-latest", temperature=0.2)


#changed to 2024 to avoid decaying internet content
request_museums = """
Could you give me a sorted list of the top 3 museums in the world in 2024,
along with their visitor count (in millions) that year, and the approximate daily temperature
in July at their location ?"""

web_agent = CodeAgent(
    model=model,
    tools=[web_search, VisitWebpageTool()],
    max_steps=10,
    name="web_agent",
    description="Runs web searches for you."
)
web_agent.logger.console.width=66

from smolagents import Tool
from typing import Any
from smolagents.utils import make_image_url, encode_image_base64

def check_reasoning_and_plot(final_answer, agent_memory):
    final_answer
    multimodal_model = LiteLLMModel(
        "anthropic/claude-3-5-sonnet-latest",
    )
    filepath = "saved_map.png"
    assert os.path.exists(filepath), "Make sure to save the plot under saved_map.png!"
    image = Image.open(filepath)
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
        "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
        "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
        "Don't be harsh: if the plot mostly solves the task, it should pass."
        "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
        "Also, any run that invents numbers should fail."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True


manager_agent = CodeAgent(

    model = LiteLLMModel("anthropic/claude-3-5-sonnet-latest", temperature=0.2),
    tools=[],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "plotly.express",  #added
        "plotly.express.colors",  #added
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],
    max_steps=15,
)
manager_agent.logger.console.width=66

manager_agent.visualize()