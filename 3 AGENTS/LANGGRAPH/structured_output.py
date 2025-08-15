# Bind the schema to the model

from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langfuse.callback import CallbackHandler
from langchain_tavily import TavilySearch

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
load_dotenv()

langfuse_handler = CallbackHandler()


tools = [
    tavily_search_tool
]

class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""
    answer: str = Field(description="The answer to the user's question")
    followup_question: str = Field(description="A followup question the user could ask")

model = ChatMistralAI(
    model="mistral-medium-latest",
    temperature=0,
    max_retries=2
)
model_with_tools = model.bind_tools(tools)
model_with_structure = model_with_tools.with_structured_output(ResponseFormatter)
# Invoke the model
structured_output = model_with_structure.invoke("Look up the latest news in search tool and respond.", config={"callbacks": [langfuse_handler]} )
# Get back the pydantic object
print(structured_output)