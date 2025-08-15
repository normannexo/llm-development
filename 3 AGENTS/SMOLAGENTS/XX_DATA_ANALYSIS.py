import smolagents
from smolagents import LiteLLMModel, CodeAgent
from dotenv import load_dotenv
from ftv_smolagent_utils import create_anthropic_model, create_mistral_model
import os

load_dotenv()

#model = create_anthropic_model(api_key=os.getenv("ANTHROPIC_API_KEY"), model_id="claude-3-7-sonnet-20250219")
model = create_mistral_model(api_key=os.getenv("MISTRAL_API_KEY"), model_id="mistral/mistral-medium-latest")




system_prompt = """
## Instructions
You are acting as a expert data analyst.
Given a pandas DataFrame.

## Analytics Steps
1. Load the provided CSV file into a pandas DataFrame.
2. Perform exploratory data analysis (EDA) on the DataFrame.
3. Generate visualizations to summarize the data.
4. Provide a summary of the findings.

## Data information
# [Provide column names, data types (numerical/categorical), and brief descriptions here]

## User Question:
{query}
"""

agent = CodeAgent(
    tools=[],  # Empty list since we'll use built-in tools only
    model=model, 
    add_base_tools=True,  # Enable standard Python execution capabilities
    additional_authorized_imports=[  # SECURITY: Strictly limit allowed libraries
        "pandas", "numpy", "datetime",
        "matplotlib", "plotly", "seaborn", "sklearn"
    ]
    , verbosity_level=2
)

agent.run("make a basic eda on Clean_Dataset.csv and draw one example plot.")