from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent, CodeAgent

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env into environment

def setup_telemetry(langfuse_public_key, langfuse_secret_key):
    """Set up telemetry with Langfuse"""
    import base64
    from opentelemetry.sdk.trace import TracerProvider
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    import os
    
    langfuse_auth = base64.b64encode(f"{langfuse_public_key}:{langfuse_secret_key}".encode()).decode()
    
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"  # EU data region
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
    
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    
    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

def create_anthropic_model(api_key, model_id):
    """Create and return an Anthropic LiteLLM model"""
    return LiteLLMModel(
        max_tokens=2096,
        temperature=0.0,
        model_id=model_id,
        api_key=api_key,
        tpm=8000
    )
def create_mistral_model(api_key, model_id):
    """Create and return a Mistral LiteLLM model"""
    return LiteLLMModel(
        max_tokens=2096,
        temperature=0.0,
        model_id=model_id,
        api_key=api_key,
        tpm=8000
    )


def create_code_agent(model, tools, managed_agents=None,additional_authorized_imports=None, name=None, description=None):
    """Create and return a CodeAgent with the specified model and tools
    
    Args:
        model: The model to use for the agent
        tools: List of tools to provide to the agent
        managed_agents: Optional list of agents to be managed by this agent
    """
    return CodeAgent(
        tools=tools,
        model=model,
        add_base_tools=False,  # Add any additional base tools
        planning_interval=3,
        managed_agents=managed_agents,  # Add managed agents if provided
        additional_authorized_imports=["time", "bs4", "pandas"],   # Enable planning every 3 steps
        name=name,
        description=description
    )

def create_tool_calling_agent(model, tools, managed_agents=None, name=None, description=None):
    """Create and return a ToolCallingAgent with the specified model and tools
    
    Args:
        model: The model to use for the agent
        tools: List of tools to provide to the agent
        managed_agents: Optional list of agents to be managed by this agent
        name: Optional name for the agent
        description: Optional description for the agent
    """
    return ToolCallingAgent(
        tools=tools,
        model=model,
        add_base_tools=False,
        managed_agents=managed_agents,
        name=name,
        description=description
    )
