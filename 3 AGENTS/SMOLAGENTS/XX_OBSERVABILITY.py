PROJECT_NAME = "Customer-Success"
import base64
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
import os
from ftv_smolagent_utils import create_mistral_model, create_code_agent
from smolagents import DuckDuckGoSearchTool
from dotenv import load_dotenv
load_dotenv()

langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")  
    
langfuse_auth = base64.b64encode(f"{langfuse_public_key}:{langfuse_secret_key}".encode()).decode()
    
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"  # EU data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
    
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    
SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

duckduckgo_search = DuckDuckGoSearchTool()  
model = create_mistral_model(api_key=MISTRAL_KEY, model_id="mistral/mistral-medium-latest") 
tools = [duckduckgo_search] 
agent = create_code_agent(model=model, tools=tools, name="CodeAgent", description="A code agent")
agent.run("Who won the Nations League in 2025?")
