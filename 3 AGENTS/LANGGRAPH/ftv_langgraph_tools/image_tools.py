import os
import base64
from typing import Optional, Type, List, Dict, Any
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from dotenv import load_dotenv

# Conditional import for Mistral client
try:
    from mistralai.client import Mistral
    from mistralai.models.chat_completion import ChatMessage
except ImportError:
    Mistral = None # type: ignore
    ChatMessage = None # type: ignore

load_dotenv()

def _encode_image_to_base64(image_source: str) -> Optional[str]:
    """
    Encodes an image to base64. Handles local file paths and URLs.
    Helper function for the MistralImageQueryTool.
    """
    try:
        if image_source.startswith(('http://', 'https://')):
            response = requests.get(image_source, stream=True, timeout=10)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        else:
            # This check is now primarily for the helper's direct use if any.
            # The main tool _run method should pre-validate local paths.
            if not os.path.exists(image_source):
                return f"Error: Local image file not found at {image_source}" # Return error string
            with open(image_source, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return f"Error: The local file {image_source} was not found."
    except requests.exceptions.Timeout:
        return f"Error: Timeout while fetching image from URL: {image_source}"
    except requests.exceptions.RequestException as e:
        return f"Error downloading image from URL {image_source}: {e}"
    except Exception as e:
        return f"Error encoding image {image_source}: {e}"


class MistralImageQueryInput(BaseModel):
    """Input schema for MistralImageQueryTool."""
    image_source: str = Field(description="URL or local file path of the image.")
    query_text: str = Field(default="Describe this image in detail.", description="The text query to ask about the image.")
    model_name: Optional[str] = Field(default="mistral-large-latest", description="The Mistral model to use (e.g., 'mistral-large-latest'). Ensure it's vision-capable.")

class MistralImageQueryTool(BaseTool):
    """
    Tool to query a Mistral model with an image and a text prompt.
    Requires MISTRAL_API_KEY environment variable.
    The image can be a URL or a local file path.
    """
    name: str = "mistral_image_query"
    description: str = (
        "Sends an image (from URL or local path) and a text query to a Mistral model "
        "for analysis or description. Requires MISTRAL_API_KEY. "
        "Example query: 'What objects are in this image?' or 'Describe the scene.'"
    )
    args_schema: Type[BaseModel] = MistralImageQueryInput

    def _run(self, image_source: str, query_text: str = "Describe this image in detail.", model_name: Optional[str] = "mistral-large-latest") -> str:
        if Mistral is None or ChatMessage is None:
            return "Error: Mistral Python client is not installed. Please run 'pip install mistralai'."

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            return "Error: MISTRAL_API_KEY environment variable not set."

        if not image_source.startswith(('http://', 'https://')) and not os.path.exists(image_source):
            return f"Error: Local image file not found at {image_source}"

        base64_image_data = _encode_image_to_base64(image_source)
        if base64_image_data is None or base64_image_data.startswith("Error:"):
            return base64_image_data if base64_image_data is not None else "Error: Failed to encode image."

        image_type = "jpeg"  # Default
        lower_image_source = image_source.lower()
        if lower_image_source.endswith(".png"):
            image_type = "png"
        elif lower_image_source.endswith(".gif"):
            image_type = "gif"
        elif lower_image_source.endswith(".webp"):
            image_type = "webp"
        elif lower_image_source.endswith(".jpg") or lower_image_source.endswith(".jpeg"):
            image_type = "jpeg"
        
        # Ensure model_name is sensible
        current_model_name = model_name if model_name else "mistral-large-latest"

        try:
            client = Mistral(api_key=api_key)
            
            user_message_content: List[Dict[str, Any]] = [
                {"type": "text", "text": query_text},
                {
                    "type": "image_url",
                    "image_url": f"data:image/{image_type};base64,{base64_image_data}"
                }
            ]
            
            messages = [ChatMessage(role="user", content=user_message_content)]
            
            chat_response = client.chat(
                model=current_model_name, 
                messages=messages
            )
            
            if chat_response.choices and chat_response.choices[0].message and chat_response.choices[0].message.content:
                return chat_response.choices[0].message.content
            else:
                return "Error: No content received from Mistral API or unexpected response structure."

        except Exception as e:
            # Check for specific Mistral API errors if possible, e.g., authentication, model not found
            return f"Error during Mistral API call for image query ({current_model_name}): {e}"
