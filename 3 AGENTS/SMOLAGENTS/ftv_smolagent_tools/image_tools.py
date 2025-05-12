"""
Image tools for the FTV Smol Agent.
This module contains tools for working with images.
"""

from smolagents import Tool, tool
import os
import base64
import requests
from dotenv import load_dotenv
from mistralai import Mistral

# Load environment variables from .env file
load_dotenv()


def encode_image(image_source):
    """
    Encode the image to base64. Can handle both local file paths and URLs.
    
    Args:
        image_source (str): Path to local image file or URL to an image
        
    Returns:
        str: Base64 encoded image or None if there was an error
    """
    try:
        # Check if the image_source is a URL
        if image_source.startswith(('http://', 'https://')):            
            # Download the image from the URL
            response = requests.get(image_source, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            # Encode the downloaded image
            return base64.b64encode(response.content).decode('utf-8')
        else:
            # Handle local file
            with open(image_source, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_source} was not found.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from URL: {e}")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None


def send_image_query(image_source, query_text="What's in this image?", model="pixtral-12b-2409"):
    """
    Encapsulates the process of encoding an image and sending it to the Mistral API.
    Can handle both local file paths and URLs.
    
    Args:
        image_source (str): Path to local image file or URL to an image
        query_text (str): The text query to send along with the image
        model (str): The model to use for the query
        
    Returns:
        The response from the Mistral API or None if there was an error
    """
    # Encode the image (works with both local files and URLs)
    base64_image = encode_image(image_source)
    if base64_image is None:
        return None
    
    try:
        # Retrieve the API key from environment variables
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            print("Error: MISTRAL_API_KEY environment variable not set.")
            return None
        
        # Initialize the Mistral client
        client = Mistral(api_key=api_key)
        
        # Define the messages for the chat
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query_text
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}" 
                    }
                ]
            }
        ]
        
        # Get the chat response
        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )
        
        return chat_response
    
    except Exception as e:
        print(f"Error sending image query: {e}")
        return None


class PixtralImageQueryTool(Tool):
    """Tool for querying the Pixtral model with images."""
    
    name = "pixtral_image_query"
    description = "Analyzes images using the Pixtral model. Can accept local file paths or image URLs."
    inputs = {
        "image_source": {
            "type": "string",
            "description": "Path to a local image file or URL of an image to analyze."
        },
        "query_text": {
            "type": "string",
            "description": "The text query to send along with the image.",
            "default": "What's in this image?",
            "nullable": True
        },
        "model": {
            "type": "string",
            "description": "The Pixtral model to use for the query.",
            "default": "pixtral-12b-2409",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, image_source: str, query_text: str = "What's in this image?", model: str = "pixtral-12b-2409"):
        """
        Process an image with Pixtral and return the analysis.
        
        Args:
            image_source: Path to local image file or URL to an image
            query_text: The text query to send along with the image
            model: The model to use for the query
            
        Returns:
            The response content from the Pixtral API or error message
        """
        try:
            # Check if MISTRAL_API_KEY is set
            if not os.environ.get("MISTRAL_API_KEY"):
                return "Error: MISTRAL_API_KEY environment variable not set."
            
            # Send the image query
            response = send_image_query(image_source, query_text, model)
            
            if response:
                return response.choices[0].message.content
            else:
                return "Error: Failed to get a response from Pixtral."
                
        except Exception as e:
            return f"Error processing image with Pixtral: {str(e)}"


@tool
def analyze_image(image_source: str, query_text: str = "What's in this image?") -> str:
    """
    Analyzes an image using the Pixtral model.
    
    Args:
        image_source: Path to local image file or URL to an image
        query_text: The text query to send along with the image
    """
    try:
        # Check if MISTRAL_API_KEY is set
        if not os.environ.get("MISTRAL_API_KEY"):
            return "Error: MISTRAL_API_KEY environment variable not set."
        
        # Use the default model
        model = "pixtral-12b-2409"
        
        # Send the image query
        response = send_image_query(image_source, query_text, model)
        
        if response:
            return response.choices[0].message.content
        else:
            return "Error: Failed to get a response from Pixtral."
            
    except Exception as e:
        return f"Error processing image with Pixtral: {str(e)}"
