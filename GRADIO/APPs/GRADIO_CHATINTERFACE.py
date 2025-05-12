from langchain_mistralai import ChatMistralAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import gradio as gr
import os
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-latest", 
    api_key=os.getenv("MISTRAL_API_KEY"),
    streaming=True
)

# Default system message
DEFAULT_SYSTEM_MSG = "You are a helpful AI assistant."


def predict(message, history, system_message=DEFAULT_SYSTEM_MSG):
    # Don't process empty messages
    if not message or message.strip() == "":
        return ""
        
    history_langchain_format = []
    
    # Add system message if provided
    if system_message and system_message.strip():
        history_langchain_format.append(SystemMessage(content=system_message))
    
    # Process history
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    
    # Add current message
    history_langchain_format.append(HumanMessage(content=message))
    
    try:
        # Use a mock response if API key is not available or invalid
        if not os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY").strip() == "":
            # Mock response for testing without API key
            return "This is a mock response since no valid API key was provided. Please set your MISTRAL_API_KEY in the .env file."
        
        # For streaming responses
        partial_message = ""
        for chunk in model.stream(history_langchain_format):
            if chunk and hasattr(chunk, 'content'):
                partial_message += chunk.content
                yield partial_message
        
        return partial_message
    except Exception as e:
        # Handle errors gracefully
        return f"Error: {str(e)}"

def save_conversation(history):
    if not history:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(history, f, indent=2)
    
    return filename

def upload_conversation(file):
    try:
        with open(file.name, "r") as f:
            history = json.load(f)
        return history
    except Exception as e:
        return [{"role": "assistant", "content": f"Error loading conversation: {str(e)}"}]

# Create the ChatInterface with additional components
demo = gr.ChatInterface(
    fn=predict,
    chatbot=gr.Chatbot(type="messages", height=500),
    textbox=gr.Textbox(placeholder="Type your message here...", label="Your message", scale=4),
    title="Enhanced AI Chat Assistant",
    description="Chat with an AI assistant powered by Mistral AI. You can customize the system message to change the assistant's behavior.",
    theme=gr.themes.Monochrome(),
    examples=[
        ["Tell me about artificial intelligence"],
        ["Write a short poem about nature"],
        ["Explain quantum computing to a 10-year-old"],
        ["What are the best practices for Python coding?"],
        ["Tell me a joke"]
    ],
    additional_inputs=[
        gr.Textbox(label="System Message", placeholder="Set the AI's behavior...", value=DEFAULT_SYSTEM_MSG, lines=3)
    ]
)

# Add file operations in a separate tab
with demo:
    with gr.Accordion("File Operations", open=False):
        with gr.Row():
            save_btn = gr.Button("Save Conversation")
            saved_file = gr.File(label="Download Conversation", interactive=False, visible=False)
        upload_file = gr.File(label="Upload Conversation", file_types=[".json"])
    
    # Set up event handlers for file operations
    save_btn.click(save_conversation, [demo.chatbot], [saved_file]).then(
        lambda x: gr.update(visible=True), [saved_file], [saved_file]
    )
    upload_file.change(upload_conversation, [upload_file], [demo.chatbot])

# Launch the app with share=True to create a public link
demo.launch(share=False)