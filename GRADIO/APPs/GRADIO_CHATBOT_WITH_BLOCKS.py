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

def predict(message, history, system_message):
    # Don't process empty messages
    if not message or message.strip() == "":
        return "", history
        
    # Convert Gradio chat history format to LangChain format
    history_langchain_format = []
    
    # Add system message if provided
    if system_message and system_message.strip():
        history_langchain_format.append(SystemMessage(content=system_message))
    
    # Process existing history
    for msg in history:
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            if msg['role'] == "user":
                history_langchain_format.append(HumanMessage(content=msg['content']))
            elif msg['role'] == "assistant":
                history_langchain_format.append(AIMessage(content=msg['content']))
    
    # Add the current message
    history_langchain_format.append(HumanMessage(content=message))
    
    try:
        # Get response from the model with streaming
        response_content = ""
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_content})
        
        # Use a mock response if API key is not available or invalid
        if not os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY").strip() == "":
            # Mock response for testing without API key
            response_content = "This is a mock response since no valid API key was provided. Please set your MISTRAL_API_KEY in the .env file."
            history[-1]["content"] = response_content
            return "", history
        
        for chunk in model.stream(history_langchain_format):
            if chunk and hasattr(chunk, 'content'):
                response_content += chunk.content
                history[-1]["content"] = response_content
                yield "", history
        
        return "", history
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"Error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history

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

# Add a new function for text generation in the second tab
def generate_text(prompt, max_length):
    if not prompt or prompt.strip() == "":
        return "Please enter a prompt to generate text."
        
    try:
        # Create a simple message format for the model
        messages = [
            SystemMessage(content=DEFAULT_SYSTEM_MSG),
            HumanMessage(content=f"{prompt}\nGenerate a response with approximately {max_length} words.")
        ]
        
        # Use a mock response if API key is not available or invalid
        if not os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY").strip() == "":
            return "This is a mock response since no valid API key was provided. Please set your MISTRAL_API_KEY in the .env file."
        
        # Get response from the model
        response = model.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Enhanced AI Assistant")
    
    with gr.Tabs() as tabs:
        # First tab - Chat Interface
        with gr.TabItem("Chat"):
            gr.Markdown("Chat with an AI assistant powered by Mistral AI. You can customize the system message to change the assistant's behavior.")
            
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(type="messages", height=500)
                    msg = gr.Textbox(placeholder="Type your message here...", label="Your message")
                    with gr.Row():
                        clear = gr.ClearButton([msg, chatbot], value="Clear chat")
                        submit_btn = gr.Button("Submit", variant="primary")
                
                with gr.Column(scale=1):
                    system_msg = gr.Textbox(
                        label="System Message", 
                        placeholder="Set the AI's behavior...",
                        value=DEFAULT_SYSTEM_MSG,
                        lines=3
                    )
                    with gr.Accordion("File Operations", open=False):
                        save_btn = gr.Button("Save Conversation")
                        saved_file = gr.File(label="Download Conversation", interactive=False, visible=False)
                        upload_file = gr.File(label="Upload Conversation", file_types=[".json"])
            
            with gr.Accordion("Examples", open=False):
                gr.Examples(
                    examples=[
                        ["Tell me about artificial intelligence", DEFAULT_SYSTEM_MSG],
                        ["Write a short poem about nature", DEFAULT_SYSTEM_MSG],
                        ["Explain quantum computing to a 10-year-old", DEFAULT_SYSTEM_MSG],
                        ["What are the best practices for Python coding?", DEFAULT_SYSTEM_MSG],
                        ["Tell me a joke", "You are a comedian who specializes in clean, family-friendly humor."]
                    ],
                    inputs=[msg, system_msg],
                )
        
        # Second tab - Text Generation
        with gr.TabItem("Text Generation"):
            gr.Markdown("Generate text based on a prompt. Specify the approximate length of the response.")
            
            with gr.Column():
                prompt = gr.Textbox(placeholder="Enter your prompt here...", label="Prompt", lines=3)
                max_length = gr.Slider(minimum=50, maximum=500, value=200, step=50, label="Approximate Word Count")
                generate_btn = gr.Button("Generate", variant="primary")
                output = gr.Textbox(label="Generated Text", lines=10)
                
                with gr.Accordion("Example Prompts", open=False):
                    gr.Examples(
                        examples=[
                            ["Write a short story about a space explorer discovering a new planet."],
                            ["Create a recipe for a delicious chocolate cake."],
                            ["Write a professional email requesting a meeting with a potential client."],
                            ["Explain the concept of climate change in simple terms."]
                        ],
                        inputs=[prompt],
                    )
    
    # Event handlers for first tab
    msg.submit(predict, [msg, chatbot, system_msg], [msg, chatbot])
    submit_btn.click(predict, [msg, chatbot, system_msg], [msg, chatbot])
    save_btn.click(save_conversation, [chatbot], [saved_file]).then(
        lambda x: gr.update(visible=True), [saved_file], [saved_file]
    )
    upload_file.change(upload_conversation, [upload_file], [chatbot])
    
    # Event handlers for second tab
    generate_btn.click(generate_text, [prompt, max_length], [output])

# Launch the app with share=True to create a public link
demo.launch(share=False)