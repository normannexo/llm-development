#!/usr/bin/env python
# coding=utf-8
"""
Gradio UI for testing the agent with GAIA questions.
"""

import os
import json
import re
import shutil
from pathlib import Path

import gradio as gr
from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent, PlanningStep
from smolagents.memory import ActionStep, FinalAnswerStep, MemoryStep
from smolagents.utils import _is_package_available

from .ftv_gradio_ui import get_step_footnote_content, pull_messages_from_step, stream_to_gradio


class GaiaTestGradioUI:
    """A Gradio interface for testing agents with GAIA questions."""

    def __init__(self, agent: MultiStepAgent, questions_path=None, file_upload_folder=None):
        """
        Initialize the Gradio UI for GAIA testing.
        
        Args:
            agent: The agent to use for answering questions
            questions_path: Path to the questions.json file
            file_upload_folder: Directory to store uploaded files
        """
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        
        self.agent = agent
        self.name = agent.name if hasattr(agent, "name") else "GAIA Test Agent"
        self.description = agent.description if hasattr(agent, "description") else "Test agent for GAIA questions"
        self.file_upload_folder = file_upload_folder
        
        # Load questions from the provided path or use default
        self.questions_path = questions_path or os.path.join(
            "agent_course_final_assignment", "data", "questions.json"
        )
        self.questions = self._load_questions()
        
        # Create a dictionary of files associated with questions
        self.files_dir = os.path.join(os.path.dirname(self.questions_path), "files")
        
    def _load_questions(self):
        """Load questions from the JSON file."""
        try:
            if os.path.exists(self.questions_path):
                with open(self.questions_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                print(f"Questions file not found at {self.questions_path}")
                return []
        except Exception as e:
            print(f"Error loading questions: {e}")
            return []

    def interact_with_agent(self, prompt, chat_history, session_state):
        """Handle interaction with the agent."""
        if not prompt:
            return chat_history
        
        # Check if this is a file-related question
        has_file = False
        file_path = None
        
        for question in self.questions:
            if question["question"] in prompt and question.get("file_name"):
                file_name = question.get("file_name")
                if file_name:
                    file_path = os.path.join(self.files_dir, file_name)
                    if os.path.exists(file_path):
                        has_file = True
                        break
        
        # Prepare task and additional arguments
        task = prompt
        additional_args = {}
        
        # If there's an associated file, add it to the arguments
        if has_file:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                # Handle image files
                additional_args["image_url"] = file_path
            elif file_extension in ['.mp3', '.wav', '.ogg']:
                # Handle audio files
                additional_args["audio_url"] = file_path
            elif file_extension in ['.py', '.js', '.html', '.css', '.txt']:
                # Handle code or text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    additional_args["code"] = f.read()
            elif file_extension in ['.xlsx', '.csv']:
                # Handle data files - just pass the path
                additional_args["data_file"] = file_path
        
        # Set the button to disabled while processing
        gr.Button(interactive=False)
        gr.Textbox(interactive=False, placeholder="Processing...")
        
        # Stream the agent's response
        for message in stream_to_gradio(
            self.agent, task, reset_agent_memory=True, additional_args=additional_args
        ):
            chat_history = chat_history + [message]
            yield chat_history
    
    def upload_file(self, file, file_uploads_log, allowed_file_types=None):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        if file is None:
            return gr.Textbox(visible=False), file_uploads_log
        
        if allowed_file_types is None:
            allowed_file_types = [".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".mp3", ".wav"]
        
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in allowed_file_types:
            return gr.Textbox(
                value=f"File type {file_ext} not allowed. Allowed types: {', '.join(allowed_file_types)}",
                visible=True,
            ), file_uploads_log
        
        if self.file_upload_folder:
            # Create the upload folder if it doesn't exist
            os.makedirs(self.file_upload_folder, exist_ok=True)
            
            # Copy the file to the upload folder
            target_path = os.path.join(self.file_upload_folder, os.path.basename(file.name))
            shutil.copy(file.name, target_path)
            
            # Add the file to the uploads log
            file_uploads_log = file_uploads_log + [target_path]
            
            return gr.Textbox(value=f"File uploaded: {os.path.basename(file.name)}", visible=True), file_uploads_log
        else:
            return gr.Textbox(value="File upload folder not configured", visible=True), file_uploads_log
    
    def log_user_message(self, text_input, file_uploads_log):
        """Log a user message and prepare the UI for agent response."""
        if not text_input:
            return [], text_input, gr.Button(interactive=True)
        
        # Disable the input while processing
        return [text_input], "", gr.Button(interactive=False)
    
    def launch(self, share: bool = True, **kwargs):
        """Launch the Gradio app."""
        app = self.create_app()
        app.queue()
        app.launch(share=share, **kwargs)
    
    def create_app(self):
        """Create the Gradio app with examples from the questions.json file."""
        with gr.Blocks(theme="monochrome", fill_height=False) as demo:
            # Add session state to store session-specific data
            session_state = gr.State({})
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            
            with gr.Sidebar():
                gr.Markdown(
                    f"# {self.name.replace('_', ' ').capitalize()}"
                    "\n> This web UI allows you to test the agent with GAIA questions."
                    + (f"\n\n**Agent description:**\n{self.description}" if self.description else "")
                )
                
                with gr.Group():
                    gr.Markdown("**Your request**", container=True)
                    text_input = gr.Textbox(
                        lines=3,
                        label="Chat Message",
                        container=False,
                        placeholder="Enter your prompt here and press Shift+Enter or press the button",
                    )
                    submit_btn = gr.Button("Submit", variant="primary")
                
                # If an upload folder is provided, enable the upload feature
                if self.file_upload_folder is not None:
                    upload_file = gr.File(label="Upload a file")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                    upload_file.change(
                        self.upload_file,
                        [upload_file, file_uploads_log],
                        [upload_status, file_uploads_log],
                    )
                
                gr.HTML("<br><br><h4><center>Powered by:</center></h4>")
                with gr.Row():
                    gr.HTML("""<div style="display: flex; align-items: center; gap: 8px; font-family: system-ui, -apple-system, sans-serif;">
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png" style="width: 32px; height: 32px; object-fit: contain;" alt="logo">
            <a target="_blank" href="https://github.com/huggingface/smolagents"><b>huggingface/smolagents</b></a>
            </div>""")
            
            # Main chat interface
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                ),
                resizeable=True,
                scale=1,
                height=600,  # Set a fixed height
                container=False,  # Disable container to prevent double scrollbars
            )
            
            # Helper function to add user message to chatbot
            def add_user_message(message, chat_history):
                import gradio as gr
                # Create a ChatMessage object with the user's message
                user_message = gr.ChatMessage(role="user", content=message)
                # Return a new chat history with the user message added
                return chat_history + [user_message]
            
            # Set up event handlers - simplified chain
            def process_query(user_input, chat_history, state):
                # First add user message to chat
                import gradio as gr
                user_message = gr.ChatMessage(role="user", content=user_input)
                chat_history = chat_history + [user_message]
                
                # Then get agent response
                for response in stream_to_gradio(
                    self.agent, user_input, reset_agent_memory=True, additional_args={}
                ):
                    chat_history = chat_history + [response]
                    yield chat_history
            
            # Handle text input submission
            text_input.submit(
                lambda x: x,  # Just pass the input through
                [text_input],
                [text_input],
                queue=False
            ).then(
                process_query,
                [text_input, chatbot, session_state],
                [chatbot]
            ).then(
                lambda: (
                    "",  # Clear the text input
                    gr.Button(interactive=True)  # Re-enable the button
                ),
                None,
                [text_input, submit_btn]
            )
            
            # Handle button click
            submit_btn.click(
                lambda x: x,  # Just pass the input through
                [text_input],
                [text_input],
                queue=False
            ).then(
                process_query,
                [text_input, chatbot, session_state],
                [chatbot]
            ).then(
                lambda: (
                    "",  # Clear the text input
                    gr.Button(interactive=True)  # Re-enable the button
                ),
                None,
                [text_input, submit_btn]
            )
            
            # Add examples from the questions.json file
            examples = []
            for question in self.questions:
                if question.get("question"):
                    examples.append(question["question"])
            
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=text_input,
                    label="GAIA Test Questions"
                )
        
        return demo


# Export the class
__all__ = ["GaiaTestGradioUI"]
