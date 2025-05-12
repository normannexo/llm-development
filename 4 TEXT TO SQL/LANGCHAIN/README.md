# LangChain Text-to-SQL

A collection of tools and applications for converting natural language queries into SQL using LangChain.

## Features

- Convert natural language to SQL queries
- Support for multiple database types
- Interactive Gradio UI for easy query testing
- Command-line script for batch processing

## Usage

### Script

Run the command-line script for quick text-to-SQL conversion:

```bash
python 02_LANGCHAIN_TEXT2SQL_SCRIPT.py "Show me all customers who made purchases last month"
```

### Application

Launch the interactive application with:

```bash
python 03_LANGCHAIN_TEXT2SQL_APP.py
```

## Requirements

- Python 3.8+
- LangChain
- Anthropic Claude or other supported LLM
- SQLAlchemy
- Gradio (for UI version)

## Project Structure

- `02_LANGCHAIN_TEXT2SQL_SCRIPT.py` - Command-line script for text-to-SQL conversion
- `03_LANGCHAIN_TEXT2SQL_APP.py` - Interactive application with UI
- `../SCRIPTS/langchain_text2sql.py` - Core implementation of text-to-SQL functionality
- `../UI_GRADIO/text2sql_gradio_app.py` - Gradio-based UI implementation
