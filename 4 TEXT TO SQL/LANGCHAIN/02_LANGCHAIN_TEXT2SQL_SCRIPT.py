#!/usr/bin/env python3
"""
LangChain Text to SQL

This script converts natural language questions to SQL queries,
executes them against a SQLite database, and provides human-readable answers.
Based on https://python.langchain.com/docs/tutorials/sql_qa/
"""

import os
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraphh

# Load environment variables from .env file
load_dotenv()

# Define the State type for the graph
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# Define the QueryOutput type for structured output
class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def main():
    # Connect to the SQLite database
    path_sqlite_chinook_db = "DATA/EXAMPLEFILES/SQLITE/chinook.db"
    db = SQLDatabase.from_uri("sqlite:///" + path_sqlite_chinook_db)
    
    print(f"Database dialect: {db.dialect}")
    print(f"Available tables: {db.get_usable_table_names()}")
    
    # Initialize the language model
    model_name = "claude-3-5-haiku-20241022"  # You can change this to any supported model
    llm = init_chat_model(
        model_name, 
        model_provider="anthropic", 
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Check if model is working with a simple test
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Testing LLM connection...")
        response = llm.invoke(["Hello, are you ready to help with SQL queries?"])
        print(f"LLM response: {response.content}\n")
    else:
        print("Warning: No ANTHROPIC_API_KEY found in environment variables.")
        print("The script will run with mock responses.\n")
    
    # Create the prompt template for SQL query generation
    system_message = """
    Given an input question, create a syntactically correct {dialect} query to
    run to help find the answer. Unless the user specifies in his question a
    specific number of examples they wish to obtain, always limit your query to
    at most {top_k} results. You can order the results by a relevant column to
    return the most interesting examples in the database.

    Never query for all the columns from a specific table, only ask for a the
    few relevant columns given the question.

    Pay attention to use only the column names that you can see in the schema
    description. Be careful to not query for columns that do not exist. Also,
    pay attention to which column is in which table.

    Only use the following tables:
    {table_info}
    """

    user_prompt = "Question: {input}"

    query_prompt_template = ChatPromptTemplate(
        [("system", system_message), ("user", user_prompt)]
    )
    
    # Define the functions for the graph
    def write_query(state: State):
        """Generate SQL query to fetch information."""
        prompt = query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                "input": state["question"],
            }
        )
        
        # If no API key, return a mock query
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Using mock query (no API key provided)")
            if "employee" in state["question"].lower():
                return {"query": "SELECT COUNT(*) AS EmployeeCount FROM employees;"}
            else:
                return {"query": "SELECT * FROM artists LIMIT 5;"}
        
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}

    def execute_query(state: State):
        """Execute SQL query."""
        print(f"Executing query: {state['query']}")
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        return {"result": execute_query_tool.invoke(state["query"])}

    def generate_answer(state: State):
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {state["question"]}\n'
            f'SQL Query: {state["query"]}\n'
            f'SQL Result: {state["result"]}'
        )
        
        # If no API key, return a mock answer
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Using mock answer (no API key provided)")
            if "employee" in state["question"].lower():
                return {"answer": "There are 8 employees in the database."}
            else:
                return {"answer": "Here are the first 5 artists in the database."}
        
        response = llm.invoke(prompt)
        return {"answer": response.content}

    # Build the graph
    graph_builder = StateGraph(State).add_sequence(
        [write_query, execute_query, generate_answer]
    )
    graph_builder.add_edge(START, "write_query")
    graph = graph_builder.compile()
    
    # Interactive loop
    print("\n=== Text to SQL Interactive Query System ===")
    print("Type 'exit' to quit\n")
    
    while True:
        user_question = input("Enter your question: ")
        if user_question.lower() in ['exit', 'quit', 'q']:
            break
            
        print("\nProcessing your question...")
        
        # Execute the graph with the user's question
        for step in graph.stream(
            {"question": user_question}, stream_mode="updates"
        ):
            # Print each step's output in a readable format
            if "write_query" in step:
                print(f"\nGenerated SQL Query:\n{step['write_query']['query']}")
            elif "execute_query" in step:
                print(f"\nQuery Result:\n{step['execute_query']['result']}")
            elif "generate_answer" in step:
                print(f"\nAnswer:\n{step['generate_answer']['answer']}")
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
