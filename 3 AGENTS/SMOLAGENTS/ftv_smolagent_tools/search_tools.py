"""
Search tools for the FTV Smol Agent.
This module contains tools for searching various sources of information.
"""

from smolagents import Tool, tool
import requests
import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables from .env file
load_dotenv()


class GuardianSearchTool(Tool):
    """Tool for searching articles from The Guardian."""
    
    name = "guardian_search"
    description = "Searches for articles from The Guardian based on a query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to find relevant articles."
        },
        "section": {
            "type": "string",
            "description": "The section in which to search (e.g. sports, technology, politics)",
            "nullable": True
        },
        "page_size": {
            "type": "integer",
            "description": "The number of items returned in this call",
            "nullable": True
        },
        "from_date": {
            "type": "string",
            "description": "The start date for the search in format YYYY-MM-DD",
            "nullable": True
        },
        "to_date": {
            "type": "string",
            "description": "The end date for the search in format YYYY-MM-DD",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, query: str, section: str = None, page_size: int = 5, from_date: str = None, to_date: str = None):
        """
        Search for articles from The Guardian.
        
        Args:
            query: The search query to find relevant articles
            section: The section in which to search (optional)
            page_size: The number of items to return (optional)
            from_date: The start date for the search in format YYYY-MM-DD (optional)
            to_date: The end date for the search in format YYYY-MM-DD (optional)
            
        Returns:
            Formatted search results or error message
        """
        try:
            # Get API key from environment variable
            api_key = os.environ.get("GUARDIAN_API_KEY")
            if not api_key:
                return "Error: Guardian API key not found. Please set the GUARDIAN_API_KEY environment variable."

            # API endpoint
            url = "https://content.guardianapis.com/search"
            
            # Parameters for the API request
            params = {
                "q": query,
                "api-key": api_key,
                "show-fields": "headline,byline,trailText",
                "page-size": page_size,
                "order-by": "newest"
            }
            
            # Add optional parameters only if they're not None
            if section:
                params["section"] = section
            
            # Add date parameters if provided
            if from_date:
                params["from-date"] = from_date
            
            if to_date:
                params["to-date"] = to_date
            
            # Make the request
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()
            results = data.get("response", {}).get("results", [])
            
            if not results:
                return f"No articles found for query: '{query}'."
            
            # Format the results
            formatted_results = f"Top articles for '{query}' from The Guardian:\n\n"
            for i, article in enumerate(results, 1):
                headline = article.get("webTitle", "No title")
                section = article.get("sectionName", "Unknown section")
                date = article.get("webPublicationDate", "Unknown date").split("T")[0]
                url = article.get("webUrl", "#")
                
                # Get additional fields if available
                fields = article.get("fields", {})
                trail_text = fields.get("trailText", "")
                
                formatted_results += f"{i}. {headline}\n"
                formatted_results += f"   Section: {section} | Date: {date}\n"
                if trail_text:
                    formatted_results += f"   Summary: {trail_text}\n"
                formatted_results += f"   URL: {url}\n\n"
            
            return formatted_results.strip()
            
        except requests.exceptions.RequestException as e:
            return f"Error fetching articles from The Guardian: {str(e)}"
        except Exception as e:
            return f"Unexpected error during Guardian search: {str(e)}"


@tool
def tavily_search(query: str) -> str:
    """
    Searches the web for your query using Tavily.
    
    Args:
        query: Your search query
        
    Returns:
        Search results from Tavily
    """
    try:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "Error: Tavily API key not found. Please set the TAVILY_API_KEY environment variable."
            
        tavily_client = TavilyClient(api_key=api_key)
        response = tavily_client.search(query)
        return str(response["results"])
    except Exception as e:
        return f"Error searching with Tavily: {str(e)}"


class WikipediaSearchTool(Tool):
    """Tool for searching Wikipedia."""
    
    name = "wikipedia_search"
    description = "Searches Wikipedia for information on a topic."
    inputs = {
        "query": {
            "type": "string",
            "description": "The topic to search for on Wikipedia."
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return",
            "nullable": True
        }
    }
    output_type = "string"
    
    def forward(self, query: str, limit: int = 3):
        """
        Search Wikipedia for information on a topic.
        
        Args:
            query: The topic to search for
            limit: Maximum number of results to return
            
        Returns:
            Formatted search results from Wikipedia
        """
        try:
            import wikipedia
            
            # Search for the query
            search_results = wikipedia.search(query, results=limit)
            
            if not search_results:
                return f"No Wikipedia articles found for '{query}'."
            
            # Format the results
            formatted_results = f"Wikipedia search results for '{query}':\n\n"
            
            for i, title in enumerate(search_results, 1):
                try:
                    # Get a summary of the page
                    summary = wikipedia.summary(title, sentences=3)
                    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    
                    formatted_results += f"{i}. {title}\n"
                    formatted_results += f"   Summary: {summary}\n"
                    formatted_results += f"   URL: {url}\n\n"
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    options = e.options[:5]  # Limit options to 5
                    formatted_results += f"{i}. {title} (disambiguation)\n"
                    formatted_results += f"   Options: {', '.join(options)}\n\n"
                except wikipedia.exceptions.PageError:
                    # Handle page not found
                    formatted_results += f"{i}. {title}\n"
                    formatted_results += f"   Error: Could not retrieve page content.\n\n"
            
            return formatted_results.strip()
            
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"
