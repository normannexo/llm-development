"""
Langchain Search tools, translated from FTV Smol Agent tools.
This module contains tools for searching various sources of information.
"""
import os
from typing import Optional, Type

import requests
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Conditional imports for tool-specific libraries
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None # type: ignore

try:
    import wikipedia
except ImportError:
    wikipedia = None # type: ignore

load_dotenv()

# --- Guardian Search Tool ---
class GuardianSearchInput(BaseModel):
    """Input schema for GuardianSearchTool."""
    query: str = Field(description="The search query to find relevant articles.")
    section: Optional[str] = Field(default=None, description="The section in which to search (e.g. sports, technology, politics)")
    page_size: Optional[int] = Field(default=5, description="The number of items returned in this call")
    from_date: Optional[str] = Field(default=None, description="The start date for the search in format YYYY-MM-DD")
    to_date: Optional[str] = Field(default=None, description="The end date for the search in format YYYY-MM-DD")

class GuardianSearchTool(BaseTool):
    """Tool for searching articles from The Guardian."""
    
    name: str = "guardian_search"
    description: str = "Searches for articles from The Guardian based on a query. Requires GUARDIAN_API_KEY environment variable."
    args_schema: Type[BaseModel] = GuardianSearchInput

    def _run(self, query: str, section: Optional[str] = None, page_size: Optional[int] = 5, from_date: Optional[str] = None, to_date: Optional[str] = None) -> str:
        api_key = os.environ.get("GUARDIAN_API_KEY")
        if not api_key:
            return "Error: Guardian API key not found. Please set the GUARDIAN_API_KEY environment variable."
        try:
            url = "https://content.guardianapis.com/search"
            params = {
                "q": query,
                "api-key": api_key,
                "show-fields": "headline,byline,trailText",
                "page-size": page_size if page_size is not None else 5,
                "order-by": "newest"
            }
            if section:
                params["section"] = section
            if from_date:
                params["from-date"] = from_date
            if to_date:
                params["to-date"] = to_date
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("response", {}).get("results", [])
            
            if not results:
                return f"No articles found for query: '{query}' in The Guardian."
            
            formatted_results_list = [f"Top {len(results)} articles for '{query}' from The Guardian:\n"]
            for i, article in enumerate(results, 1):
                headline = article.get("webTitle", "No title")
                article_section_name = article.get("sectionName", "Unknown section")
                date_published = article.get("webPublicationDate", "Unknown date").split("T")[0]
                web_url = article.get("webUrl", "#")
                fields = article.get("fields", {})
                trail_text = fields.get("trailText", "")
                
                formatted_results_list.append(f"{i}. {headline}")
                formatted_results_list.append(f"   Section: {article_section_name} | Date: {date_published}")
                if trail_text:
                    formatted_results_list.append(f"   Summary: {trail_text}")
                formatted_results_list.append(f"   URL: {web_url}\n")
            
            return "\n".join(formatted_results_list).strip()
            
        except requests.exceptions.RequestException as e:
            return f"Error fetching articles from The Guardian: {e}"
        except Exception as e:
            return f"Unexpected error during Guardian search: {e}"

# --- Tavily Search Tool ---
class TavilySearchInput(BaseModel):
    """Input schema for TavilySearchTool."""
    query: str = Field(description="Your search query for Tavily.")
    max_results: Optional[int] = Field(default=5, description="Maximum number of search results to return.")

class TavilySearchTool(BaseTool):
    """
    Tool for searching the web using Tavily.
    Requires TAVILY_API_KEY environment variable.
    Note: Langchain has a native TavilySearchResults tool which might be preferred.
    """
    name: str = "tavily_search_custom"
    description: str = (
        "Searches the web for your query using Tavily. "
        "Requires TAVILY_API_KEY environment variable. "
        "Returns a list of search results with titles, URLs, and content snippets."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str, max_results: Optional[int] = 5) -> str:
        if TavilyClient is None:
            return "Error: TavilyClient is not available. Please install the 'tavily-python' package."
        
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "Error: Tavily API key not found. Please set the TAVILY_API_KEY environment variable."
        
        try:
            tavily_client = TavilyClient(api_key=api_key)
            response = tavily_client.search(
                query=query, 
                search_depth="basic",
                max_results=max_results if max_results is not None else 5,
            )
            
            if not response or "results" not in response or not response["results"]:
                return f"No results found from Tavily for query: '{query}'"

            formatted_results_list = [f"Tavily search results for '{query}':\n"]
            for res_item in response["results"]:
                title = res_item.get("title", "No Title")
                url = res_item.get("url", "#")
                content_snippet = res_item.get("content", "No content snippet.")
                formatted_results_list.append(f"- Title: {title}\n  URL: {url}\n  Content: {content_snippet}\n")
            
            return "\n".join(formatted_results_list).strip()

        except Exception as e:
            return f"Error searching with Tavily: {e}"

# --- Wikipedia Search Tool ---
class WikipediaSearchInput(BaseModel):
    """Input schema for WikipediaSearchTool."""
    query: str = Field(description="The topic to search for on Wikipedia.")
    limit: Optional[int] = Field(default=3, description="Maximum number of results to return.")
    lang: Optional[str] = Field(default="en", description="Wikipedia language code (e.g., 'en', 'es', 'fr').")

class WikipediaSearchTool(BaseTool):
    """
    Tool for searching Wikipedia and getting summaries.
    Note: Langchain has a native WikipediaQueryRun tool which might be preferred.
    """
    name: str = "wikipedia_search_custom"
    description: str = (
        "Searches Wikipedia for a topic and returns summaries of the top results. "
        "Specify query and optionally the number of results (limit) and language (lang)."
    )
    args_schema: Type[BaseModel] = WikipediaSearchInput

    def _run(self, query: str, limit: Optional[int] = 3, lang: Optional[str] = "en") -> str:
        if wikipedia is None:
            return "Error: Wikipedia library is not available. Please install the 'wikipedia' package."
        try:
            wikipedia.set_lang(lang if lang else "en")
            search_results_titles = wikipedia.search(query, results=limit if limit is not None else 3)
            
            if not search_results_titles:
                return f"No Wikipedia articles found for '{query}' (lang: {lang})."
            
            formatted_output_list = [f"Wikipedia search results for '{query}' (lang: {lang}):\n"]
            
            for i, title in enumerate(search_results_titles, 1):
                try:
                    page_obj = wikipedia.page(title, auto_suggest=False, redirect=True)
                    summary_text = wikipedia.summary(title, sentences=3, auto_suggest=False, redirect=True)
                    page_url = page_obj.url
                    
                    formatted_output_list.append(f"{i}. {page_obj.title}")
                    formatted_output_list.append(f"   Summary: {summary_text}")
                    formatted_output_list.append(f"   URL: {page_url}\n")
                except wikipedia.exceptions.DisambiguationError as e_dis:
                    options = e_dis.options[:5] 
                    formatted_output_list.append(f"{i}. {title} (Disambiguation)")
                    formatted_output_list.append(f"   This topic is ambiguous. Options include: {', '.join(options)}\n")
                except wikipedia.exceptions.PageError:
                    formatted_output_list.append(f"{i}. {title}")
                    formatted_output_list.append(f"   Error: Could not retrieve page content for this title.\n")
                except Exception as e_inner:
                    formatted_output_list.append(f"{i}. {title}")
                    formatted_output_list.append(f"   Error processing page '{title}': {e_inner}\n")
            
            return "\n".join(formatted_output_list).strip()
            
        except ImportError:
            return "Error: The 'wikipedia' Python package is not installed. Please install it by running 'pip install wikipedia'."
        except Exception as e:
            return f"Error searching Wikipedia: {e}"
