"""
Langchain Web tools, translated from FTV Smol Agent tools.
This module contains tools for interacting with web pages.
"""
import re
from typing import Optional, Type
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class VisitWebpageInput(BaseModel):
    """Input schema for VisitWebpageTool."""
    url: str = Field(description="The URL of the webpage to visit.")
    extract_links: Optional[bool] = Field(
        default=True, description="Whether to extract links from the webpage."
    )
    extract_text: Optional[bool] = Field(
        default=True, description="Whether to extract text from the webpage."
    )


class VisitWebpageTool(BaseTool):
    """Tool for visiting and extracting content from web pages."""
    
    name: str = "visit_webpage"
    description: str = "Visits a webpage and extracts its content. Can optionally extract text and/or links."
    args_schema: Type[BaseModel] = VisitWebpageInput
    # return_direct: bool = False # Default, can be set if needed

    def _run(self, url: str, extract_links: Optional[bool] = True, extract_text: Optional[bool] = True) -> str:
        """
        Visit a webpage and extract its content.
        
        Args:
            url: The URL of the webpage to visit.
            extract_links: Whether to extract links from the webpage.
            extract_text: Whether to extract text from the webpage.
            
        Returns:
            Extracted content from the webpage as a string.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()
            
            result_parts = [f"Visited: {url}\n"]
            
            title = soup.title.string if soup.title else "No title found"
            result_parts.append(f"Title: {title}\n")
            
            if extract_text:
                text = soup.get_text(separator='\n')
                clean_text = re.sub(r'\n\s*\n', '\n', text).strip() # More robust newline cleaning
                clean_text = re.sub(r'[ \t]+', ' ', clean_text) # Consolidate multiple spaces/tabs
                
                if len(clean_text) > 8000: # Truncation
                    clean_text = clean_text[:7950] + "... [content truncated]"
                result_parts.append(f"Content:\n{clean_text}\n")
            
            if extract_links:
                links_found = []
                for link_tag in soup.find_all('a', href=True):
                    href = link_tag['href']
                    
                    # Resolve relative URLs
                    if href.startswith('/'):
                        parsed_original_url = urlparse(url)
                        base_url = f"{parsed_original_url.scheme}://{parsed_original_url.netloc}"
                        href = base_url + href
                    elif not href.startswith(('http://', 'https://')):
                        # If it's not absolute and not starting with /, it might be relative to the current path
                        href = urlparse(url)._replace(path=href).geturl()
                        
                    link_text = link_tag.get_text().strip()
                    if href.startswith(('http://', 'https://')) and link_text:
                        links_found.append(f"- {link_text}: {href}")
                
                if links_found:
                    result_parts.append("Links found:\n" + "\n".join(links_found) + "\n")
                else:
                    result_parts.append("No links extracted or found.\n")
            
            return "\n".join(result_parts).strip()

        except requests.exceptions.Timeout:
            return f"Error: The request to {url} timed out."
        except requests.exceptions.RequestException as e:
            return f"Error fetching {url}: {e}"
        except Exception as e:
            return f"An unexpected error occurred while processing {url}: {e}"

# Example usage (optional, for testing)
if __name__ == '__main__':
    tool = VisitWebpageTool()
    
    # Test case 1: Extract everything
    # result1 = tool.invoke({"url": "https://example.com", "extract_links": True, "extract_text": True})
    # print("--- Test Case 1 (All) ---")
    # print(result1)

    # Test case 2: Extract only text
    # result2 = tool.invoke({"url": "https://www.google.com", "extract_links": False, "extract_text": True})
    # print("\n--- Test Case 2 (Text Only) ---")
    # print(result2)

    # Test case 3: Extract only links
    # result3 = tool.invoke({"url": "https://www.wikipedia.org", "extract_links": True, "extract_text": False})
    # print("\n--- Test Case 3 (Links Only) ---")
    # print(result3)
    
    # Test case 4: Non-existent URL
    # result4 = tool.invoke({"url": "https://nonexistentdomain12345.com"})
    # print("\n--- Test Case 4 (Error) ---")
    # print(result4)
    pass
