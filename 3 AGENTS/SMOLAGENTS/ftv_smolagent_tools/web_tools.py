"""
Web tools for the FTV Smol Agent.
This module contains tools for interacting with web pages.
"""

from smolagents import Tool
import requests
from bs4 import BeautifulSoup
import re


class VisitWebpageTool(Tool):
    """Tool for visiting and extracting content from web pages."""
    
    name = "visit_webpage"
    description = "Visits a webpage and extracts its content."
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the webpage to visit."
        },
        "extract_links": {
            "type": "boolean",
            "description": "Whether to extract links from the webpage.",
            "nullable": True
        },
        "extract_text": {
            "type": "boolean",
            "description": "Whether to extract text from the webpage.",
            "nullable": True
        }
    }
    output_type = "string"
    
    def forward(self, url: str, extract_links: bool = True, extract_text: bool = True):
        """
        Visit a webpage and extract its content.
        
        Args:
            url: The URL of the webpage to visit
            extract_links: Whether to extract links from the webpage
            extract_text: Whether to extract text from the webpage
            
        Returns:
            Extracted content from the webpage
        """
        try:
            # Send a GET request to the URL
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            result = f"Visited: {url}\n\n"
            
            # Extract the page title
            title = soup.title.string if soup.title else "No title found"
            result += f"Title: {title}\n\n"
            
            # Extract text if requested
            if extract_text:
                # Get the text content
                text = soup.get_text(separator='\n')
                
                # Clean up the text (remove excessive newlines and whitespace)
                clean_text = re.sub(r'\n+', '\n', text).strip()
                clean_text = re.sub(r'\s+', ' ', clean_text)
                
                # Truncate if too long (limit to ~8000 characters)
                if len(clean_text) > 8000:
                    clean_text = clean_text[:8000] + "... [content truncated]"
                
                result += "Content:\n" + clean_text + "\n\n"
            
            # Extract links if requested
            if extract_links:
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    # Convert relative URLs to absolute URLs
                    if href.startswith('/'):
                        from urllib.parse import urlparse
                        parsed_url = urlparse(url)
                        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        href = base_url + href
                    
                    # Only include http/https links
                    if href.startswith(('http://', 'https://')):
                        link_text = link.get_text().strip()
                        if link_text:  # Only include links with text
                            links.append((link_text, href))
                
                if links:
                    result += "Links:\n"
                    # Limit to 20 links to avoid overwhelming output
                    for i, (text, href) in enumerate(links[:20], 1):
                        result += f"{i}. {text}: {href}\n"
                    
                    if len(links) > 20:
                        result += f"... and {len(links) - 20} more links\n"
            
            return result
            
        except requests.exceptions.RequestException as e:
            return f"Error visiting webpage: {str(e)}"
        except Exception as e:
            return f"Error processing webpage: {str(e)}"
