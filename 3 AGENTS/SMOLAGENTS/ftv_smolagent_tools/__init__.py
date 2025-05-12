"""
Tools package for the FTV Smol Agent.
This package contains all the tools used by the agent.
"""

# Import tools from smolagents
from smolagents import WikipediaSearchTool, VisitWebpageTool

# Import our custom tools
from .search_tools import GuardianSearchTool, tavily_search
from .image_tools import PixtralImageQueryTool, analyze_image
from .data_tools import excel_data_tool

# Dictionary of all available tools
__all__ = [
    # From smolagents
    "WikipediaSearchTool",
    "VisitWebpageTool",
    
    # Custom tools
    "GuardianSearchTool",
    "tavily_search",
    "PixtralImageQueryTool",
    "analyze_image",
    
    # Data tools
    "excel_data_tool"
]
