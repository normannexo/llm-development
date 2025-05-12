"""
Data tools for the FTV Smol Agent.
This module contains tools for working with data files like Excel, CSV, etc.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional
from smolagents.tools import Tool
import openpyxl


class ExcelDataTool(Tool):
    """Tool for opening Excel files and returning pandas DataFrames."""
    
    name = "excel_data_tool"
    description = "Opens an Excel file and returns its contents as a pandas DataFrame. Use this tool when you need to analyze Excel data."
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the Excel file to be opened"
        },
        "sheet_name": {
            "type": "string",
            "description": "Name of the sheet to read. If not provided, reads the first sheet.",
            "required": False,
            "nullable": True
        }
    }
    output_type = "object"
    
    def __init__(self):
        super().__init__()
    
    def forward(self, file_path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """Required method for smolagents Tool compatibility"""
        return self.__call__(file_path, sheet_name)
    
    def __call__(self, file_path: str, sheet_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Open an Excel file and return its contents as a pandas DataFrame.
        
        Args:
            file_path (str): Path to the Excel file
            sheet_name (str, optional): Name of the sheet to read. If None, reads the first sheet.
            
        Returns:
            Dict containing:
                - 'dataframe': The pandas DataFrame with the Excel data
                - 'shape': Tuple with (rows, columns) of the DataFrame
                - 'columns': List of column names
                - 'sheet_name': The sheet that was read
                - 'file_name': The name of the file that was read
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    "error": f"File not found: {file_path}",
                    "success": False
                }
            
            # Check if file is an Excel file
            if not file_path.endswith(('.xlsx', '.xls', '.xlsm')):
                return {
                    "error": f"Not an Excel file: {file_path}",
                    "success": False
                }
            
            # Read the Excel file
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            
            # Get file name without path
            file_name = os.path.basename(file_path)
            
            # Return the DataFrame and metadata
            return {
                "dataframe": df,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "sheet_name": sheet_name or "default",
                "file_name": file_name,
                "success": True,
                "summary": f"Successfully loaded Excel file {file_name} with {df.shape[0]} rows and {df.shape[1]} columns."
            }
        
        except Exception as e:
            return {
                "error": f"Error reading Excel file: {str(e)}",
                "success": False
            }


# Create an instance of the tool
excel_data_tool = ExcelDataTool()
