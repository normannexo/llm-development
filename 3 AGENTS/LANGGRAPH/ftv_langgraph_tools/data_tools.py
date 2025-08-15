import os
from typing import Optional, Type, Any, Dict
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# Ensure openpyxl is installed for .xlsx support if not already handled
# try:
#     import openpyxl
# except ImportError:
#     # Handle missing openpyxl, e.g., by raising an error or logging a warning
#     # For now, we assume it's available if pandas is used for Excel files.
#     pass

class ExcelDataInput(BaseModel):
    """Input schema for ExcelDataTool."""
    file_path: str = Field(description="Path to the Excel file to be opened.")
    sheet_name: Optional[str] = Field(
        default=None, 
        description="Name of the sheet to read. If not provided, reads the first sheet."
    )
    head_rows: Optional[int] = Field(default=5, description="Number of rows from the head of the DataFrame to return in JSON.")

class ExcelDataTool(BaseTool):
    """
    Opens an Excel file, returns a summary and the head of its content as a JSON string.
    Use this tool when you need to inspect the contents of an Excel file.
    """
    
    name: str = "excel_data_inspector"
    description: str = (
        "Opens an Excel file (.xlsx, .xls, .xlsm) and returns a summary (shape, columns) "
        "and the first few rows (head) as a JSON string. "
        "Specify the file_path and optionally the sheet_name and number of head_rows to display."
    )
    args_schema: Type[BaseModel] = ExcelDataInput

    def _run(self, file_path: str, sheet_name: Optional[str] = None, head_rows: Optional[int] = 5) -> str:
        """
        Opens an Excel file, returns a summary and the head of its content as a JSON string.
        """
        try:
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}"
            
            if not file_path.lower().endswith(('.xlsx', '.xls', '.xlsm')):
                return f"Error: Not a recognized Excel file extension: {file_path}. Supported: .xlsx, .xls, .xlsm"

            # Explicitly use openpyxl for .xlsx, pandas default might try xlrd for .xls
            engine = 'openpyxl' if file_path.lower().endswith('.xlsx') else None
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)
            
            file_name = os.path.basename(file_path)
            
            # Try to get actual sheet name if not provided
            actual_sheet_name = sheet_name
            if actual_sheet_name is None:
                if isinstance(df, dict): # If multiple sheets are read when sheet_name=None
                    actual_sheet_name = list(df.keys())[0]
                    df = df[actual_sheet_name]
                else: # Single sheet read
                    # For older pandas, sheet name might not be in attrs. Defaulting then.
                    actual_sheet_name = getattr(df, 'attrs', {}).get('sheet_name', 'first sheet')

            summary = (
                f"Successfully loaded Excel file: '{file_name}' (Sheet: '{actual_sheet_name}')\n"
                f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
                f"Columns: {df.columns.tolist()}\n"
            )
            
            current_head_rows = head_rows if head_rows is not None else 5
            df_head_json = df.head(current_head_rows).to_json(orient="records", date_format="iso", default_handler=str)

            return f"{summary}\nData (first {current_head_rows} rows):\n{df_head_json}"

        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except pd.errors.EmptyDataError:
            return f"Error: The Excel file or sheet '{sheet_name if sheet_name else 'default'}' at {file_path} is empty."
        except ValueError as ve:
            return f"Error reading Excel file '{file_path}': {ve}. Ensure it's a valid Excel file and sheet name is correct."
        except ImportError as ie:
            if 'openpyxl' in str(ie).lower():
                 return "Error: The 'openpyxl' package is required for .xlsx files. Please install it (`pip install openpyxl`)."
            elif 'xlrd' in str(ie).lower():
                 return "Error: The 'xlrd' package is required for .xls files. Please install it (`pip install xlrd`)."
            return f"Import error while reading Excel file: {ie}"
        except Exception as e:
            return f"Error processing Excel file '{file_path}': {e}"
