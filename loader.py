import pandas as pd
from langchain_community.document_loaders import UnstructuredExcelLoader
from typing import List, Dict, Any, Optional
import logging
import sys
import os
import json
from datetime import datetime

class ExcelDataLoader:
    """
    A comprehensive module to load Excel data using both pandas and LangChain,
    and convert rows to dictionaries with column names as keys.
    All functionality is contained within this class.
    """
    
    def __init__(self, file_path: str, log_level: int = logging.INFO):
        self.file_path = file_path
        self._setup_logging(log_level)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Log initialization
        self.logger.info(f"ExcelDataLoader initialized with file: {file_path}")
        self.logger.debug(f"Absolute file path: {os.path.abspath(file_path)}")
    
    def _setup_logging(self, log_level: int):
        """Setup comprehensive logging configuration"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'excel_loader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
    
    def _validate_file(self) -> bool:
        """
        Validate that the Excel file exists and is accessible.
        
        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            if not os.path.exists(self.file_path):
                self.logger.error(f"File does not exist: {self.file_path}")
                return False
            
            if not os.path.isfile(self.file_path):
                self.logger.error(f"Path is not a file: {self.file_path}")
                return False
            
            file_size = os.path.getsize(self.file_path)
            self.logger.debug(f"File size: {file_size} bytes")
            
            if file_size == 0:
                self.logger.error("File is empty")
                return False
                
            self.logger.info(f"File validation successful: {self.file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"File validation failed: {e}", exc_info=True)
            return False
    
    def load_with_pandas(self) -> List[Dict[str, Any]]:
        """
        Load Excel file using pandas and convert each row to a dictionary.
        
        Returns:
            List of dictionaries where each dict represents a row
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid Excel file
            Exception: For other unexpected errors
        """
        self.logger.info("Starting pandas Excel loading process")
        
        if not self._validate_file():
            raise FileNotFoundError(f"File validation failed: {self.file_path}")
        
        try:
            # Read the Excel file with additional parameters for robustness
            self.logger.debug("Reading Excel file with pandas")
            df = pd.read_excel(
                self.file_path,
                engine='openpyxl',  # Use openpyxl for better xlsx support
                na_values=['', 'NULL', 'null', 'NaN', 'N/A', 'n/a'],
                keep_default_na=True
            )
            
            self.logger.info(f"Successfully read Excel file. Shape: {df.shape}")
            self.logger.debug(f"Columns: {list(df.columns)}")
            self.logger.debug(f"Data types:\n{df.dtypes}")
            
            # Convert NaN values to None for cleaner JSON serialization
            self.logger.debug("Cleaning data - converting NaN to None")
            df_cleaned = df.where(pd.notnull(df), None)
            
            # Log sample data for debugging
            sample_data = df_cleaned.head(3).to_dict('records')
            self.logger.debug(f"Sample data (first 3 rows): {sample_data}")
            
            # Convert DataFrame to list of dictionaries
            rows_as_dicts = df_cleaned.to_dict('records')
            
            self.logger.info(f"Successfully converted {len(rows_as_dicts)} rows to dictionaries")
            return rows_as_dicts
            
        except FileNotFoundError as e:
            self.logger.error(f"Excel file not found: {e}", exc_info=True)
            raise
        except ValueError as e:
            self.logger.error(f"Invalid Excel file format: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during pandas loading: {e}", exc_info=True)
            raise
    
    def load_with_langchain(self) -> List[Dict[str, Any]]:
        """
        Load Excel file using LangChain and convert to list of dictionaries.
        
        Returns:
            List of dictionaries where each dict represents document content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: For other unexpected errors
        """
        self.logger.info("Starting LangChain Excel loading process")
        
        if not self._validate_file():
            raise FileNotFoundError(f"File validation failed: {self.file_path}")
        
        try:
            # Load with LangChain UnstructuredExcelLoader
            self.logger.debug("Loading with UnstructuredExcelLoader")
            loader = UnstructuredExcelLoader(self.file_path)
            documents = loader.load()
            
            self.logger.info(f"Successfully loaded {len(documents)} documents with LangChain")
            
            # Convert documents to dictionaries
            docs_as_dicts = []
            for i, doc in enumerate(documents):
                doc_dict = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': self.file_path,
                    'document_index': i
                }
                docs_as_dicts.append(doc_dict)
                
                # Log document details at debug level
                self.logger.debug(f"Document {i} metadata: {doc.metadata}")
                self.logger.debug(f"Document {i} content preview: {doc.page_content[:100]}...")
            
            self.logger.info(f"Converted {len(docs_as_dicts)} documents to dictionaries")
            return docs_as_dicts
            
        except Exception as e:
            self.logger.error(f"Error during LangChain loading: {e}", exc_info=True)
            raise
    
    def get_rows_as_dicts(self, method: str = 'pandas') -> List[Dict[str, Any]]:
        """
        Main method to get rows as dictionaries.
        
        Args:
            method: 'pandas' for structured data, 'langchain' for document-based extraction
            
        Returns:
            List of dictionaries representing rows
            
        Raises:
            ValueError: If invalid method is specified
        """
        self.logger.info(f"Getting rows as dictionaries using method: {method}")
        
        if method not in ['pandas', 'langchain']:
            error_msg = f"Invalid method: {method}. Must be 'pandas' or 'langchain'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            if method == 'pandas':
                return self.load_with_pandas()
            else:
                return self.load_with_langchain()
                
        except Exception as e:
            self.logger.error(f"Failed to get rows as dictionaries: {e}", exc_info=True)
            raise
    
    def get_top_n_rows(self, n: int = 10, method: str = 'pandas') -> List[Dict[str, Any]]:
        """
        Get top N rows as dictionaries.
        
        Args:
            n: Number of rows to return
            method: 'pandas' or 'langchain'
            
        Returns:
            List of first N rows as dictionaries
            
        Raises:
            ValueError: If n is not positive
        """
        self.logger.info(f"Getting top {n} rows using method: {method}")
        
        if n <= 0:
            error_msg = f"Number of rows must be positive, got: {n}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            all_rows = self.get_rows_as_dicts(method)
            
            if n > len(all_rows):
                self.logger.warning(
                    f"Requested {n} rows but only {len(all_rows)} available. "
                    f"Returning all {len(all_rows)} rows."
                )
                n = len(all_rows)
            
            result = all_rows[:n]
            self.logger.info(f"Successfully retrieved top {len(result)} rows")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get top {n} rows: {e}", exc_info=True)
            raise
    
    def get_row_by_index(self, index: int, method: str = 'pandas') -> Dict[str, Any]:
        """
        Get a specific row by index as dictionary.
        
        Args:
            index: Row index (0-based)
            method: 'pandas' or 'langchain'
            
        Returns:
            Dictionary representing the row
            
        Raises:
            IndexError: If index is out of range
        """
        self.logger.info(f"Getting row at index {index} using method: {method}")
        
        try:
            all_rows = self.get_rows_as_dicts(method)
            
            if index < 0 or index >= len(all_rows):
                error_msg = f"Index {index} out of range. Total rows: {len(all_rows)}"
                self.logger.error(error_msg)
                raise IndexError(error_msg)
            
            result = all_rows[index]
            self.logger.info(f"Successfully retrieved row at index {index}")
            self.logger.debug(f"Row {index} data: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get row at index {index}: {e}", exc_info=True)
            raise
    
    def get_column_names(self) -> List[str]:
        """
        Get column names from the Excel file.
        
        Returns:
            List of column names
            
        Raises:
            Exception: If unable to read column names
        """
        self.logger.info("Getting column names from Excel file")
        
        if not self._validate_file():
            raise FileNotFoundError(f"File validation failed: {self.file_path}")
        
        try:
            # Read just the header to get column names
            df = pd.read_excel(self.file_path, nrows=0, engine='openpyxl')
            columns = df.columns.tolist()
            
            self.logger.info(f"Successfully retrieved {len(columns)} column names")
            self.logger.debug(f"Column names: {columns}")
            return columns
            
        except Exception as e:
            self.logger.error(f"Failed to get column names: {e}", exc_info=True)
            raise
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the Excel file.
        
        Returns:
            Dictionary with file information
        """
        self.logger.info("Getting comprehensive file information")
        
        try:
            file_stats = os.stat(self.file_path)
            
            info = {
                'file_path': self.file_path,
                'file_size_bytes': file_stats.st_size,
                'file_size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                'last_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'absolute_path': os.path.abspath(self.file_path)
            }
            
            # Try to get row and column count
            try:
                df_sample = pd.read_excel(self.file_path, nrows=5, engine='openpyxl')
                info['columns_count'] = len(df_sample.columns)
                info['sample_columns'] = df_sample.columns.tolist()
            except Exception as e:
                self.logger.warning(f"Could not read sample data for info: {e}")
                info['columns_count'] = None
                info['sample_columns'] = None
            
            self.logger.info(f"File info collected: {info}")
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get file info: {e}", exc_info=True)
            raise
    
    def demonstrate_capabilities(self) -> None:
        """
        Demonstrate the capabilities of the ExcelDataLoader.
        Useful for testing and understanding the module's functionality.
        """
        self.logger.info("=" * 60)
        self.logger.info("EXCEL DATA LOADER CAPABILITIES DEMONSTRATION")
        self.logger.info("=" * 60)
        
        try:
            # Get file information
            file_info = self.get_file_info()
            self.logger.info(f"File information: {file_info}")
            
            # Get column names
            columns = self.get_column_names()
            self.logger.info(f"Column names: {columns}")
            
            # Load with pandas (structured data)
            self.logger.info("Loading with pandas (structured data)")
            pandas_rows = self.get_top_n_rows(n=3, method='pandas')
            self.logger.info(f"Loaded {len(pandas_rows)} rows with pandas")
            
            for i, row_dict in enumerate(pandas_rows):
                self.logger.info(f"Pandas Row {i} - {dict(list(row_dict.items()))}")
            
            # Load with LangChain (document-based)
            self.logger.info("Loading with LangChain (document-based)")
            langchain_docs = self.get_top_n_rows(n=2, method='langchain')
            self.logger.info(f"Loaded {len(langchain_docs)} documents with LangChain")
            
            for i, doc_dict in enumerate(langchain_docs):
                self.logger.info(f"LangChain Document {i} - Content preview: {doc_dict['content'][:100]}...")
                self.logger.debug(f"Document {i} full metadata: {doc_dict['metadata']}")
            
            self.logger.info("Excel Data Loader demonstration completed successfully")
            
        except Exception as e:
            self.logger.error(f"Demonstration failed: {e}", exc_info=True)
            raise
    
    def quick_load(self, method: str = 'pandas', n_rows: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Quick utility method to load Excel file and return rows as dictionaries.
        
        Args:
            method: 'pandas' for structured data, 'langchain' for document extraction
            n_rows: Number of rows to return (None for all rows)
            
        Returns:
            List of dictionaries where each dict represents a row
        """
        self.logger.info(f"Quick loading Excel file with method: {method}, n_rows: {n_rows}")
        
        try:
            if n_rows is None:
                result = self.get_rows_as_dicts(method)
            else:
                result = self.get_top_n_rows(n_rows, method)
            
            self.logger.info(f"Quick load completed. Returned {len(result)} items")
            return result
            
        except Exception as e:
            self.logger.error(f"Quick load failed: {e}", exc_info=True)
            raise
class JsonDataLoader:
    def __init__(self, json_paths: List[str]):
        self.json_paths = json_paths
    
    def get_all_nodes(self) -> List[Dict]:
        nodes = []
        for path in self.json_paths:
            if "nodes" in path:  # Filter node files
                with open(path, 'r', encoding="UTF-8") as f:
                    data = json.load(f)
                    nodes.extend(data)  # Assume each JSON is a list of dicts
        return nodes
    
    def get_all_edges(self) -> List[Dict]:
        edges = []
        for path in self.json_paths:
            if "edges" in path:  # Filter edge files
                with open(path, 'r', encoding="UTF-8") as f:
                    data = json.load(f)
                    edges.extend(data)
        return edges