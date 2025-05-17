import logging
import os
from typing import List, Dict
from pathlib import Path
from database_utils.execution import execute_sql
def find_sqlite_files(db_root_directory: str) -> Dict[str, str]:
    """
       Finds all .sqlite files in the given directory and its subdirectories.

       Args:
           db_root_directory (str): The root directory to search for .sqlite files.

       Returns:
           Dict[str, str]: A dictionary mapping db_id (directory name) to full .sqlite file paths.
       """
    db_files = {}
    for sqlite_file in Path(db_root_directory).rglob('*.sqlite'):
        db_id = sqlite_file.parent.name
        db_files[db_id] = str(sqlite_file.resolve())
    return db_files

def get_db_all_tables(db_path: str) -> List[str]:
    """
    Retrieves all table names from the database.
    
    Args:
        db_path (str): The path to the database file.
        
    Returns:
        List[str]: A list of table names.
    """
    try:
        raw_table_names = execute_sql(db_path, "SELECT name FROM sqlite_master WHERE type='table';")
        return [table[0].replace('\"', '').replace('`', '') for table in raw_table_names if table[0] != "sqlite_sequence"]
    except Exception as e:
        logging.error(f"Error in get_db_all_tables: {e}")
        raise e

def get_table_all_columns(db_path: str, table_name: str) -> List[str]:
    """
    Retrieves all column names for a given table.
    
    Args:
        db_path (str): The path to the database file.
        table_name (str): The name of the table.
        
    Returns:
        List[str]: A list of column names.
    """
    try:
        table_info_rows = execute_sql(db_path, f"PRAGMA table_info(`{table_name}`);")
        return [row[1].replace('\"', '').replace('`', '') for row in table_info_rows]
    except Exception as e:
        logging.error(f"Error in get_table_all_columns: {e}\nTable: {table_name}")
        raise e

def get_db_schema(db_path: str) -> Dict[str, List[str]]:
    """
    Retrieves the schema of the database.
    
    Args:
        db_path (str): The path to the database file.
        
    Returns:
        Dict[str, List[str]]: A dictionary mapping table names to lists of column names.
    """
    try:
        table_names = get_db_all_tables(db_path)
        return {table_name: get_table_all_columns(db_path, table_name) for table_name in table_names}
    except Exception as e:
        logging.error(f"Error in get_db_schema: {e}")
        raise e
def process_all_databases(db_root_directory: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Processes all .sqlite databases in the specified directory and retrieves their schemas.

    Args:
        db_root_directory (str): The root directory containing .sqlite files.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary mapping db_id to database schema.
    """
    schemas = {}
    db_files = find_sqlite_files(db_root_directory)
    for db_id, db_path in db_files.items():
        logging.info(f"Processing database {db_id} at {db_path}")
        try:
            schemas[db_id] = get_db_schema(db_path)
        except Exception as e:
            logging.error(f"Failed to process database {db_id}: {e}")
    return schemas
