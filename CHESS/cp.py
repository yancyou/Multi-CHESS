import sqlite3
import json
from itertools import zip_longest
import os

def execute_sql(conn, sql):
    """Execute SQL statement and return result list"""
    cursor = conn.cursor()
    cursor.execute(sql)
    return cursor.fetchall()

def normalize_result(result):
    """Convert result to a flat list (applicable for single column results)"""
    return [row[0] for row in result if row[0] is not None]

def compare_columns(expected_columns, merged_columns):
    """
    Check if there are columns in merged_columns that match one-to-one with expected_columns (values and order)
    :param expected_columns: list of lists, each element is an expected column
    :param merged_columns: list of lists, columns in the merged result
    :return: True or False
    """
    # Copy merged_columns to allow removal of matched columns
    candidate_cols = [col[:] for col in merged_columns]

    for expected_col in expected_columns:
        matched = False
        for i, merged_col in enumerate(candidate_cols):
            if len(expected_col) != len(merged_col):
                continue
            
            if all(e == m for e, m in zip(expected_col, merged_col)):
                # Match successful, remove this column from candidate columns
                print(f"Matched! we have {len(expected_col)}")
                print(expected_col)
                print(merged_col)
                candidate_cols.pop(i)
                matched = True
                break
        if not matched:
            return False
    return True

def validate_sql_results(conn, sql_1, sql_2, sql_merged):
    """
    Validate if the merged SQL result is consistent with the results of the original two SQL queries
    :param conn: database connection
    :param sql_1: first SQL statement
    :param sql_2: second SQL statement
    :param sql_merged: merged SQL statement
    :return: Validation result dictionary
    """
    # Execute three SQL statements
    result_1 = execute_sql(conn, sql_1)
    print("PASS 1")
    
    result_2 = execute_sql(conn, sql_2)
    print("PASS 2")
   
    result_merged = execute_sql(conn, sql_merged)
    
    # Convert to column-priority format (each column is a list)
    def to_columns(result):
        return [list(col) for col in zip(*result)]

    # Extract columns from original SQL results
    expected_columns = to_columns(result_1) + to_columns(result_2)
    print(expected_columns)
    # Extract columns from merged result (ignore Info_Type column)
    merged_columns = to_columns(result_merged)[:]

    # Remove NULL values from merged results
    merged_columns_cleaned = [
        [val for val in col if val is not None]
        for col in merged_columns
    ]

    print(merged_columns_cleaned)

    # Extract columns from original results (also remove NULL)
    expected_columns_cleaned = [
        [val for val in col if val is not None]
        for col in expected_columns
    ]

    # Compare columns
    isvalid = compare_columns(expected_columns_cleaned, merged_columns_cleaned)
    validation_result = {
        "sql_1": to_columns(result_1),
        "sql_2": to_columns(result_2),
        "sql_merged": merged_columns,
        "isvalid": isvalid
    }
    return validation_result

# Example call
DB_CONNECTIONS = {}
DB_FILE_PATHS = {}
def get_db_connection(db_id: str, db_dir: str = r"data/spider_data/test_database") -> sqlite3.Connection:
    """Get or create a database connection by scanning db_dir for .sqlite files"""
    global DB_FILE_PATHS, DB_CONNECTIONS
    # Build mapping from db_id to file path once
    if not DB_FILE_PATHS:
        for root, _, files in os.walk(db_dir):
            for fname in files:
                if fname.endswith(".sqlite"):
                    id = os.path.splitext(fname)[0]
                    DB_FILE_PATHS[id] = os.path.join(root, fname)
    if db_id not in DB_FILE_PATHS:
        raise ValueError(f"No sqlite file found for DB ID: {db_id} in {db_dir}")
    if db_id not in DB_CONNECTIONS:
        DB_CONNECTIONS[db_id] = sqlite3.connect(DB_FILE_PATHS[db_id])
    return DB_CONNECTIONS[db_id]

if __name__ == "__main__":
    # Load JSON file (using check.py's output_path)
    input_path = r"data/spider_data/small_test_val.json"
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    
    #db_id = "bbc_channels"
    #conn = get_db_connection(db_id)

    # Iterate through each question
    for entry in data:
        question = entry["question"]
        sql_1 = entry["sql_1"]
        sql_2 = entry["sql_2"]
        sql_merged = entry["sql_merged"]
        db_id = entry["db_id"]
        conn = get_db_connection(db_id)

        print(f"Processing question: {question}")
        try:
            result = validate_sql_results(conn, sql_1, sql_2, sql_merged)
            print(result)
        except Exception as e:
            print(f"Error executing queries: {e}")