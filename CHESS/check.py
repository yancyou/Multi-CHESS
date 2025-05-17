import json
import sqlite3
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import openai
from tqdm import tqdm
import os
from cp import validate_sql_results
from dotenv import load_dotenv

# Load OpenAI API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")
client = openai.Client(api_key=api_key)

# Database connection cache
DB_CONNECTIONS = {}
DB_FILE_PATHS = {}
SCHEMA_FILE_PATHS = {}

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

def execute_query(db_id: str, query: str) -> Tuple[List[Dict], str]:
    """Execute SQL query and return results + schema"""
    try:
        conn = get_db_connection(db_id)
        df = pd.read_sql_query(query, conn)
        return df.to_dict('records'), str(df.dtypes.to_dict())
    except Exception as e:
        return [], f"Error executing query: {str(e)}"

def load_schema(db_id: str, schema_dir: str = "data/spider_data/test_database") -> str:
    """Load schema file by scanning schema_dir for .sql files"""
    global SCHEMA_FILE_PATHS
    if not SCHEMA_FILE_PATHS:
        for root, _, files in os.walk(schema_dir):
            for fname in files:
                if fname.endswith(".sql"):
                    id = os.path.splitext(fname)[0]
                    SCHEMA_FILE_PATHS[id] = os.path.join(root, fname)
    if db_id not in SCHEMA_FILE_PATHS:
        return f"Schema not found for DB: {db_id} in {schema_dir}"
    path = SCHEMA_FILE_PATHS[db_id]
    try:
        with open(path, "r", encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error loading schema: {str(e)}"

def generate_merge_prompt(entry: Dict[str, Any], schema: str) -> str:
    """Generate prompt for initial SQL merging"""
    return f"""

 Please analyze whether these SQL queries can be merged into one query, considering the database schema.
 Respond with the following 2 components:
 1. "sql_merged": 【merged SQL (empty if not possible)】
 2. "reason": 【technical justification】

Question: {entry['question']}

SQL1: {entry['sql_1']}
SQL2: {entry['sql_2']}
SQL3: {entry['sql_3']}

Database Schema:
{schema}

Analysis Guidelines:
1. Check for identical table joins and grouping
2. Verify whether the result of aggregation function (SUM/COUNT/AVG/MAX/MIN...) can be replicated
3. Consider UNION ALL/CASE expressions
4. Maintain exact semantics
5. Note that the merged SQL should have same output meaning based on the same condition from the separate two SQLs

+Example 1:
+Question: "Find the number of channels that do not run any program and the names of the remaining channels."
+SQL1: SELECT count(*) FROM channel WHERE channel_id NOT IN (SELECT channel_id FROM program)
+SQL2: SELECT name FROM channel WHERE channel_id IN (SELECT channel_id FROM program)
+
+Merged result:
+-- Count of channels without any programs
+SELECT 
+    'Channels Without Programs' AS Category,
+    COUNT(*) AS Channel_Count,
+    CAST(NULL AS VARCHAR) AS Channel_Name
+FROM channel
+WHERE channel_id NOT IN (SELECT DISTINCT channel_id FROM program)
+
+UNION ALL
+
+-- Names of channels with at least one program
+SELECT 
+    'Channels With Programs' AS Category,
+    CAST(NULL AS INT) AS Channel_Count,
+    name AS Channel_Name
+FROM channel
+WHERE channel_id IN (SELECT DISTINCT channel_id FROM program);
+
+Example 2:
+Question: "What is the most common type of engine used by drivers, and how many drivers use each constructor?"
+SQL1: SELECT Engine FROM driver GROUP BY Engine ORDER BY COUNT(*) DESC LIMIT 1
+SQL2: SELECT CONSTRUCTOR, COUNT(*) FROM driver GROUP BY CONSTRUCTOR
+
+Merged result:
+SELECT 
+    'Most Common Engine' AS Result_Type,
+    Engine AS Engine_Info,
+    CAST(NULL AS VARCHAR) AS Constructor,
+    CAST(NULL AS INT) AS Driver_Count
+FROM driver
+GROUP BY Engine
+ORDER BY COUNT(*) DESC
+LIMIT 1
+
+UNION ALL
+
+-- Retrieve number of drivers per constructor
+SELECT 
+    'Driver Count Per Constructor' AS Result_Type,
+    CAST(NULL AS VARCHAR) AS Engine_Info,
+    CONSTRUCTOR AS Constructor,
+    COUNT(*) AS Driver_Count
+FROM driver
+GROUP BY CONSTRUCTOR;
+

Return JSON format only:
{{
    "sql_merged": "",
    "reason": ""
}}
"""

# This result is directly based on querying results, could potentially be improved by using the response dictionary from cp.py
def generate_validation_prompt_with_results(
    entry: Dict[str, Any], 
    results_1: List[Dict],
    results_2: List[Dict],
    results_merged: List[Dict],
    schemas: Dict[str, str]
) -> str:
    """Generate advanced validation prompt with execution results"""
    return f"""
Please rigorously validate if the merged SQL query produces equivalent results to running the original queries separately.

Original Question: {entry['question']}

--- Original Queries ---
SQL1: {entry['sql_1']}
Result Schema: {schemas['schema_1']}
First 3 Rows: {results_1[:3]}

SQL2: {entry['sql_2']}
Result Schema: {schemas['schema_2']}
First 3 Rows: {results_2[:3]}

--- Merged Query ---
SQL3: {entry['sql_3']}
Result Schema: {schemas['schema_3']}
First 3 Rows: {results_merged[:3]}

Validation Criteria:
1. Result equivalence (same rows in any order)
2. Schema compatibility (matching column names/types)
3. Edge case handling (NULLs, duplicates)
4. Semantic correctness for the original question
5. The merged result must represent the question requirements and convince the user

Technical Analysis:
- Compare result cardinality
- Verify all original conditions are preserved
- Check for type coercion issues
- Validate sorting/grouping behavior
- The merged result shall represent the question requirements and convince the user

Respond in JSON format with:
{{
    "is_valid": bool,
    "issues": "detailed technical analysis",
    "result_equivalence": "full/partial/none",
    "schema_compatibility": bool
}}
"""

def llm_generate(input: str) -> Dict[str, Any]:
    """Enhanced LLM generation with robust JSON parsing"""
    try:
        result = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # Using OpenAI's gpt-4o-mini-2024-07-18 model
            messages=[
                {
                    "content": "You are an expert in analyzing SQLs. Please respond with valid JSON only.",
                    "role": "system",
                },
                {
                    "content": input,
                    "role": "user",
                }
            ],
            max_tokens=10000,
            temperature=0.1,
        )
        output = result.choices[0].message.content
        
        # Debug output
        print("Raw response:", output)
        
        # Robust JSON parsing
        try:
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                return json.loads(output[start_idx:end_idx])
            return {"error": "No JSON found in response"}
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            return {"error": f"Invalid JSON: {str(e)}"}
            
    except Exception as e:
        print(f"API call failed: {str(e)}")
        return {"error": str(e)}

def validate_with_execution(entry: Dict[str, Any], db_dir: str) -> Dict[str, Any]:
    """Perform actual query execution and advanced validation"""
    if not entry.get("sql_merged"):
        return entry
    
    # Execute all queries
    results_1, schema_1 = execute_query(entry["db_id"], entry["sql_1"])
    results_2, schema_2 = execute_query(entry["db_id"], entry["sql_2"])
    results_merged, schema_3 = execute_query(entry["db_id"], entry["sql_merged"])
    
    # Prepare validation prompt
    schemas = {
        "schema_1": schema_1,
        "schema_2": schema_2,
        "schema_3": schema_3
    }

    print(schemas)
    
    prompt = generate_validation_prompt_with_results(
        entry, results_1, results_2, results_merged, schemas
    )
    
    # Call LLM for validation
    validation_result = llm_generate(prompt)
    
    if validation_result and not validation_result.get("error"):
        entry.update({
            "validation_method": "execution_based",
            "results_1_sample": results_1[:3],
            "results_2_sample": results_2[:3],
            "results_merged_sample": results_merged[:3],
            "is_valid": validation_result.get("is_valid", False),
            "validation_issues": validation_result.get("issues", ""),
            "result_equivalence": validation_result.get("result_equivalence", "none"),
            "schema_compatibility": validation_result.get("schema_compatibility", False)
        })
    
    return entry

def process_entry(entry: Dict[str, Any], schema_dir: str, db_dir: str) -> Dict[str, Any]:
    """Full processing pipeline for an entry with up to 3 merge/validate retries."""
    schema = load_schema(entry["db_id"], schema_dir)
    max_attempts = 3
    attempt = 0
    previous_issues = None
    while attempt < max_attempts:
        # Step 1: Generate or regenerate merged SQL
        merge_prompt = generate_merge_prompt(entry, schema)
        if previous_issues:
            merge_prompt += f"\n-- Previous validation issues: {previous_issues}\nPlease revise the merged SQL to address these issues."
        merge_result = llm_generate(merge_prompt)
        if not merge_result or merge_result.get("error"):
            break
        entry["sql_merged"] = merge_result.get("sql_merged", "")
        entry["reason"] = merge_result.get("reason", "")
        # Step 2: Validate merged SQL against actual DB via cp.py
        try:
            conn = get_db_connection(entry["db_id"], db_dir)
            cp_result = validate_sql_results(conn, entry["sql_1"], entry["sql_2"], entry["sql_merged"])
            entry["is_valid"] = cp_result.get("isvalid", False)
        except Exception as e:
            # Execution error (e.g., syntax error), consider as validation failure, record exception and retry
            entry["is_valid"] = False
            previous_issues = str(e)
            attempt += 1
            continue
        # If valid, stop retry
        if entry["is_valid"]:
            break
        # Otherwise record generic issue and retry
        previous_issues = "SQL execution mismatch"
        attempt += 1
    # Remove detailed validation fields, keep only is_valid
    for _key in [
        "validation_method",
        "results_1_sample",
        "results_2_sample",
        "results_merged_sample",
        "validation_issues",
        "result_equivalence",
        "schema_compatibility"
    ]:
        entry.pop(_key, None)
    return entry

def main(input_path: str = "data/spider_data/small_test.json",
         output_path: str = "data/spider_data/small_test_val.json",
         schema_dir: str = "data/spider_data/test_database",
         db_dir: str = "data/spider_data/test_database"):
    """Main execution function"""
    processed_data = []

    # Load input data
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Process each entry and write results incrementally
    for entry in tqdm(data):
        try:
            processed_entry = process_entry(entry, schema_dir, db_dir)
        except Exception as e:
            print(f"Error processing entry {entry.get('db_id', '')}: {e}")
            processed_entry = {**entry, "error": str(e)}
        processed_data.append(processed_entry)
        # Write current results to output file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing output file: {e}")

    print(f"Processed {len(processed_data)} entries. Output saved to {output_path}")
    # Close database connections
    for conn in DB_CONNECTIONS.values():
        conn.close()

if __name__ == "__main__":
    # Main processing workflow
    main()