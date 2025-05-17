import sqlite3
import json
from itertools import zip_longest
import os

def execute_sql(conn, sql):
    """执行SQL语句，返回结果列表"""
    cursor = conn.cursor()
    cursor.execute(sql)
    return cursor.fetchall()

def normalize_result(result):
    """将结果转换为扁平列表（适用于单列结果）"""
    return [row[0] for row in result if row[0] is not None]

def compare_columns(expected_columns, merged_columns):
    """
    检查merged_columns中是否存在与expected_columns一一匹配的列（值和顺序）
    :param expected_columns: list of lists，每个元素是期望的列
    :param merged_columns: list of lists，合并结果中的列
    :return: True or False
    """
    # 复制merged_columns以便删除已匹配的列
    candidate_cols = [col[:] for col in merged_columns]

    for expected_col in expected_columns:
        matched = False
        for i, merged_col in enumerate(candidate_cols):
            if len(expected_col) != len(merged_col):
                continue
            
            if all(e == m for e, m in zip(expected_col, merged_col)):
                # 匹配成功，从候选列中移除该列
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
    验证合并后的SQL是否与原始两个SQL结果一致
    :param conn: 数据库连接
    :param sql_1: 第一个SQL语句
    :param sql_2: 第二个SQL语句
    :param sql_merged: 合并后的SQL语句
    :return: True or False
    """
    # 执行三个SQL语句
    result_1 = execute_sql(conn, sql_1)
    print("PASS 1")
    
    result_2 = execute_sql(conn, sql_2)
    print("PASS 2")
   
    result_merged = execute_sql(conn, sql_merged)
    
    # 转换为列优先格式（每列是一个列表）
    def to_columns(result):
        return [list(col) for col in zip(*result)]

    # 提取原始SQL结果的列
    expected_columns = to_columns(result_1) + to_columns(result_2)
    print(expected_columns)
    # 提取合并后的列（忽略 Info_Type 列）
    merged_columns = to_columns(result_merged)[:]

    # 从合并结果中移除 NULL 值
    merged_columns_cleaned = [
        [val for val in col if val is not None]
        for col in merged_columns
    ]

    print(merged_columns_cleaned)

    # 从原始结果中提取列（同样移除 NULL）
    expected_columns_cleaned = [
        [val for val in col if val is not None]
        for col in expected_columns
    ]

    # 比较列
    isvalid = compare_columns(expected_columns_cleaned, merged_columns_cleaned)
    validation_result = {
        "sql_1": to_columns(result_1),
        "sql_2": to_columns(result_2),
        "sql_merged": merged_columns,
        "isvalid": isvalid
    }
    return validation_result

# 示例调用
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
    # 加载 JSON 文件（使用 check.py 的 output_path）
    input_path = r"data/spider_data/small_test_val.json"
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    
    #db_id = "bbc_channels"
    #conn = get_db_connection(db_id)

    # 遍历每个问题
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