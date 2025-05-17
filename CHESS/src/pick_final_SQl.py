import json
import re
import html  # 添加html模块用于解码HTML实体
import os
import sys
import glob
from datetime import datetime

def extract_sql_queries(input_files, output_files):

    # Load the json files
    with open(input_files, 'r', encoding='utf-8')  as input_file:
        data = json.load(input_file)

    # Initialize a list to hold the formatted output
    formatted_output = []

    # Regular expressions to extract content with SQL tags
    sql_pattern = re.compile(r'<SQL_(\d+)>(.*?)</SQL_\1>', re.DOTALL)

    # Process each question ID in the input JSON
    for question_id, sql_content in data.items():

        # Find all SQL statements for the current question ID
        sql_matches = sql_pattern.findall(sql_content)

        question_entry = {
            "question_id": question_id,
            "sql_1": "",
            "sql_2": "",
            "sql_3": ""
        }

        for match in sql_matches:
            sql_number, sql_query = match
            # 解码HTML实体（例如将&lt;转换为<，&gt;转换为>等）
            sql_query = html.unescape(sql_query.strip())
            key = f"sql_{sql_number}"
            if key in question_entry:
                question_entry[key] = sql_query

        formatted_output.append(question_entry)

    with open(output_files, 'w', encoding='utf-8') as f:
        json.dump(formatted_output, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # 获取命令行参数或使用默认值
    base_dir = "./results/dev/CHESS_IR_SS_CG/small_test"
    
    if len(sys.argv) > 1:
        # 如果提供了路径参数，使用它
        result_path = sys.argv[1]
    else:
        # 否则，找到最新的结果目录
        result_dirs = glob.glob(f"{base_dir}/*")
        if not result_dirs:
            print(f"No result directories found in {base_dir}")
            sys.exit(1)
        
        # 按修改时间排序，选择最新的
        latest_dir = max(result_dirs, key=os.path.getmtime)
        result_path = latest_dir
    
    input_file = os.path.join(result_path, "-predictions.json")
    output_file = os.path.join(result_path, "-output.json")
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)
    
    extract_sql_queries(input_file, output_file)    
    print(f"Extracted SQL queries have been saved to {output_file}")    