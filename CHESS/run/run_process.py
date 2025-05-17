#!/usr/bin/env python3

import os
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load variables from .env file
data_mode = os.getenv("DATA_MODE", "dev")
data_path = os.getenv("DATA_PATH", "./data/spider_data/dev.json")
db_root_directory = os.getenv("DB_ROOT_DIRECTORY", "./data/spider_data/database")
data_tables_path = os.getenv("DATA_TABLES_PATH", "./data/spider_data/tables.json")
index_server_host = os.getenv("INDEX_SERVER_HOST", "localhost")
index_server_port = os.getenv("INDEX_SERVER_PORT", "12345")
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")

# Signature parameters
signature_size = 100
n_gram = 3
threshold = 0.01

# Construct the command to execute the Python script
command = [
    "python3", "-u", "./src/preprocess.py",
    "--db_root_directory", db_root_directory,
    "--data_mode", data_mode,
    "--data_path", data_path,
    "--data_tables_path", data_tables_path,
    "--index_server_host", index_server_host,
    "--index_server_port", str(index_server_port),
    "--signature_size", str(signature_size),
    "--n_gram", str(n_gram),
    "--threshold", str(threshold),
    "--verbose", "true"
]

# Execute the command
try:
    result = subprocess.run(command, check=True, text=True)
    print(f"Command executed successfully: {result}")
except subprocess.CalledProcessError as e:
    print(f"Error while executing the command: {e}")

