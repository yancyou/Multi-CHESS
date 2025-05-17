import argparse
import yaml
import json
import os
from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path

from runner.run_manager import RunManager

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the pipeline with the specified configuration.")
    parser.add_argument('--data_mode', type=str, default="dev", required=False, help="Mode of the data to be processed.")
    parser.add_argument('--data_path', type=str, default="./data/spider_data/small_test.json", required=False, help="Path to the data file.")
    parser.add_argument('--config', type=str, default="./run/configs/CHESS_IR_SS_CG.yaml", required=False, help="Path to the configuration file.")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers to use.")
    parser.add_argument('--log_level', type=str, default='info', help="Logging level.")
    parser.add_argument('--pick_final_sql', type=bool, default=True, help="Pick the final SQL from the generated SQLs.")
    args = parser.parse_args()

    args.run_start_time = datetime.now().isoformat()
    with open(args.config, 'r') as file:
        args.config=yaml.safe_load(file)
    
    return args

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

def main():
    """
    Main function to run the pipeline with the specified configuration.
    """
    
    args = parse_arguments()
    dataset = load_dataset(args.data_path)

    run_manager = RunManager(args)
    run_manager.initialize_tasks(dataset)
    run_manager.run_tasks()
    run_manager.generate_sql_files()

if __name__ == '__main__':
    main()
