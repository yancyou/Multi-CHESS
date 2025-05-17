import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Tuple

@dataclass
# Defines a Statistics class for storing and managing statistical data.
# This class works with StatisticsManager to track task correctness, error rates, and total task counts
class Statistics:
    corrects: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    incorrects: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    errors: Dict[str, List[Union[Tuple[str, str], Tuple[str, str, str]]]] = field(default_factory=dict)
    total: Dict[str, int] = field(default_factory=dict)

    # Converts the statistical data in the class to dictionary format for serialization to JSON or other formats.
    def to_dict(self) -> Dict[str, Dict[str, Union[Dict[str, int], List[Tuple[str, str]]]]]:
        """
        Converts the statistics data to a dictionary format.

        Returns:
            Dict[str, Dict[str, Union[Dict[str, int], List[Tuple[str, str]]]]]: The statistics data as a dictionary.
        """
        return {
            "counts": {
                # Count task numbers
                key: {
                    "correct": len(self.corrects.get(key, [])),
                    "incorrect": len(self.incorrects.get(key, [])),
                    "error": len(self.errors.get(key, [])),
                    "total": self.total.get(key, 0)
                }
                for key in self.total
            },
            "ids": {
                # Return task IDs
                key: {
                    "correct": sorted(self.corrects.get(key, [])),
                    "incorrect": sorted(self.incorrects.get(key, [])),
                    "error": sorted(self.errors.get(key, []))
                }
                for key in self.total
            }
        }
# Finally returns dictionary format{
#     "counts": {
#         "task_type_1": {
#             "correct": 10,
#             "incorrect": 5,
#             "error": 2,
#             "total": 17
#         }
#     },
#     "ids": {
#         "task_type_1": {
#             "correct": [("db1", "q1"), ("db2", "q2")],
#             "incorrect": [("db1", "q3")],
#             "error": [("db1", "q4", "exec_err")]
#         }
#     }
# }

class StatisticsManager:
    def __init__(self, result_directory: str):
        """
        Initializes the StatisticsManager.

        Args:
            result_directory (str): The directory to store results.
        """
        self.result_directory = Path(result_directory)
        self.statistics = Statistics()

        # Ensure the statistics file exists
        self.statistics_file_path = self.result_directory / "-statistics.json"
        if not self.statistics_file_path.exists():
            # If the file doesn't exist, create an empty file and write initial statistics data.
            self.statistics_file_path.touch()
            self.dump_statistics_to_file()

    def update_stats(self, db_id: str, question_id: str, validation_for: str, result: Dict[str, Any]):
        """
        Updates the statistics based on the validation result.

        Args:
            db_id (str): The database ID.
            question_id (str): The question ID.
            validation_for (str): The validation context.
            result (Dict[str, Any]): The validation result.
        """
        # Execution result
        exec_res = result["exec_res"]
        # Execution error message
        exec_err = result["exec_err"]
        # Update total task count under current validation context (validation_for)
        self.statistics.total[validation_for] = self.statistics.total.get(validation_for, 0) + 1
        # Add correct tasks
        if exec_res == 1:
            if validation_for not in self.statistics.corrects:
                self.statistics.corrects[validation_for] = []
            self.statistics.corrects[validation_for].append((db_id, question_id))
        else:
            # Handle failed tasks
            if exec_err == "incorrect answer":
                if validation_for not in self.statistics.incorrects:
                    self.statistics.incorrects[validation_for] = []
                self.statistics.incorrects[validation_for].append((db_id, question_id))
            else:
                if validation_for not in self.statistics.errors:
                    self.statistics.errors[validation_for] = []
                self.statistics.errors[validation_for].append((db_id, question_id, exec_err))
    # Write statistics data to file
    def dump_statistics_to_file(self):
        """
        Dumps the current statistics to a JSON file.
        """
        with self.statistics_file_path.open('w') as f:
            json.dump(self.statistics.to_dict(), f, indent=4)
