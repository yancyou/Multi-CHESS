import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Tuple

@dataclass
# 定义了一个 Statistics 类，用于存储和管理统计数据。
# 这个类与 StatisticsManager 配合使用，用于追踪任务的正确性、错误率以及任务总数等
class Statistics:
    corrects: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    incorrects: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    errors: Dict[str, List[Union[Tuple[str, str], Tuple[str, str, str]]]] = field(default_factory=dict)
    total: Dict[str, int] = field(default_factory=dict)

    # 用于将类中的统计数据转换为字典格式，以便于序列化为 JSON 或其他格式。
    def to_dict(self) -> Dict[str, Dict[str, Union[Dict[str, int], List[Tuple[str, str]]]]]:
        """
        Converts the statistics data to a dictionary format.

        Returns:
            Dict[str, Dict[str, Union[Dict[str, int], List[Tuple[str, str]]]]]: The statistics data as a dictionary.
        """
        return {
            "counts": {
                # 统计任务数量
                key: {
                    "correct": len(self.corrects.get(key, [])),
                    "incorrect": len(self.incorrects.get(key, [])),
                    "error": len(self.errors.get(key, [])),
                    "total": self.total.get(key, 0)
                }
                for key in self.total
            },
            "ids": {
                # 返回任务 ID
                key: {
                    "correct": sorted(self.corrects.get(key, [])),
                    "incorrect": sorted(self.incorrects.get(key, [])),
                    "error": sorted(self.errors.get(key, []))
                }
                for key in self.total
            }
        }
# 最终返回字典格式{
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
            # 如果文件不存在，创建一个空文件并写入初始的统计数据。
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
        # 执行结果
        exec_res = result["exec_res"]
        # 执行错误信息
        exec_err = result["exec_err"]
        # 更新当前验证上下文（validation_for）下的任务总数
        self.statistics.total[validation_for] = self.statistics.total.get(validation_for, 0) + 1
        # 添加正确的任务
        if exec_res == 1:
            if validation_for not in self.statistics.corrects:
                self.statistics.corrects[validation_for] = []
            self.statistics.corrects[validation_for].append((db_id, question_id))
        else:
            # 处理错误的任务
            if exec_err == "incorrect answer":
                if validation_for not in self.statistics.incorrects:
                    self.statistics.incorrects[validation_for] = []
                self.statistics.incorrects[validation_for].append((db_id, question_id))
            else:
                if validation_for not in self.statistics.errors:
                    self.statistics.errors[validation_for] = []
                self.statistics.errors[validation_for].append((db_id, question_id, exec_err))
    # 将统计数据写入文件
    def dump_statistics_to_file(self):
        """
        Dumps the current statistics to a JSON file.
        """
        with self.statistics_file_path.open('w') as f:
            json.dump(self.statistics.to_dict(), f, indent=4)
