# Adapted from
# https://github.com/taoyds/spider/blob/master/evaluation.py
# 
# ################################
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

from __future__ import print_function
import os
import json
import sqlite3
import argparse
from tqdm import tqdm
import multiprocessing

from process_sql import tokenize, get_schema, get_tables_with_alias, Schema, get_sql, \
    parse_from as parse_complex_from, parse_select

# Flag to disable value evaluation
DISABLE_VALUE = True
# Flag to disable distinct in select evaluation
DISABLE_DISTINCT = True

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}

levels = ['easy', 'medium', 'hard', 'extra', 'all']


def get_sqls(schema, json_data, select_only=False):
    json_data['sql_1_parsed'] = None
    json_data['sql_2_parsed'] = None
    json_data['sql_3_parsed'] = None
    json_data['sql_combined_parsed'] = None
    json_data['sql_num'] = 0
    json_data['sql_1'] = json_data['sql_1'] if 'sql_1' in json_data else ''
    json_data['sql_2'] = json_data['sql_2'] if 'sql_2' in json_data else ''
    json_data['sql_3'] = json_data['sql_3'] if 'sql_3' in json_data else ''
    json_data['sql_combined'] = json_data['sql_combined'] if 'sql_combined' in json_data else ''

    if json_data['sql_1']:
        try:
            if select_only:
                json_data['sql_1_parsed'], success = get_select_only(schema, json_data['sql_1'])
            else:
                json_data['sql_1_parsed'] = get_sql(schema, json_data['sql_1'])
        except Exception as e:
            json_data['sql_1_parsed'], success = get_select_only(schema, json_data['sql_1'])
        json_data['sql_num'] += 1
    if json_data['sql_2']:
        try:
            if select_only:
                json_data['sql_2_parsed'], success = get_select_only(schema, json_data['sql_2'])
            else:
                json_data['sql_2_parsed'] = get_sql(schema, json_data['sql_2'])
        except Exception as e:
            json_data['sql_2_parsed'], success = get_select_only(schema, json_data['sql_2'])
        json_data['sql_num'] += 1
    if json_data['sql_3']:
        try:
            if select_only:
                json_data['sql_3_parsed'], success = get_select_only(schema, json_data['sql_3'])
            else:
                json_data['sql_3_parsed'] = get_sql(schema, json_data['sql_3'])
        except Exception as e:
            json_data['sql_3_parsed'], success = get_select_only(schema, json_data['sql_3'])
        json_data['sql_num'] += 1
    if json_data['sql_combined']:
        try:
            if select_only:
                json_data['sql_combined_parsed'], success = get_select_only(schema, json_data['sql_combined'])
            else:
                json_data['sql_combined_parsed'] = get_sql(schema, json_data['sql_combined'])
        except Exception as e:
            json_data['sql_combined_parsed'], success = get_select_only(schema, json_data['sql_combined'])
    return json_data


def rebuild_sqls(schema, json_data, kmap):
    if json_data['sql_1_parsed']:
        g_valid_col_units = build_valid_col_units(json_data['sql_1_parsed']['from']['table_units'], schema)
        json_data['sql_1_parsed'] = rebuild_sql_val(json_data['sql_1_parsed'])
        json_data['sql_1_parsed'] = rebuild_sql_col(g_valid_col_units, json_data['sql_1_parsed'], kmap)
    if json_data['sql_2_parsed']:
        g_valid_col_units = build_valid_col_units(json_data['sql_2_parsed']['from']['table_units'], schema)
        json_data['sql_2_parsed'] = rebuild_sql_val(json_data['sql_2_parsed'])
        json_data['sql_2_parsed'] = rebuild_sql_col(g_valid_col_units, json_data['sql_2_parsed'], kmap)
    if json_data['sql_3_parsed']:
        g_valid_col_units = build_valid_col_units(json_data['sql_3_parsed']['from']['table_units'], schema)
        json_data['sql_3_parsed'] = rebuild_sql_val(json_data['sql_3_parsed'])
        json_data['sql_3_parsed'] = rebuild_sql_col(g_valid_col_units, json_data['sql_3_parsed'], kmap)
    if json_data['sql_combined_parsed']:
        g_valid_col_units = build_valid_col_units(json_data['sql_combined_parsed']['from']['table_units'], schema)
        json_data['sql_combined_parsed'] = rebuild_sql_val(json_data['sql_combined_parsed'])
        json_data['sql_combined_parsed'] = rebuild_sql_col(g_valid_col_units, json_data['sql_combined_parsed'], kmap)
    return json_data


def eval_hardness(evaluator, json_data):
    max_hardness = -1
    if json_data['sql_1_parsed']:
        sql_1_hardness = evaluator.eval_hardness(json_data['sql_1_parsed'])
        if levels.index(sql_1_hardness) > max_hardness:
            max_hardness = levels.index(sql_1_hardness)
    if json_data['sql_2_parsed']:
        sql_2_hardness = evaluator.eval_hardness(json_data['sql_2_parsed'])
        if levels.index(sql_2_hardness) > max_hardness:
            max_hardness = levels.index(sql_2_hardness)
    if json_data['sql_3_parsed']:
        sql_3_hardness = evaluator.eval_hardness(json_data['sql_3_parsed'])
        if levels.index(sql_3_hardness) > max_hardness:
            max_hardness = levels.index(sql_3_hardness)
    return levels[max_hardness]


def eval_exec_matchs(db, pred, gold, safe_mode=False):
    # if pred['sql_num'] == 1:
    #     return 0.0
    if pred['sql_num'] == 1 and gold['sql_combined_parsed']:
        if pred['sql_1']:
            sql_p = 'sql_1'
        elif pred['sql_2']:
            sql_p = 'sql_2'
        elif pred['sql_3']:
            sql_p = 'sql_3'
        if not pred[sql_p + '_parsed']:
            print()
        if not gold['sql_combined_parsed']:
            print()
        exec_score = eval_exec_match(db,
            pred[sql_p], gold['sql_combined'], pred[sql_p + '_parsed'], gold['sql_combined_parsed'], safe_mode)
        if exec_score:
            return 1.0
    max_sql_num = max(pred['sql_num'], gold['sql_num'])
    max_sql_num = max(pred['sql_num'], gold['sql_num'])

    # 把 pred 里的 sql_1, sql_2, sql_3(如果有) 组装成列表
    pred_sqls = []
    for i in range(1, 4):
        sql_key = f'sql_{i}'
        sql_parsed_key = f'{sql_key}_parsed'
        if pred.get(sql_key):  # 非空才加入
            pred_sqls.append((pred[sql_key], pred[sql_parsed_key]))

    # 把 gold 里的 sql_1, sql_2, sql_3(如果有) 组装成列表
    gold_sqls = []
    for i in range(1, 4):
        sql_key = f'sql_{i}'
        sql_parsed_key = f'{sql_key}_parsed'
        if gold.get(sql_key):  # 非空才加入
            gold_sqls.append((gold[sql_key], gold[sql_parsed_key]))

    matched_count = 0

    # 依次从 pred_sqls 中取出每一个 pred_sql，尝试与 gold_sqls 匹配
    for (pred_sql, pred_sql_parsed) in pred_sqls:
        if not pred_sql_parsed:
            print()
        for j, (gold_sql, gold_sql_parsed) in enumerate(gold_sqls):
            if not gold_sql_parsed:
                print()
            # 如果匹配成功
            if eval_exec_match(db, pred_sql, gold_sql, pred_sql_parsed, gold_sql_parsed, safe_mode):
                matched_count += 1
                # 从 gold_sqls 中删除该条，防止重复匹配
                gold_sqls.pop(j)
                break  # 跳出 gold_sqls 的循环，继续下一个 pred_sql

    # 若 max_sql_num 为 0，防止除以 0
    if max_sql_num == 0:
        return 0.0

    return matched_count / max_sql_num


def get_select_only(schema, sql):
    try:
        toks = tokenize(sql)
        tables_with_alias = get_tables_with_alias(schema.schema, toks)
        len_ = len(toks)
        start_idx = 0
        idx = start_idx
        from_end_idx, table_units, conds, default_tables, sqls = parse_complex_from(toks, start_idx, tables_with_alias, schema)
        _, select_col_units = parse_select(toks, idx, tables_with_alias, schema, default_tables)
        idx = from_end_idx
        for s in sqls:
            if s['select'][1] and s['select'][1][0] != select_col_units[1][-1]:
                select_col_units[1].extend(s['select'][1])
        select = select_col_units
        success = True
    except Exception as e:
        select = [False, []]
        success = False
    return {
        "except": None,
        "from": {
            "conds": [],
            "table_units": []
        },
        "groupBy": [],
        "having": [],
        "intersect": None,
        "limit": None,
        "orderBy": [],
        "select": select,
        "union": None,
        "where": []
    }, success


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                               [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count


class Evaluator:
    """A simple evaluator"""

    def __init__(self):
        self.partial_scores = None

    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"


def print_scores(scores):
    print("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *levels))
    counts = [scores[level]['count'] for level in levels]
    print("{:20} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d}".format("count", *counts))

    print('=====================   EXECUTION ACCURACY     =====================')
    this_scores = [scores[level]['acc'] for level in levels]
    print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("accuracy", *this_scores))

    print('=====================   EXECUTION SCORE     =====================')
    this_scores = [scores[level]['exec'] for level in levels]
    print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("score", *this_scores))


def evaluate(gold, predict, db_dir, kmaps, safe_mode=False):
    with open(gold) as f:
        glist = json.load(f)

    with open(predict) as f:
        plist = json.load(f)
    # plist = [("select max(Share),min(Share) from performance where Type != 'terminal'", "orchestra")]
    # glist = [("SELECT max(SHARE) ,  min(SHARE) FROM performance WHERE TYPE != 'Live final'", "orchestra")]
    evaluator = Evaluator()

    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    scores = {}
    detail_scores = []

    for level in levels:
        scores[level] = {'count': 0, 'exact': 0.}
        scores[level]['exec'] = 0
        scores[level]['acc'] = 0

    # for (i,p), g in zip(enumerate(plist), glist):
    for (i, p), g in tqdm(zip(enumerate(plist), glist), total=len(plist), desc="Processing Evaluation"):
        # print(i)
        # if i != 864:
        #     continue
        db_name = g['db_id']
        db = os.path.join(db_dir, db_name, db_name + ".sqlite")
        schema = Schema(get_schema(db))
        g = get_sqls(schema, g)
        hardness = eval_hardness(evaluator, g)
        scores[hardness]['count'] += 1
        scores['all']['count'] += 1

        p = get_sqls(schema, p, select_only=True)
        g = get_sqls(schema, g, select_only=True)
        # rebuild sql for value evaluation
        kmap = kmaps[db_name]
        p = rebuild_sqls(schema, p, kmap)
        g = rebuild_sqls(schema, g, kmap)

        exec_score = eval_exec_matchs(db, p, g, safe_mode)
        p = get_sqls(schema, p, select_only=True)
        scores[hardness]['exec'] += exec_score
        scores['all']['exec'] += exec_score
        scores[hardness]['acc'] += 1 if exec_score == 1 else 0
        scores['all']['acc'] += 1 if exec_score == 1 else 0
        detail_scores.append(exec_score)
    for level in levels:
        if scores[level]['count'] == 0:
            continue
        scores[level]['exec'] /= scores[level]['count']
        scores[level]['acc'] /= scores[level]['count']
    print_scores(scores)
    return detail_scores


import time


# ------------- SQLWorker 和 SQLProcessPool 定义 -------------
class SQLWorker(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.daemon = True  # 让进程随主进程退出

    def run(self):
        """ 每个 SQLWorker 在自己的进程里运行 SQLite 连接 """
        while True:
            try:
                # 从任务队列中取任务
                task = self.task_queue.get(timeout=5)
                if task is None:
                    break  # 收到结束信号退出
                query, db, task_id = task  # 任务包含 SQL 查询、数据库路径和任务 ID

                # 打开对应的数据库连接
                conn = sqlite3.connect(db)
                cursor = conn.cursor()
                try:
                    cursor.execute(query)
                    result = cursor.fetchall()
                    self.result_queue.put((task_id, result))
                except Exception:
                    self.result_queue.put((task_id, "SQL Error!"))
                finally:
                    conn.close()
            except Exception:
                # 队列超时未获取任务，继续循环
                pass


class SQLProcessPool:
    def __init__(self, num_workers=8):
        self.num_workers = num_workers
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.workers = []
        self._start_workers()

    def _start_workers(self):
        """ 初始化并启动多个 SQLWorker 进程 """
        for _ in range(self.num_workers):
            worker = SQLWorker(self.task_queue, self.result_queue)
            worker.start()
            self.workers.append(worker)

    def execute_sql_with_timeout(self, sql, db, timeout=5):
        """ 在进程池中执行 SQL 查询，保证单次查询不会超过 timeout 秒 """
        task_id = time.time()  # 使用当前时间作为任务标识
        self.task_queue.put((sql, db, task_id))

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result_task_id, result = self.result_queue.get(timeout=0.5)
                if result_task_id == task_id:
                    return result  # 找到对应任务的结果则返回
            except Exception:
                pass  # 超时等待结果，继续轮询

        # 超时：打印提示并重启可能被卡住的 worker
        print(f"Query on {db} timed out! Restarting worker...")
        self._restart_worker()
        return None

    def _restart_worker(self):
        """ 检查每个进程状态，如果进程死掉则重启 """
        for i, worker in enumerate(self.workers):
            if worker.is_alive():
                worker.terminate()  # 强制终止
                print(f"Restarting worker {i}")
                new_worker = SQLWorker(self.task_queue, self.result_queue)
                new_worker.start()
                self.workers[i] = new_worker

    def close(self):
        """ 结束所有工作进程 """
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join()


# **封装进程池 API**
def execute_sql_with_timeout2(sql, db, timeout=5):
    """ 兼容原来的调用方式 """
    return sql_pool.execute_sql_with_timeout(sql, db, timeout)


def run_sql(query, db, result_queue):
    """ 在子进程中运行SQL查询 """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        result_queue.put(cursor.fetchall())  # 存储查询结果
    except Exception as e:
        result_queue.put(f"SQL Error!")
    finally:
        conn.close()


def execute_sql_with_timeout(sql, db, timeout=5):
    result_queue = multiprocessing.Queue()
    sql_process = multiprocessing.Process(target=run_sql, args=(sql, db, result_queue))

    sql_process.start()
    sql_process.join(timeout)  # 等待 `timeout` 秒

    if sql_process.is_alive():
        sql_process.terminate()  # 强制终止进程
        print("Query timed out!")
        return None

    return result_queue.get() if not result_queue.empty() else None


def compare_sql_results_unordered(result1, result2):
    """
    判断两个 SQL 返回的结果是否一致，但允许列的顺序不同。
    先按行判断行数一致，再按列判断（允许列顺序不同）。

    参数：
      result1: 第一个查询返回结果，为列表，元素为元组（每个元组代表一行）
      result2: 第二个查询返回结果，为列表，元素为元组

    返回：
      True 如果两个结果列数据一致（忽略列的顺序）；否则返回 False
    """
    # 检查行数
    if len(result1) != len(result2):
        return False

    # 若结果为空，视为一致
    if not result1 and not result2:
        return True
    elif not result1 or not result2:
        return False

    # 检查每一行的列数是否一致
    num_columns = len(result1[0])
    for row in result1 + result2:
        if len(row) != num_columns:
            return False

    # 转置得到按列的数据
    columns1 = list(zip(*result1))
    columns2 = list(zip(*result2))

    # 如果列数不一致，则不匹配
    if len(columns1) != len(columns2):
        return False

    # 标记 columns2 中哪些列已经被匹配
    matched = [False] * len(columns2)
    for idx1, col1 in enumerate(columns1):
        found_match = False
        for idx2, col2 in enumerate(columns2):
            if not matched[idx2] and col1 == col2:
                matched[idx2] = True
                found_match = True
                break
        if not found_match:
            return False

    return True


def eval_exec_match(db, p_str, g_str, pred, gold, safe_mode=False):
    """
    return 1 if the values between prediction and gold are matching
    in the corresponding index. Currently not support multiple col_unit(pairs).
    """
    # if 'with recursive' in p_str.lower():
    #     return False

    if not safe_mode:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
        except:
            conn.close()
            return False
    else:
        p_res = execute_sql_with_timeout2(p_str, db, timeout=15)
        if p_res is None or p_res == "SQL Error!":
            return False
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

    cursor.execute(g_str)
    q_res = cursor.fetchall()
    conn.close()

    def res_map(res, val_units):
        rmap = {}
        for idx, val_unit in enumerate(val_units):
            # key = tuple(val_unit[1]) if not val_unit[2] else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
            key = val_unit
            rmap[key] = [r[idx] for r in res]
        return rmap

    p_val_units = [unit for unit in pred['select'][1]]
    q_val_units = [unit for unit in gold['select'][1]]
    # if p_res and len(p_res[0]) == len(q_val_units) and len(p_res[0]) != len(p_val_units):
    #     p_val_units = q_val_units
    # if (q_res and len(q_res[0]) != len(q_val_units)) or (p_res and len(p_res[0]) != len(p_val_units)):
    #     return compare_sql_results_unordered(p_res, q_res)
    if set(p_val_units) == set(q_val_units):
        # if not res_map(p_res, p_val_units) == res_map(q_res, q_val_units) and  compare_sql_results_unordered(p_res, q_res):
        #     print(p_res, q_res)
        return res_map(p_res, p_val_units) == res_map(q_res, q_val_units)
    else:
        return compare_sql_results_unordered(p_res, q_res)
    # return compare_sql_results_unordered(p_res, q_res)


# Rebuild SQL functions for value evaluation
def rebuild_cond_unit_val(cond_unit):
    if cond_unit is None or not DISABLE_VALUE:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    if type(val1) is not dict:
        val1 = None
    else:
        val1 = rebuild_sql_val(val1)
    if type(val2) is not dict:
        val2 = None
    else:
        val2 = rebuild_sql_val(val2)
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_val(condition):
    if condition is None or not DISABLE_VALUE:
        return condition

    res = []
    for idx, it in enumerate(condition):
        if idx % 2 == 0:
            res.append(rebuild_cond_unit_val(it))
        else:
            res.append(it)
    return res


def rebuild_sql_val(sql):
    if sql is None or not DISABLE_VALUE:
        return sql

    sql['from']['conds'] = rebuild_condition_val(sql['from']['conds'])
    sql['having'] = rebuild_condition_val(sql['having'])
    sql['where'] = rebuild_condition_val(sql['where'])
    sql['intersect'] = rebuild_sql_val(sql['intersect'])
    sql['except'] = rebuild_sql_val(sql['except'])
    sql['union'] = rebuild_sql_val(sql['union'])

    return sql


# Rebuild SQL functions for foreign key evaluation
def build_valid_col_units(table_units, schema):
    col_ids = [table_unit[1] for table_unit in table_units if table_unit[0] == TABLE_TYPE['table_unit']]
    prefixs = [col_id[:-2] for col_id in col_ids]
    valid_col_units = []
    for value in schema.idMap.values():
        if '.' in value and value[:value.index('.')] in prefixs:
            valid_col_units.append(value)
    return valid_col_units


def rebuild_col_unit_col(valid_col_units, col_unit, kmap):
    if col_unit is None:
        return col_unit

    agg_id, col_id, distinct = col_unit
    if col_id in kmap and col_id in valid_col_units:
        col_id = kmap[col_id]
    if DISABLE_DISTINCT:
        distinct = None
    return agg_id, col_id, distinct


def rebuild_val_unit_col(valid_col_units, val_unit, kmap):
    if val_unit is None:
        return val_unit

    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, kmap)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, kmap)
    return unit_op, col_unit1, col_unit2


def rebuild_table_unit_col(valid_col_units, table_unit, kmap):
    if table_unit is None:
        return table_unit

    table_type, col_unit_or_sql = table_unit
    if isinstance(col_unit_or_sql, tuple):
        col_unit_or_sql = rebuild_col_unit_col(valid_col_units, col_unit_or_sql, kmap)
    return table_type, col_unit_or_sql


def rebuild_cond_unit_col(valid_col_units, cond_unit, kmap):
    if cond_unit is None:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    val_unit = rebuild_val_unit_col(valid_col_units, val_unit, kmap)
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_col(valid_col_units, condition, kmap):
    for idx in range(len(condition)):
        if idx % 2 == 0:
            condition[idx] = rebuild_cond_unit_col(valid_col_units, condition[idx], kmap)
    return condition


def rebuild_select_col(valid_col_units, sel, kmap):
    if sel is None:
        return sel
    distinct, _list = sel
    new_list = []
    for it in _list:
        agg_id, val_unit = it
        new_list.append((agg_id, rebuild_val_unit_col(valid_col_units, val_unit, kmap)))
    if DISABLE_DISTINCT:
        distinct = None
    return distinct, new_list


def rebuild_from_col(valid_col_units, from_, kmap):
    if from_ is None:
        return from_

    from_['table_units'] = [rebuild_table_unit_col(valid_col_units, table_unit, kmap) for table_unit in
                            from_['table_units']]
    from_['conds'] = rebuild_condition_col(valid_col_units, from_['conds'], kmap)
    return from_


def rebuild_group_by_col(valid_col_units, group_by, kmap):
    if group_by is None:
        return group_by

    return [rebuild_col_unit_col(valid_col_units, col_unit, kmap) for col_unit in group_by]


def rebuild_order_by_col(valid_col_units, order_by, kmap):
    if order_by is None or len(order_by) == 0:
        return order_by

    direction, val_units = order_by
    new_val_units = [rebuild_val_unit_col(valid_col_units, val_unit, kmap) for val_unit in val_units]
    return direction, new_val_units


def rebuild_sql_col(valid_col_units, sql, kmap):
    if sql is None:
        return sql

    sql['select'] = rebuild_select_col(valid_col_units, sql['select'], kmap)
    sql['from'] = rebuild_from_col(valid_col_units, sql['from'], kmap)
    sql['where'] = rebuild_condition_col(valid_col_units, sql['where'], kmap)
    sql['groupBy'] = rebuild_group_by_col(valid_col_units, sql['groupBy'], kmap)
    sql['orderBy'] = rebuild_order_by_col(valid_col_units, sql['orderBy'], kmap)
    sql['having'] = rebuild_condition_col(valid_col_units, sql['having'], kmap)
    sql['intersect'] = rebuild_sql_col(valid_col_units, sql['intersect'], kmap)
    sql['except'] = rebuild_sql_col(valid_col_units, sql['except'], kmap)
    sql['union'] = rebuild_sql_col(valid_col_units, sql['union'], kmap)

    return sql


def build_foreign_key_map(entry):
    cols_orig = entry["column_names_original"]
    tables_orig = entry["table_names_original"]

    # rebuild cols corresponding to idmap in Schema
    cols = []
    for col_orig in cols_orig:
        if col_orig[0] >= 0:
            t = tables_orig[col_orig[0]]
            c = col_orig[1]
            cols.append("__" + t.lower() + "." + c.lower() + "__")
        else:
            cols.append("__all__")

    def keyset_in_list(k1, k2, k_list):
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set

    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]

    return foreign_key_map


def build_foreign_key_map_from_json(table):
    with open(table) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry['db_id']] = build_foreign_key_map(entry)
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', dest='gold', default="final_data/dataset/spider/small_test.json", type=str)
    parser.add_argument('--pred', dest='pred', default="results/o3-mini/spider_small.json", type=str)
    parser.add_argument('--db', dest='db', default="spider_data/test_database", type=str)
    parser.add_argument('--table', dest='table', default="spider_data/test_tables.json", type=str)
    parser.add_argument('--output_detail_scores_to_pred_file', action='store_true', help="Enable detailed score output")
    parser.add_argument('--safe_mode', action='store_true', help="Enable safe mode to bypass long-running SQL queries")
    sql_pool = SQLProcessPool(num_workers=1)
    args = parser.parse_args()

    args.safe_mode = True
    gold = args.gold
    pred = args.pred
    db_dir = args.db
    table = args.table

    kmaps = build_foreign_key_map_from_json(table)

    detail_scores = evaluate(gold, pred, db_dir, kmaps, args.safe_mode)
    sql_pool.close()

    if args.output_detail_scores_to_pred_file:
        with open(pred, 'r') as f:
            data = json.load(f)
        for i, score in enumerate(detail_scores):
            data[i]['detail_score'] = score
        with open(pred, 'w') as f:
            json.dump(data, f, indent=4)

# python evaluation.py --gold /Users/yujiangan/Documents/python/Text2SQLs/data/test_gold.sql --pred /Users/yujiangan/Documents/python/Text2SQLs/data/test_gold.sql --etype exec --db /Users/yujiangan/Documents/python/Text2SQLs/spider_data/test_database --table /Users/yujiangan/Documents/python/Text2SQLs/spider_data/test_tables.json

# python evaluation.py --gold data/small_test.json --pred output/predictions.json  --db data/spider_data/test_database --table data/spider_data/test_tables.json

# python evaluation.py --gold data/spider_data/small_test.json --pred results/dev/CHESS_IR_SS_CG/small_test/2025-02-06T14-46-08.134708/-output.json  --db data/spider_data/test_database --table data/spider_data/test_tables.json

# python evaluation.py --gold data/spider_data/small_test.json --pred results/dev/CHESS_IR_SS_CG/small_test/2025-02-06T14-54-01.974010/-output.json  --db data/spider_data/test_database --table data/spider_data/test_tables.json

# python evaluation.py --gold data/spider_data/small_test1.json --pred results/dev/CHESS_IR_SS_CG/small_test1/2025-02-07T00-43-43.416445/-output.json  --db data/spider_data/test_database --table data/spider_data/test_tables.json

#(多条sql+多例子) python evaluation.py --gold data/spider_data/small_test.json --pred results/dev/CHESS_IR_SS_CG/small_test/2025-04-28T00-47-49.027406/-output.json  --db data/spider_data/test_database --table data/spider_data/test_tables.json

#（多条sql+多单例子） python evaluation.py --gold data/spider_data/small_test.json --pred results/dev/CHESS_IR_SS_CG/small_test/2025-04-28T08-02-46.311481/-output.json  --db data/spider_data/test_database --table data/spider_data/test_tables.json