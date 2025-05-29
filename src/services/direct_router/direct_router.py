from concurrent.futures import ProcessPoolExecutor
import os
from typing import Tuple, List, Optional
from functools import partial
import numpy as np
import pandas as pd

from src.data.data_loader import load_map_data
from src.services.direct_router.worker import Worker

# 缓存目录
cache_dir = './cache'
# 运行时缓存文件
cache_file = f'{cache_dir}/path_duration.csv'


class DirectRouter:
    _instance = None

    def __new__(cls, warehouse_coord: Optional[Tuple[float, float]] = None,
                points: Optional[np.ndarray] = None, workers: Optional[int] = None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            if warehouse_coord is None or points is None:
                raise ValueError("必须指定warehouse_coord和points参数")

            # 读取本地缓存
            cls._instance.path_duration_cache = {}
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            for filename in os.listdir(cache_dir):  # 遍历缓存目录
                if filename.endswith('.csv'):
                    file_path = os.path.join(cache_dir, filename)
                    cache = pd.read_csv(file_path, names=['key', 'duration'], dtype={'key': str, 'duration': float})
                    for _, row in cache.iterrows():
                        cls._instance.path_duration_cache[row.key] = float(row.duration)

            # 创建进程池执行器
            max_workers = os.cpu_count()
            workers = workers if workers is not None else max_workers
            executor = ProcessPoolExecutor(max_workers=workers)
            # 并行加载地图数据
            partial_func = partial(load_map_data, warehouse_coord, points)
            list(executor.map(partial_func, range(workers)))
            cls._instance.executor = executor

        return cls._instance

    def batch_calc_path_duration(self, point_pairs_list: List[Tuple[int, int]]):
        """ 批量查询路径用时 """
        no_cache_list: List[Tuple[int, int]] = []
        for start_id, end_id in point_pairs_list:
            if start_id == end_id:
                continue
            if self.path_duration_cache.get(f"{min(start_id, end_id)}_{max(start_id, end_id)}"):
                continue
            no_cache_list.append((start_id, end_id))

        if len(no_cache_list) > 0:
            # 并行计算
            results = list(self.executor.map(calc_path_duration, no_cache_list))
            # 缓存结果
            with open(cache_file, 'a', encoding='utf-8') as file:
                for i in range(len(no_cache_list)):
                    start_id, end_id = no_cache_list[i]
                    key = f"{min(start_id, end_id)}_{max(start_id, end_id)}"
                    self.path_duration_cache[key] = results[i]
                    file.write(f"{key},{results[i]}\n")

        duration_list = []
        for start_id, end_id in point_pairs_list:
            if start_id == end_id:
                duration_list.append(0)
                continue
            key = f"{min(start_id, end_id)}_{max(start_id, end_id)}"
            duration = self.path_duration_cache.get(key)
            if duration is None:
                raise ValueError(f"无法找到路径用时缓存：{key}")
            duration_list.append(duration)

        return duration_list

    def get_path_duration_from_cache(self, start_id: int, end_id: int) -> float:
        """ 从缓存中获取路径用时 """
        if start_id == end_id:
            return 0
        key = f"{min(start_id, end_id)}_{max(start_id, end_id)}"
        return self.path_duration_cache.get(key)


def calc_path_duration(point_pairs: Tuple[int, int]) -> float:
    return Worker().calc_path_duration(point_pairs)
