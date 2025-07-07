import threading
import time
from typing import List, Tuple, Optional
import Pyro5.client
import click


class DirectRouterRpcClient:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    from src.services.direct_router.direct_router_rpc_server import SERVER_ADDRESS, SERVER_PORT, \
                        PYRO_OBJECT_ID
                    cls._instance._direct_router = Pyro5.client.Proxy(
                        f"PYRO:{PYRO_OBJECT_ID}@{SERVER_ADDRESS}:{SERVER_PORT}")
                    cls._instance._direct_router._pyroSerializer = 'marshal'
                    if cls._instance._direct_router is None:
                        raise RuntimeError("无法获取DirectRouter实例，请检查RPC服务器是否已启动")
                    cls._instance._cache = {}  # Dict[Tuple[int, int], float]
        return cls._instance

    def init(self):
        """ 初始化客户端，在fork进程时，在linux平台会复制_instance，但连接不可用，所以需要重新初始化 """
        from src.services.direct_router.direct_router_rpc_server import SERVER_ADDRESS, SERVER_PORT, PYRO_OBJECT_ID
        self._direct_router = Pyro5.client.Proxy(f"PYRO:{PYRO_OBJECT_ID}@{SERVER_ADDRESS}:{SERVER_PORT}")
        self._direct_router._pyroSerializer = 'marshal'
        if self._direct_router is None:
            raise RuntimeError("无法获取DirectRouter实例，请检查RPC服务器是否已启动")
        self._cache = {}

    def _generate_key(self, start_id: int, end_id: int) -> Tuple[int, int]:
        """生成缓存键 - 返回元组"""
        return min(start_id, end_id), max(start_id, end_id)

    def calc_path_duration(self, start_id: int, end_id: int) -> float:
        """ 计算路径用时 """
        key = self._generate_key(start_id, end_id)
        if key in self._cache:
            return self._cache[key]
        duration = self._direct_router.calc_path_duration(start_id, end_id)
        self._cache[key] = duration
        return duration

    def batch_calc_path_duration(self, point_pairs_list: List[Tuple[int, int]]) -> List[float]:
        """ 批量计算路径用时 """
        # 先检查缓存
        not_cached_pairs = []
        for start_id, end_id in point_pairs_list:
            key = self._generate_key(start_id, end_id)
            cached_duration = self._cache.get(key)
            if cached_duration is None:
                not_cached_pairs.append((start_id, end_id))

        # 如果有未缓存的点对，则计算
        if not_cached_pairs:
            results = self._direct_router.batch_calc_path_duration(not_cached_pairs)
            # 缓存计算结果
            for (start_id, end_id), duration in zip(not_cached_pairs, results):
                key = self._generate_key(start_id, end_id)
                self._cache[key] = duration

        # 从缓存中获取所有点对的用时
        return [
            self._cache.get(self._generate_key(start_id, end_id))
            for start_id, end_id in point_pairs_list
        ]

    def get_path_duration_from_cache(self, start_id: int, end_id: int) -> Optional[float]:
        """ 从缓存获取路径用时 """
        # 先查本地缓存
        key = self._generate_key(start_id, end_id)
        if key in self._cache:
            return self._cache[key]
        # 如果本地缓存没有，则查询远程
        duration = self._direct_router.get_path_duration_from_cache(start_id, end_id)
        if duration is not None:
            self._cache[key] = duration
        return duration

    def batch_get_path_duration_from_cache(self, point_pairs_list: List[Tuple[int, int]]) -> List[Optional[float]]:
        """ 批量从缓存获取路径用时 """
        # 先检查缓存
        not_cached_pairs = []
        for start_id, end_id in point_pairs_list:
            key = self._generate_key(start_id, end_id)
            cached_duration = self._cache.get(key)
            if cached_duration is None:
                not_cached_pairs.append((start_id, end_id))
        # 如果有未缓存的点对，则查询远程
        if not_cached_pairs:
            results = self._direct_router.batch_get_path_duration_from_cache(not_cached_pairs)
            # 缓存查询结果
            for (start_id, end_id), duration in zip(not_cached_pairs, results):
                key = self._generate_key(start_id, end_id)
                self._cache[key] = duration
        # 从缓存中获取所有点对的用时
        return [
            self._cache.get(self._generate_key(start_id, end_id))
            for start_id, end_id in point_pairs_list
        ]
