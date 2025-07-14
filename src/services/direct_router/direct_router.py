from typing import Tuple, Optional, List
import threading
import Pyro5.server

from src.services.direct_router.cache_manager import CacheManager
from src.services.direct_router.direct_router_executor import DirectRouterProcessExecutor
from src.services.direct_router.direct_router_worker import DirectRouterWorker


@Pyro5.server.expose
class DirectRouter:
    _calc_lock = threading.Lock()  # 类级别的锁，用于串行化计算方法

    @staticmethod
    def calc_path_duration(start_id: int, end_id: int) -> Tuple[float, int, int]:
        """计算单条路径用时"""
        with DirectRouter._calc_lock:
            # 先检查缓存
            cache_manager = CacheManager()
            cached_duration = cache_manager.get_from_cache(start_id, end_id)
            if cached_duration is not None:
                return cached_duration, start_id, end_id

            # 如果缓存中没有，则计算
            future = DirectRouterProcessExecutor().submit(calc_path_duration_global, (start_id, end_id))
            duration, s_id, e_id = future.result()

            # 缓存结果
            cache_manager.add_item(s_id, e_id, duration)

            return duration, s_id, e_id

    @staticmethod
    def batch_calc_path_duration(point_pairs_list: List[Tuple[int, int]]) -> List[Tuple[float, int, int]]:
        """批量计算路径用时"""
        with DirectRouter._calc_lock:
            # 先检查缓存
            cache_manager = CacheManager()
            not_cached_pairs = []
            for start_id, end_id in point_pairs_list:
                cached_duration = cache_manager.get_from_cache(start_id, end_id)
                if cached_duration is None:
                    not_cached_pairs.append((start_id, end_id))

            # 如果有未缓存的点对，则计算
            if not_cached_pairs:
                process_executor = DirectRouterProcessExecutor()
                futures = [process_executor.submit(calc_path_duration_global, pair) for pair in not_cached_pairs]
                results = [future.result() for future in futures]
                # 缓存计算结果
                for duration, start_id, end_id in results:
                    cache_manager.add_item(start_id, end_id, duration)

            # 从缓存中获取所有点对的用时
            return [
                (cache_manager.get_from_cache(start_id, end_id), start_id, end_id)
                for start_id, end_id in point_pairs_list
            ]

    @staticmethod
    def get_path_duration_from_cache(start_id: int, end_id: int) -> Tuple[Optional[float], int, int]:
        """从缓存中获取路径用时"""
        return CacheManager().get_from_cache(start_id, end_id), start_id, end_id

    @staticmethod
    def batch_get_path_duration_from_cache(point_pairs_list: List[Tuple[int, int]]) -> List[
        Tuple[Optional[float], int, int]]:
        """批量从缓存中获取路径用时"""
        cache_manager = CacheManager()
        return [
            (cache_manager.get_from_cache(start_id, end_id), start_id, end_id)
            for start_id, end_id in point_pairs_list
        ]

    @staticmethod
    def connected() -> bool:
        """ 检查RPC服务器是否连接成功 """
        return True


def calc_path_duration_global(point_pairs: Tuple[int, int]) -> Tuple[float, int, int]:
    return DirectRouterWorker().calc_path_duration(point_pairs)
