from typing import Tuple, Optional, List
from src.services.direct_router.cache_manager import CacheManager
from src.services.direct_router.direct_router_executor import DirectRouterProcessExecutor
from src.services.direct_router.worker import Worker


class DirectRouter:

    @staticmethod
    def calc_path_duration(start_id: int, end_id: int) -> float:
        """计算单条路径用时"""
        # 先检查缓存
        cache_manager = CacheManager()
        cached_duration = cache_manager.get_from_cache(start_id, end_id)
        if cached_duration is not None:
            return cached_duration

        # 如果缓存中没有，则计算
        future = DirectRouterProcessExecutor().submit(calc_path_duration_global, (start_id, end_id))
        duration = future.result()

        # 缓存结果
        cache_manager.add_item(start_id, end_id, duration)

        return duration

    @staticmethod
    def batch_calc_path_duration(point_pairs_list: List[Tuple[int, int]]) -> List[float]:
        """批量计算路径用时"""
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
            for (start_id, end_id), duration in zip(not_cached_pairs, results):
                cache_manager.add_item(start_id, end_id, duration)

        # 从缓存中获取所有点对的用时
        return [cache_manager.get_from_cache(start_id, end_id) for start_id, end_id in point_pairs_list]

    @staticmethod
    def get_path_duration_from_cache(start_id: int, end_id: int) -> Optional[float]:
        """从缓存中获取路径用时"""
        return CacheManager().get_from_cache(start_id, end_id)

    @staticmethod
    def batch_get_path_duration_from_cache(point_pairs_list: List[Tuple[int, int]]) -> List[Optional[float]]:
        """批量从缓存中获取路径用时"""
        cache_manager = CacheManager()
        return [cache_manager.get_from_cache(start_id, end_id) for start_id, end_id in point_pairs_list]

    @staticmethod
    def shutdown():
        """关闭进程池"""
        CacheManager().shutdown()


def calc_path_duration_global(point_pairs: Tuple[int, int]) -> float:
    return Worker().calc_path_duration(point_pairs)
