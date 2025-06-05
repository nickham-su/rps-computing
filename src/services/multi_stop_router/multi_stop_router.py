import numpy as np
from typing import Tuple, List, Optional
import Pyro5.server

from src.services.multi_stop_router.multi_stop_router_executor import MultiStopRouterProcessExecutor
from src.services.multi_stop_router.multi_stop_worker import MultiStopRouterWorker


@Pyro5.server.expose
class MultiStopRouter:
    # 添加类属性保存共享的DirectRouterProxy实例

    @staticmethod
    def batch_calc_route_duration(
            params: List[Tuple[np.ndarray, Optional[Tuple[float, float]]]]
    ) -> List[float]:
        """ 计算路径用时 """
        process_executor = MultiStopRouterProcessExecutor()
        futures = [
            process_executor.submit(calc_route_duration_global, waypoints, start_coord)
            for waypoints, start_coord in params
        ]
        results = [future.result()[0] for future in futures]
        return results

    @staticmethod
    def batch_calc_route_duration_with_indexes(
            params: List[Tuple[np.ndarray, Optional[Tuple[float, float]]]]
    ) -> List[Tuple[float, List[int]]]:
        """ 计算路径用时并返回索引 """
        process_executor = MultiStopRouterProcessExecutor()
        futures = [
            process_executor.submit(calc_route_duration_global, waypoints, start_coord)
            for waypoints, start_coord in params
        ]
        results = [future.result() for future in futures]
        return results

    @staticmethod
    def connected() -> bool:
        """ 检查RPC服务器是否连接成功 """
        return True


def calc_route_duration_global(waypoints: np.ndarray, start_coord: Optional[Tuple[float, float]] = None) \
        -> Tuple[float, List[int]]:
    return MultiStopRouterWorker.calc_route_duration(waypoints, start_coord)
