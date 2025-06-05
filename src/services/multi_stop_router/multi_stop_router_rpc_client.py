import threading
from typing import List, Tuple, Optional
import Pyro5.client


class MultiStopRouterRpcClient:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    from src.services.multi_stop_router.multi_stop_router_rpc_server import SERVER_ADDRESS, SERVER_PORT, \
                        PYRO_OBJECT_ID
                    cls._instance._multi_stop_router = Pyro5.client.Proxy(
                        f"PYRO:{PYRO_OBJECT_ID}@{SERVER_ADDRESS}:{SERVER_PORT}")
                    cls._instance._multi_stop_router._pyroSerializer = 'marshal'
                    if cls._instance._multi_stop_router is None:
                        raise RuntimeError("无法获取MultiStopRouter实例，请检查RPC服务器是否已启动")
        return cls._instance

    def batch_calc_route_duration(
            self, params: List[Tuple[List[Tuple[float, float]], Optional[Tuple[float, float]]]]
    ) -> List[float]:
        """ 计算路径用时 """
        return self._multi_stop_router.batch_calc_route_duration(params)

    def batch_calc_route_duration_with_indexes(
            self, params: List[Tuple[List[Tuple[float, float]], Optional[Tuple[float, float]]]]
    ) -> List[Tuple[float, List[int]]]:
        """ 计算路径用时并返回索引 """
        return self._multi_stop_router.batch_calc_route_duration_with_indexes(params)
