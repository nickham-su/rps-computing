import threading
from typing import Optional, List, Tuple
import numpy as np
from src.services.rpc_manager.rpc_manager import RPCManager, RPC_SERVER_ADDRESS, RPC_SERVER_AUTHKEY


class RPCClient:
    """ DirectRouter RPC客户端 """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    manager = RPCManager(address=RPC_SERVER_ADDRESS, authkey=RPC_SERVER_AUTHKEY)
                    manager.connect()
                    cls._instance._manager = manager
                    cls._instance._direct_router = manager.get_direct_router()
                    cls._instance._multi_stop_router = manager.get_multi_stop_router()
                    if cls._instance._direct_router is None or cls._instance._multi_stop_router is None:
                        raise RuntimeError("无法获取DirectRouter或MultiStopRouter实例，请检查RPC服务器是否已启动")
        return cls._instance

    def calc_path_duration(self, start_id: int, end_id: int) -> float:
        """ 计算路径用时 """
        if not self._manager:
            raise RuntimeError("未连接到RPC服务器，请先调用connect()")
        return self._direct_router.calc_path_duration(start_id, end_id)

    def batch_calc_path_duration(self, point_pairs_list: List[Tuple[int, int]]) -> List[float]:
        """ 批量计算路径用时 """
        if not self._manager:
            raise RuntimeError("未连接到RPC服务器，请先调用connect()")
        return self._direct_router.batch_calc_path_duration(point_pairs_list)

    def get_path_duration_from_cache(self, start_id: int, end_id: int) -> Optional[float]:
        """ 从缓存获取路径用时 """
        if not self._manager:
            raise RuntimeError("未连接到RPC服务器，请先调用connect()")
        return self._direct_router.get_path_duration_from_cache(start_id, end_id)

    def batch_get_path_duration_from_cache(self, point_pairs_list: List[Tuple[int, int]]) -> List[Optional[float]]:
        """ 批量从缓存获取路径用时 """
        if not self._manager:
            raise RuntimeError("未连接到RPC服务器，请先调用connect()")
        return self._direct_router.batch_get_path_duration_from_cache(point_pairs_list)

    def batch_calc_route_duration(
            self, params: List[Tuple[np.ndarray, Optional[Tuple[float, float]]]]
    ) -> List[Tuple[float, List[int]]]:
        """ 计算路径用时 """
        if not self._manager:
            raise RuntimeError("未连接到RPC服务器，请先调用connect()")
        return self._multi_stop_router.batch_calc_route_duration(params)

    def batch_calc_route_duration_with_indexes(
            self, params: List[Tuple[np.ndarray, Optional[Tuple[float, float]]]]
    ) -> List[Tuple[float, List[int]]]:
        """ 计算路径用时并返回索引 """
        if not self._manager:
            raise RuntimeError("未连接到RPC服务器，请先调用connect()")
        return self._multi_stop_router.batch_calc_route_duration_with_indexes(params)
