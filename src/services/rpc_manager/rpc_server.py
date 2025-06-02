import os
import threading
from typing import Tuple, Optional
import click
from src.data.data_loader import load_map_data
from src.services.direct_router.direct_router import DirectRouter
from src.services.direct_router.direct_router_executor import DirectRouterProcessExecutor
from src.services.multi_stop_router.multi_stop_router_executor import MultiStopRouterProcessExecutor
from src.services.rpc_manager.rpc_manager import RPCManager, RPC_SERVER_ADDRESS, RPC_SERVER_AUTHKEY


class RPCServer:
    """ DirectRouter RPC服务器 """

    def __init__(self):
        self.manager = None  # 管理器实例
        self.server_thread = None  # 后台线程
        self.server = None  # RPC服务器实例

    def start_server(self, bbox: Tuple[float, float, float, float], workers: Optional[int] = None):
        """ 启动RPC服务器（非阻塞） """
        if self.server_thread and self.server_thread.is_alive():
            click.echo("服务器已经在运行中")
            return

        # 初始化进程池执行器，并加载地图数据
        workers = workers or os.cpu_count()  # 默认使用CPU核心数
        direct_router_executor = DirectRouterProcessExecutor()
        direct_router_executor.init(workers)
        futures = [direct_router_executor.submit(load_map_data, bbox) for _ in range(workers)]
        multi_stop_router_executor = MultiStopRouterProcessExecutor()
        multi_stop_router_executor.init(workers)
        futures = futures + [multi_stop_router_executor.submit(load_map_data, bbox) for _ in range(workers)]
        # 等待所有任务完成
        for future in futures:
            future.result()

        # 创建管理器
        self.manager = RPCManager(address=RPC_SERVER_ADDRESS, authkey=RPC_SERVER_AUTHKEY)
        self.server = self.manager.get_server()

        # 在后台线程中启动服务器
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

        click.echo(f"DirectRouter RPC服务器启动在 {RPC_SERVER_ADDRESS}")

    def _run_server(self):
        """ 在后台线程中运行服务器 """
        try:
            self.server.serve_forever()
        except Exception as e:
            click.echo(f"服务器运行错误: {e}")

    def stop_server(self):
        """ 停止RPC服务器 """
        if self.server:
            self.server.stop_event.set()  # 停止serve_forever

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)  # 等待线程结束，最多5秒

        # 清理DirectRouter
        DirectRouter.shutdown()
        # 清理进程池执行器
        DirectRouterProcessExecutor().shutdown()
        MultiStopRouterProcessExecutor().shutdown()

        click.echo("DirectRouter RPC服务器已停止")

    def is_running(self):
        """ 检查服务器是否在运行 """
        return self.server_thread and self.server_thread.is_alive()
