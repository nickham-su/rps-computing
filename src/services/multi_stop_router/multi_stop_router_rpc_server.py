import multiprocessing
import os
import time
from typing import Tuple, Optional
import Pyro5.server
import Pyro5.client
import Pyro5.errors
import click

Pyro5.config.SERVERTYPE = "multiplex"
Pyro5.config.SERIALIZER = 'marshal'

SERVER_ADDRESS = 'localhost'
SERVER_PORT = 50002
PYRO_OBJECT_ID = "multi_stop_router"


# Helper function to run the Pyro server in a separate process
def _run_pyro_server_process(shutdown_event: multiprocessing.Event, bbox: Tuple[float, float, float, float],
                             workers_count: int):
    """
    在单独的进程中运行Pyro5服务器。
    """
    from src.services.multi_stop_router.multi_stop_router import MultiStopRouter
    from src.services.multi_stop_router.multi_stop_router_executor import MultiStopRouterProcessExecutor
    from src.data.data_loader import load_map_data
    try:
        # 初始化进程池执行器，并加载地图数据
        multi_stop_router_executor = MultiStopRouterProcessExecutor()
        multi_stop_router_executor.init(workers_count)
        futures = [multi_stop_router_executor.submit(load_map_data, bbox) for _ in range(workers_count)]
        for future in futures:
            future.result()

        daemon = Pyro5.server.Daemon(host=SERVER_ADDRESS, port=SERVER_PORT)
        daemon.register(MultiStopRouter, objectId=PYRO_OBJECT_ID)
        daemon.requestLoop(loopCondition=lambda: not shutdown_event.is_set())
        click.echo(f"MultiStopRouter RPC服务器 (子进程 PID: {os.getpid()}) 正在关闭...")
        daemon.close()  # 清理Pyro守护进程
        MultiStopRouterProcessExecutor().shutdown()  # 清理进程池执行器
        click.echo(f"MultiStopRouter RPC服务器 (子进程 PID: {os.getpid()}) 已关闭。")
    except Exception as e:
        click.echo(f"MultiStopRouter RPC服务器 (子进程 PID: {os.getpid()}) 运行错误: {e}")


class MultiStopRouterRPCServer:
    """ MultiStopRouter RPC服务器 """

    def __init__(self):
        self.server_process: Optional[multiprocessing.Process] = None  # 后台进程
        self.shutdown_event: Optional[multiprocessing.Event] = None  # 用于通知进程关闭的事件

    def _wait_for_server_ready(self, timeout: int) -> bool:
        """ 等待服务器就绪并尝试连接 """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.server_process or not self.server_process.is_alive():
                return False
            try:
                client = Pyro5.client.Proxy(
                    f"PYRO:{PYRO_OBJECT_ID}@{SERVER_ADDRESS}:{SERVER_PORT}")
                return client.connected()
            except Pyro5.errors.CommunicationError:
                time.sleep(1)
            except Exception as e:
                click.echo(f"连接到 MultiStopRouter RPC 服务器时发生未知错误: {e}")
                return False

        click.echo(f"MultiStopRouter RPC服务器在 {timeout} 秒内未能成功连接。请检查服务器日志。")
        return False

    def start_server(self, bbox: Tuple[float, float, float, float], workers: Optional[int] = None,
                     connection_timeout: int = 180):
        """ 启动RPC服务器（非阻塞） """
        multiprocessing.set_start_method('spawn', force=True)  # 强制使用spawn模式
        
        if self.is_running():
            click.echo("MultiStopRouter RPC服务器已经在运行中")
            return

        # 创建用于进程间通信的事件
        self.shutdown_event = multiprocessing.Event()
        workers_count = workers or (os.cpu_count() - 1 if os.cpu_count() > 1 else 1)
        # 在后台进程中启动服务器
        self.server_process = multiprocessing.Process(
            target=_run_pyro_server_process,
            args=(self.shutdown_event, bbox, workers_count)
        )
        self.server_process.start()
        # 等待服务器就绪
        if not self._wait_for_server_ready(connection_timeout):
            click.echo("由于服务器未能成功启动或连接，将尝试停止服务器。")
            self.stop_server()  # 如果等待失败，则停止服务器
            return
        click.echo(f"MultiStopRouter RPC服务器已成功启动。子进程 PID: {self.server_process.pid}")

    def stop_server(self):
        """ 停止RPC服务器 """
        if not self.is_running() and not self.shutdown_event:  # Avoid issues if called multiple times or if not started
            click.echo("MultiStopRouter RPC服务器未运行或已在停止过程中。")
            return

        if self.shutdown_event:
            self.shutdown_event.set()  # 通知服务器进程关闭

        if self.server_process:
            click.echo("正在等待 MultiStopRouter RPC 服务器进程关闭...")
            self.server_process.join(timeout=10)  # 等待进程结束，最多10秒
            if self.server_process.is_alive():
                click.echo("RPC 服务器进程未能优雅关闭，将尝试强制终止。")
                self.server_process.terminate()  # 如果超时仍未结束，则强制终止
                self.server_process.join(timeout=5)  # 等待终止完成
                if self.server_process.is_alive():
                    click.echo("RPC 服务器进程强制终止失败。")
            else:
                click.echo("MultiStopRouter RPC 服务器进程已成功关闭。")

        self.server_process = None
        self.shutdown_event = None

        click.echo("MultiStopRouter RPC服务器停止流程已执行。")

    def is_running(self) -> bool:
        """ 检查服务器是否在运行 """
        return self.server_process is not None and self.server_process.is_alive()
