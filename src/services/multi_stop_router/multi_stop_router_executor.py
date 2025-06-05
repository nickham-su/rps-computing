import os
import threading
from typing import Optional
from concurrent.futures import ProcessPoolExecutor


class MultiStopRouterProcessExecutor:
    """ 进程池执行器单例 """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._executor = None
        return cls._instance

    def init(self, workers: Optional[int] = None):
        """ 初始化进程池执行器 """
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True)  # 正确关闭旧的executor

            workers = workers if workers is not None else max(1, os.cpu_count() - 1)
            self._executor = ProcessPoolExecutor(max_workers=workers)

    def submit(self, fn, *args, **kwargs):
        """ 提交任务到进程池 """
        if self._executor is None:
            raise RuntimeError("ProcessExecutor 尚未初始化。请先调用 init() 方法。")
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True):
        """ 关闭进程池 """
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=wait)
                self._executor = None

    def __del__(self):
        """ 析构时确保资源被释放 """
        self.shutdown(wait=False)
