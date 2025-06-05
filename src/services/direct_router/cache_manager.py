from queue import Queue
import threading

import click
import pandas as pd
import time
import os
from typing import Dict, Optional, Tuple
from rwlock import RWLock

CACHE_DIR = './cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'path_duration.csv')
BATCH_SIZE = 1000  # 每次写入的批量大小
FLUSH_INTERVAL = 10.0  # 刷新间隔（秒）


class CacheManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, save_to_file: bool = False):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._queue = Queue()
                    cls._instance._cache_lock = RWLock()
                    cls._instance._cache = {}  # Dict[Tuple[int, int], float]
                    cls._instance._file_lock = threading.Lock()
                    cls._instance._worker_thread = None
                    cls._instance._running = False
                    cls._instance._load_cache()
                    if save_to_file:
                        cls._instance._running = True
                        cls._instance._worker_thread = threading.Thread(target=cls._instance._worker, daemon=True)
                        cls._instance._worker_thread.start()

        return cls._instance

    def _generate_key(self, start_id: int, end_id: int) -> Tuple[int, int]:
        """生成缓存键 - 返回元组"""
        return min(start_id, end_id), max(start_id, end_id)

    def _load_cache(self):
        """加载已有缓存"""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        if os.path.exists(CACHE_FILE):
            try:
                cache = pd.read_csv(
                    CACHE_FILE,
                    dtype={'start_id': int, 'end_id': int, 'duration': float}
                )
                with self._cache_lock.writer_lock:
                    for _, row in cache.iterrows():
                        key = self._generate_key(row.start_id, row.end_id)
                        self._cache[key] = float(row.duration)
                click.echo(f"缓存加载成功，已加载 {len(self._cache)} 条记录")
            except Exception as e:
                click.echo(f"加载缓存失败: {e}")

    def add_item(self, start_id: int, end_id: int, duration: float):
        """添加一项到写入队列"""
        key = self._generate_key(start_id, end_id)
        with self._cache_lock.writer_lock:
            self._cache[key] = duration
        if self._running:
            self._queue.put((key[0], key[1], duration))

    def get_from_cache(self, start_id: int, end_id: int) -> Optional[float]:
        """从缓存中获取数据"""
        if start_id == end_id:
            return 0.0

        key = self._generate_key(start_id, end_id)
        with self._cache_lock.reader_lock:
            return self._cache.get(key)

    def _worker(self):
        """后台工作线程处理写入操作"""
        buffer = []
        last_flush_time = time.time()

        while self._running:
            try:
                item = self._queue.get(timeout=1)
                buffer.append(item)
                self._queue.task_done()

                current_time = time.time()
                if len(buffer) >= BATCH_SIZE or (current_time - last_flush_time) >= FLUSH_INTERVAL:
                    self._flush_buffer(buffer)
                    buffer = []
                    last_flush_time = current_time

            except Exception as e:
                # 超时或其他异常
                current_time = time.time()
                if buffer and (current_time - last_flush_time) >= FLUSH_INTERVAL:
                    self._flush_buffer(buffer)
                    buffer = []
                    last_flush_time = current_time

        # 处理剩余的缓冲区数据
        if buffer:
            self._flush_buffer(buffer)

    def _flush_buffer(self, buffer):
        """将缓冲区数据写入文件"""
        if not buffer:
            return

        with self._file_lock:
            try:
                df = pd.DataFrame(buffer, columns=['start_id', 'end_id', 'duration'])
                mode = 'a' if os.path.exists(CACHE_FILE) else 'w'
                header = not os.path.exists(CACHE_FILE)
                df.to_csv(CACHE_FILE, mode=mode, header=header, index=False)
            except Exception as e:
                click.echo(f"缓存写入错误: {e}")

    def shutdown(self):
        """关闭写入器并刷新剩余数据"""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join()

        remaining_items = []
        while not self._queue.empty():
            remaining_items.append(self._queue.get())
            self._queue.task_done()

        if remaining_items:
            self._flush_buffer(remaining_items)
