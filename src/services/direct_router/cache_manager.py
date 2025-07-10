from queue import Queue
import threading

import click
import pandas as pd
import time
import os
from typing import Dict, Optional, Tuple, List  # 确保导入 List

CACHE_DIR = './cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'path_duration.csv')
BATCH_SIZE = 1000  # 每次写入的批量大小
FLUSH_INTERVAL = 10.0  # 刷新间隔（秒）
NUM_SHARDS = 16  # 缓存分片数量, 可以根据CPU核心数或经验调整


def _deterministic_hash(key: Tuple[int, int]) -> int:
    """确定性hash函数，用于分片索引计算
    使用简单的数学运算保证高性能和确定性
    """
    # 使用两个质数进行哈希，确保分布均匀
    a, b = key
    return (a * 31 + b * 37) % (2**32)


class CacheManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, save_to_file: bool = False):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._queue = Queue()
                    # 初始化分片缓存和对应的锁
                    cls._instance._shards: List[Dict[Tuple[int, int], float]] = [{} for _ in range(NUM_SHARDS)]
                    cls._instance._shard_locks: List[threading.Lock] = [threading.Lock() for _ in range(NUM_SHARDS)]
                    cls._instance._file_lock = threading.Lock()
                    cls._instance._worker_thread = None
                    cls._instance._running = False
                    cls._instance._load_cache()
                    if save_to_file:
                        cls._instance._running = True
                        cls._instance._worker_thread = threading.Thread(target=cls._instance._worker, daemon=True)
                        cls._instance._worker_thread.start()
        return cls._instance

    def _load_cache(self):
        """加载已有缓存到各个分片"""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        if os.path.exists(CACHE_FILE):
            try:
                df_cache = pd.read_csv(
                    CACHE_FILE,
                    dtype={'start_id': int, 'end_id': int, 'duration': float}
                )
                loaded_count = 0
                for _, row in df_cache.iterrows():
                    # 内联 _generate_key
                    start_id = int(row.start_id)  # 确保是整数类型
                    end_id = int(row.end_id)  # 确保是整数类型
                    key = (min(start_id, end_id), max(start_id, end_id))

                    # 内联 _get_shard 逻辑
                    shard_index = _deterministic_hash(key) % NUM_SHARDS
                    cache_shard = self._shards[shard_index]
                    lock = self._shard_locks[shard_index]

                    with lock:
                        cache_shard[key] = float(row.duration)
                    loaded_count += 1
                click.echo(f"缓存加载成功，已加载 {loaded_count} 条记录到 {NUM_SHARDS} 个分片中")
            except Exception as e:
                click.echo(f"加载缓存失败: {e}")

    def add_item(self, start_id: int, end_id: int, duration: float):
        """添加一项到写入队列，并更新对应分片的缓存"""
        # 内联 _generate_key
        key = (min(start_id, end_id), max(start_id, end_id))

        # 内联 _get_shard 逻辑
        shard_index = _deterministic_hash(key) % NUM_SHARDS
        cache_shard = self._shards[shard_index]
        lock = self._shard_locks[shard_index]

        with lock:
            cache_shard[key] = duration

        if self._running:
            self._queue.put((key[0], key[1], duration))

    def get_from_cache(self, start_id: int, end_id: int) -> Optional[float]:
        """从对应的分片缓存中获取数据"""
        if start_id == end_id:
            return 0.0

        # 内联 _generate_key
        key = (min(start_id, end_id), max(start_id, end_id))

        # 内联 _get_shard 逻辑
        shard_index = _deterministic_hash(key) % NUM_SHARDS
        cache_shard = self._shards[shard_index]
        lock = self._shard_locks[shard_index]

        with lock:
            return cache_shard.get(key)

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

        # 处理队列中剩余的数据
        remaining_items = []
        while not self._queue.empty():
            try:  # 添加try-except以防万一在清空队列时发生意外
                remaining_items.append(self._queue.get_nowait())  # 使用get_nowait避免阻塞
                self._queue.task_done()
            except Exception:  # queue.Empty or other
                break

        if remaining_items:
            self._flush_buffer(remaining_items)
