import click
import numpy as np
from datetime import datetime
from typing import Tuple, List, Set, Optional
from src.data.data_loader import load_map_data
from src.services.geo_indexer.geo_indexer import GeoIndexer
from src.services.direct_router.direct_router import DirectRouter


def preprocess_data(warehouse_coord: Tuple[float, float], points: np.ndarray, k: int, workers: Optional[int] = None):
    """ 预处理数据 """
    load_map_data(warehouse_coord, points)
    geo_service = GeoIndexer()
    path_service = DirectRouter(warehouse_coord, points, workers)

    # 待处理的点对
    point_pairs_list: List[Tuple[int, int]] = []
    added_set: Set[str] = set()

    # 仓库到收货点
    start_id = geo_service.get_nearest_node_id(warehouse_coord)
    for p in points:
        end_id = geo_service.get_nearest_node_id(p)
        key = f"{min(start_id, end_id)}_{max(start_id, end_id)}"
        if start_id == end_id \
                or key in added_set \
                or path_service.get_path_duration_from_cache(start_id, end_id):
            continue
        point_pairs_list.append((start_id, end_id))
        added_set.add(key)

    # 计算最邻近的点
    distances = np.linalg.norm(np.array(points)[:, np.newaxis] - np.array(points), axis=2)
    for i in range(len(points)):
        start_id = geo_service.get_nearest_node_id(points[i])
        nearest_indices = np.argsort(distances[i])
        for j in nearest_indices[1:k]:
            end_id = geo_service.get_nearest_node_id(points[j])
            key = f"{min(start_id, end_id)}_{max(start_id, end_id)}"
            if start_id == end_id \
                    or key in added_set \
                    or path_service.get_path_duration_from_cache(start_id, end_id):
                continue
            point_pairs_list.append((start_id, end_id))
            added_set.add(key)

    click.echo(f'待计算数量: {len(point_pairs_list)}')
    # 分批查询路径用时；每步计算都会写到本地文件，防止计算中断丢失数据
    chunk_size = 2000
    for i in range(0, len(point_pairs_list), chunk_size):
        current_time = datetime.now()
        click.echo(f'预处理进度:{i}/{len(point_pairs_list)} - {current_time.strftime("%H:%M:%S")}')
        chunk = point_pairs_list[i:i + chunk_size]
        path_service.batch_calc_path_duration(chunk)
    click.echo('预处理完成')
