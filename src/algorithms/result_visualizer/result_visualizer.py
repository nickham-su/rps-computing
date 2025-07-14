import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import gzip
from typing import Optional, Dict, List, Tuple
from src.algorithms.result_visualizer.utils.draw_map import draw_map
from src.services.direct_router.direct_router_rpc_client import DirectRouterRpcClient
from src.services.geo_indexer.geo_indexer import GeoIndexer
from src.services.multi_stop_router.multi_stop_router_rpc_client import MultiStopRouterRpcClient
from src.algorithms.clustering_algorithm.duration_limited_clusterer import find_closest_warehouse


class ResultVisualizer:

    def __init__(self, points: np.ndarray, labels: np.ndarray,
                 warehouse_coords: List[Tuple[float, float]], per_delivery_duration: int):
        self.points = points
        self.labels = (labels + 1).copy()
        self.warehouse_coords = warehouse_coords
        self.per_delivery_duration = per_delivery_duration
        self.route_series: Optional[np.ndarray] = None
        self.route_driving_duration: Dict[int, float] = {}
        self.route_first_point_duration: Dict[int, float] = {}
        # 初始化RPC客户端
        DirectRouterRpcClient().init()
        MultiStopRouterRpcClient().init()

    def _calc_route_series(self):
        self.route_series = np.full(self.points.shape[0], -1)
        un_labels = np.unique(self.labels)
        # 为每个簇计算最近仓库并生成路径参数
        batch_calc_params = []
        cluster_warehouses = {}  # 记录每个簇对应的最近仓库

        for l in un_labels:
            cluster_points = self.points[self.labels == l]
            closest_warehouse = find_closest_warehouse(cluster_points, self.warehouse_coords)
            cluster_warehouses[l] = closest_warehouse
            batch_calc_params.append((
                [(float(lat), float(lon)) for lat, lon in cluster_points],
                closest_warehouse
            ))
        geo_indexer = GeoIndexer()
        direct_router_rpc_client = DirectRouterRpcClient()
        results = MultiStopRouterRpcClient().batch_calc_route_duration_with_indexes(batch_calc_params)
        for (duration, indexes), label in zip(results, un_labels):
            self.route_driving_duration[label] = duration
            self.route_series[self.labels == label] = np.array(indexes) + 1
            cluster_points = self.points[self.labels == label]
            first_index = np.argmin(indexes)
            lat, lon = cluster_points[first_index]
            self.route_first_point_duration[label] = direct_router_rpc_client.calc_path_duration(
                geo_indexer.get_nearest_node_id(cluster_warehouses[label]),
                geo_indexer.get_nearest_node_id((lat, lon))
            )

    def draw_map(self):
        return draw_map(self.points, self.labels, self.warehouse_coords)

    def statistical(self):
        if self.route_series is None:
            self._calc_route_series()
        route_list = []
        for label in np.unique(self.labels):
            count_point = len(self.points[self.labels == label])
            driving_duration = round(self.route_driving_duration[label] / 3600, 2)
            delivery_duration = round(count_point * self.per_delivery_duration / 3600, 2)
            work_duration = round(driving_duration + delivery_duration, 2)
            first_point_duration = round(self.route_first_point_duration[label] / 3600, 2)
            route_list.append(
                (label, count_point, driving_duration, first_point_duration, delivery_duration, work_duration))
        return pd.DataFrame(route_list,
                            columns=['label', 'count_orders', 'travel_time', 'time_to_first_point',
                                     'delivery_time', 'total_working_time'])

    def export(self, order_df: pd.DataFrame, output_dir: Optional[str] = None) -> str:
        if output_dir is None:
            output_dir = f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 导出线路数据
        stat_df = self.statistical()
        stat_df.to_excel(f'{output_dir}/all_routes.xlsx', index=False)
        # 导出全部数据
        df = order_df.copy()
        df['label'] = self.labels
        df['route_series'] = self.route_series
        df.to_excel(f'{output_dir}/all_address.xlsx')
        # 导出线路
        for label in np.unique(df['label']):
            cluster_df = df[df['label'] == label].sort_values('route_series')
            cluster_df.to_excel(f'{output_dir}/route_{label}.xlsx', index=False)
        # 生成zip压缩文件
        shutil.make_archive(output_dir, 'zip', output_dir)
        shutil.rmtree(output_dir)
        # 返回压缩文件绝对路径
        return os.path.abspath(f'{output_dir}.zip')

    def export_json(self, order_df: pd.DataFrame, output_dir: Optional[str] = None) -> str:
        if output_dir is None:
            output_dir = f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 导出线路数据
        stat_df = self.statistical()
        route_list = stat_df.to_json(orient='records')
        # 导出全部数据
        df = order_df.copy()
        df['label'] = self.labels
        df['route_series'] = self.route_series
        order_list = df.to_json(orient='records')
        json_str = f'{{"version": 1.0, "routes": {route_list}, "orders": {order_list}}}'
        # 使用gzip压缩后，将json_str保存到文件
        output_filepath = f'{output_dir}/result.json.gz'
        with gzip.open(output_filepath, 'wb') as f:
            f.write(json_str.encode('utf-8'))
        # 返回压缩文件绝对路径
        return os.path.abspath(output_filepath)
