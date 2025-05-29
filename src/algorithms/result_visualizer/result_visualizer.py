import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import gzip
from typing import Optional, Dict
from src.algorithms.result_visualizer.modules.draw_map import draw_map
from src.services.multi_stop_router.multi_stop_router import MultiStopRouter


class ResultVisualizer:

    def __init__(self, points: np.ndarray, labels: np.ndarray,
                 warehouse_coord: tuple[float, float], per_delivery_duration: int):
        self.points = points
        self.labels = (labels + 1).copy()
        self.warehouse_coord = warehouse_coord
        self.per_delivery_duration = per_delivery_duration
        self.route_series: Optional[np.ndarray] = None
        self.route_driving_duration: Dict[int, float] = {}
        self.route_first_point_duration: Dict[int, float] = {}

    def _calc_route_series(self):
        self.route_series = np.full(self.points.shape[0], -1)
        for label in np.unique(self.labels):
            cluster_points = self.points[self.labels == label]
            # 计算路径
            points_sorted, route_node_ids = MultiStopRouter.routing(cluster_points, self.warehouse_coord)
            # 计算路径用时
            duration = MultiStopRouter.get_route_duration(route_node_ids, self.warehouse_coord)
            self.route_driving_duration[label] = duration
            # 计算从仓库到达第一个点的时间
            self.route_first_point_duration[label] = MultiStopRouter.get_route_duration(route_node_ids[:1],
                                                                                        self.warehouse_coord)
            # 记录路径编号
            self.route_series[self.labels == label] = points_sorted + 1

    def draw_map(self):
        if self.route_series is None:
            self._calc_route_series()
        return draw_map(self.points, self.route_series, self.labels)

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
