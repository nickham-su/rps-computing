import math

import click
import numpy as np
from typing import Tuple, Dict
from src.algorithms.clustering_algorithm.clustering_algorithm import ClusteringAlgorithm
from src.services.multi_stop_router.multi_stop_router import MultiStopRouter


class DurationLimitedClusterer(ClusteringAlgorithm):
    def __init__(self, warehouse_coord: Tuple[float, float], points: np.ndarray,
                 per_delivery_duration: int, work_duration: int):
        super().__init__(points)
        self.warehouse_coord = warehouse_coord
        self.per_delivery_duration = per_delivery_duration
        self.work_duration = work_duration
        self.min_indicator = math.inf
        self.cache_saturation_ratio: Dict[int, Tuple[float, int]] = {}
        self.cache_stop = 0
        self.cache_count_clusters = 0

    def save_checkpoint(self, labels: np.ndarray) -> Tuple[bool, bool]:
        total_duration, max_duration, min_duration, std_duration = self.calc_duration(labels)
        available = max_duration < self.work_duration * 1.3
        savable = available and std_duration < self.min_indicator * 0.95
        if savable:
            self.min_indicator = std_duration
        # click.echo(
        #     f'total: {(total_duration / 3600):.2f}, max:{(max_duration / 3600):.2f}, min:{(min_duration / 3600):.2f}, std:{(std_duration / 3600):.2f}, available:{available}, savable:{savable}')
        return available, savable

    def saturation_ratio(self, cluster_points: np.ndarray, cluster_label: int, count_clusters: int):
        # 如果stop和count_clusters改变，则清空缓存
        if self.cache_stop != self.current_step or self.cache_count_clusters != count_clusters:
            self.cache_saturation_ratio = {}
            self.cache_stop = self.current_step
            self.cache_count_clusters = count_clusters

        count_points = len(cluster_points)
        if cluster_label in self.cache_saturation_ratio:
            cache_saturation_ratio, cache_count_points = self.cache_saturation_ratio[cluster_label]
            saturation_ratio_per_point = cache_saturation_ratio / cache_count_points
            estimated_saturation_ratio = saturation_ratio_per_point * count_points
            # 如果估算饱和度与缓存饱和度差距不大，则直接返回估算饱和度，如果差距较大，则重新计算
            max_diff = 0.1 if estimated_saturation_ratio < 0.95 else 0.05
            if estimated_saturation_ratio - cache_saturation_ratio < max_diff:
                return estimated_saturation_ratio

        # 配送时间
        duration = self.per_delivery_duration * count_points
        # 在途时间
        duration += MultiStopRouter.calc_route_duration(cluster_points, self.warehouse_coord)
        saturation_ratio = duration / self.work_duration
        self.cache_saturation_ratio[cluster_label] = (saturation_ratio, count_points)
        return saturation_ratio

    def calc_duration(self, labels: np.ndarray) -> (float, float, float):
        routes_duration = []
        for label in np.unique(labels):
            cluster_points = self.points[labels == label]
            count_points = len(cluster_points)
            # 配送时间
            duration = self.per_delivery_duration * count_points
            duration += MultiStopRouter.calc_route_duration(cluster_points, self.warehouse_coord)
            routes_duration.append(duration)
        return sum(routes_duration), max(routes_duration), min(routes_duration), np.std(routes_duration)
