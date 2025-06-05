from concurrent.futures import ProcessPoolExecutor
import math
import os

import click
import numpy as np
from typing import Tuple, Dict
from src.algorithms.clustering_algorithm.clustering_algorithm import ClusteringAlgorithm
from src.data.data_loader import load_map_data
from src.services.multi_stop_router.multi_stop_router_rpc_client import MultiStopRouterRpcClient
from src.utils.utils import random_choice


class DurationLimitedClusterer(ClusteringAlgorithm):
    def __init__(self, warehouse_coord: Tuple[float, float], points: np.ndarray,
                 per_delivery_duration: int, work_duration: int):
        super().__init__(points, 1.3)
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

    def saturation_ratio(self, labels: np.ndarray, update_labels: np.ndarray, count_clusters: int):
        # 如果stop和count_clusters改变，则清空缓存
        if self.cache_stop != self.current_step or self.cache_count_clusters != count_clusters:
            self.cache_saturation_ratio = {}
            self.cache_stop = self.current_step
            self.cache_count_clusters = count_clusters

        refresh_labels = []
        for cluster_label in update_labels:
            if cluster_label not in self.cache_saturation_ratio:
                refresh_labels.append(cluster_label)
            else:
                count_points = np.sum(labels == cluster_label)
                cache_saturation_ratio, cache_count_points = self.cache_saturation_ratio[cluster_label]
                saturation_ratio_per_point = cache_saturation_ratio / cache_count_points
                estimated_saturation_ratio = saturation_ratio_per_point * count_points
                # 如果估算饱和度与缓存饱和度差距不大，则直接返回估算饱和度，如果差距较大，则重新计算
                max_diff = 0.2 if estimated_saturation_ratio < 0.9 else 0.1
                if estimated_saturation_ratio - cache_saturation_ratio > max_diff:
                    refresh_labels.append(cluster_label)

        batch_calc_route_duration_params = [
            ([(float(lat), float(lon)) for lat, lon in self.points[labels == l]], self.warehouse_coord)
            for l in refresh_labels
        ]
        # 在途时间
        travel_time_list = MultiStopRouterRpcClient().batch_calc_route_duration(batch_calc_route_duration_params)
        # 配送时间
        delivery_time_list = [self.per_delivery_duration * len(self.points[labels == l]) for l in refresh_labels]
        # 计算饱和度
        saturation_ratios_list = (np.array(delivery_time_list) + np.array(travel_time_list)) / self.work_duration
        for i in range(len(refresh_labels)):
            label = refresh_labels[i]
            count_points = np.sum(labels == label)
            saturation_ratio = saturation_ratios_list[i]
            self.cache_saturation_ratio[label] = (saturation_ratio, count_points)

        results = []
        for cluster_label in update_labels:
            if cluster_label not in self.cache_saturation_ratio:
                results.append(0.0)
            else:
                count_points = np.sum(labels == cluster_label)
                cache_saturation_ratio, cache_count_points = self.cache_saturation_ratio[cluster_label]
                saturation_ratio_per_point = cache_saturation_ratio / cache_count_points
                estimated_saturation_ratio = saturation_ratio_per_point * count_points
                results.append(estimated_saturation_ratio)
        return results

    def calc_duration(self, labels: np.ndarray) -> (float, float, float):
        un_labels = np.unique(labels)
        batch_calc_route_duration_params = [
            ([(float(lat), float(lon)) for lat, lon in self.points[labels == l]], self.warehouse_coord)
            for l in un_labels
        ]
        # 在途时间
        travel_time_list = MultiStopRouterRpcClient().batch_calc_route_duration(batch_calc_route_duration_params)
        # 配送时间
        delivery_time_list = [self.per_delivery_duration * len(self.points[labels == l]) for l in un_labels]
        # 计算总时间
        routes_duration = np.array(delivery_time_list) + np.array(travel_time_list)
        return sum(routes_duration), max(routes_duration), min(routes_duration), np.std(routes_duration)


def batch_duration_limited_cluster(bbox: Tuple[float, float, float, float], points: np.ndarray, zone_labels: np.ndarray,
                                   warehouse_coord: Tuple[float, float], per_delivery_duration: int, work_duration: int,
                                   init_cluster_size: int = 100):
    un_zone_labels = np.unique(zone_labels)

    workers = min(math.ceil(os.cpu_count() / 2), len(un_zone_labels))
    # workers = 2
    executor = ProcessPoolExecutor(max_workers=workers)
    list(executor.map(load_map_data, [bbox] * workers))
    features = []

    for zone_label in un_zone_labels:
        zone_points = points[zone_labels == zone_label]
        features.append(executor.submit(duration_limited_cluster, zone_points, warehouse_coord,
                                        per_delivery_duration, work_duration, init_cluster_size, int(zone_label) + 1))

    result_labels = np.full(len(zone_labels), -1)
    for zone_label, feature in zip(un_zone_labels, features):
        per_zone_labels = feature.result()
        result_labels[zone_labels == zone_label] = per_zone_labels + zone_label * 10000

    # 关闭线程池
    executor.shutdown(wait=True)

    if np.sum(result_labels == -1) > 0:
        raise ValueError("批量排线失败，部分区域未能分配标签。请检查输入数据。")

    # 将result_labels转换为顺序标签
    un_result_labels = np.unique(result_labels)
    sorted_result_labels = np.zeros_like(result_labels)
    for i, label in enumerate(un_result_labels):
        sorted_result_labels[result_labels == label] = i

    return sorted_result_labels


def duration_limited_cluster(points: np.ndarray, warehouse_coord: Tuple[float, float], per_delivery_duration: int,
                             work_duration: int, init_cluster_size: int = 100, zone: int = 1):
    num_clusters = math.ceil(points.shape[0] / init_cluster_size)
    centroids = random_choice(points, num_clusters)
    clusterer = DurationLimitedClusterer(warehouse_coord, points, per_delivery_duration, work_duration)
    labels, _ = clusterer.clustering(centroids, step=3, max_iter=30, zone=zone)
    return labels
