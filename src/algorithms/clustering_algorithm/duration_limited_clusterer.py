from concurrent.futures import ProcessPoolExecutor
import math
import os

import click
import numpy as np
from typing import Tuple, Dict, List
from src.algorithms.clustering_algorithm.clustering_algorithm import ClusteringAlgorithm
from src.data.data_loader import load_map_data
from src.services.direct_router.direct_router_rpc_client import DirectRouterRpcClient
from src.services.multi_stop_router.multi_stop_router_rpc_client import MultiStopRouterRpcClient
from src.utils.utils import random_choice


def find_closest_warehouse(cluster_points: np.ndarray, warehouse_coords: List[Tuple[float, float]]) -> Tuple[
    float, float]:
    """根据簇中心点找到距离最近的仓库"""
    cluster_center = np.mean(cluster_points, axis=0)
    warehouse_array = np.array(warehouse_coords)
    distances = np.linalg.norm(cluster_center - warehouse_array, axis=1)
    closest_idx = np.argmin(distances)
    return warehouse_coords[closest_idx]


class DurationLimitedClusterer(ClusteringAlgorithm):
    def __init__(self, warehouse_coords: List[Tuple[float, float]], points: np.ndarray,
                 per_delivery_duration: int, work_duration: int):
        super().__init__(points, 1.3)
        self.warehouse_coords = warehouse_coords
        self.per_delivery_duration = per_delivery_duration
        self.work_duration = work_duration
        self.min_indicator = math.inf
        self.cache_saturation_ratio: Dict[int, Tuple[float, int]] = {}
        self.cache_stop = 0
        self.cache_count_clusters = 0

    def save_checkpoint(self, labels: np.ndarray, progress: float) -> Tuple[bool, bool]:
        total_duration, max_duration, min_duration, std_duration = self.calc_duration(labels)

        limited_duration = self.work_duration * 1.3
        if 0.5 < progress <= 0.8:
            limited_duration = self.work_duration * 1.4
        elif progress > 0.8:
            limited_duration = self.work_duration * 1.5

        available = max_duration < limited_duration
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
            ([(float(lat), float(lon)) for lat, lon in self.points[labels == l]],
             find_closest_warehouse(self.points[labels == l], self.warehouse_coords))
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
            ([(float(lat), float(lon)) for lat, lon in self.points[labels == l]],
             find_closest_warehouse(self.points[labels == l], self.warehouse_coords))
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
                                   warehouse_coords: List[Tuple[float, float]], per_delivery_duration: int,
                                   work_duration: int):
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  # 强制使用spawn模式

    un_zone_labels = np.unique(zone_labels)

    workers = min(math.ceil(os.cpu_count() / 2), len(un_zone_labels))
    click.echo(f"使用 {workers} 个工作线程进行并行聚类...")
    executor = ProcessPoolExecutor(max_workers=workers)
    list(executor.map(load_map_data, [bbox] * workers))
    click.echo("完成加载地图数据，开始进行区域聚类...")

    # 计算每个区域的订单数量并按数量从多到少排序
    zone_counts = [(zone_label, np.sum(zone_labels == zone_label)) for zone_label in un_zone_labels]
    sorted_zones = sorted(zone_counts, key=lambda x: x[1], reverse=True)
    features = []
    for zone_label, _ in sorted_zones:
        zone_points = points[zone_labels == zone_label]
        features.append(executor.submit(duration_limited_cluster, zone_points, warehouse_coords,
                                        per_delivery_duration, work_duration, int(zone_label) + 1))

    result_labels = np.full(len(zone_labels), -1)
    for (zone_label, _), feature in zip(sorted_zones, features):
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


def duration_limited_cluster(points: np.ndarray, warehouse_coords: List[Tuple[float, float]],
                             per_delivery_duration: int,
                             work_duration: int, zone: int = 1):
    # 初始化RPC客户端
    DirectRouterRpcClient().init()
    MultiStopRouterRpcClient().init()

    num_clusters = math.ceil(points.shape[0] / (work_duration / 3600 * 5))  # 每小时5单是一个偏小的假设
    centroids = random_choice(points, max(num_clusters, 2))  # 至少需要2个中心点
    clusterer = DurationLimitedClusterer(warehouse_coords, points, per_delivery_duration, work_duration)
    labels, _ = clusterer.clustering(centroids, step=3, zone=zone)
    return labels
