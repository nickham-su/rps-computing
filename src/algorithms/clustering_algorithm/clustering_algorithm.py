import click
import numpy as np
from datetime import datetime
import time
from typing import Tuple, List
from abc import ABC, abstractmethod
from scipy.spatial import KDTree

from src.services.geo_indexer.geo_indexer import GeoIndexer
from src.services.rpc_manager.rpc_client import RPCClient
from src.utils.utils import random_choice, calc_weighted_centroid, find_nearest_points, max_distance


class ClusteringAlgorithm(ABC):

    def __init__(self, points: np.ndarray, split_cluster_threshold: float):
        self.current_step = 0
        self.points = points
        self.split_cluster_threshold = split_cluster_threshold
        self.timer = {
            't1': 0.0,
            't2': 0.0,
            't3': 0.0,
            't4': 0.0,
            't5': 0.0,
            't6': 0.0,
            't7': 0.0,
            't8': 0.0,
            't9': 0.0,
            't10': 0.0,
            't11': 0.0,
            't12': 0.0,
        }

    @staticmethod
    def update_centroids(points: np.ndarray, labels: np.ndarray):
        """ 更新簇中心 """
        new_centroids = []
        for l in np.unique(labels):
            cluster_points = points[labels == l]
            centroid = calc_weighted_centroid(cluster_points)
            count_points = len(cluster_points)
            neighbors, _ = find_nearest_points(centroid, cluster_points,
                                               max(round(count_points * 0.2), 1))
            p = random_choice(neighbors, 1)[0]
            new_centroids.append(p)
        return np.array(new_centroids)

    @abstractmethod
    def save_checkpoint(self, labels: np.ndarray) -> Tuple[bool, bool]:
        pass

    @abstractmethod
    def saturation_ratio(self, labels: np.ndarray, update_labels: np.ndarray, count_clusters: int):
        pass

    def assign_clusters(self, centroids: np.ndarray, can_add_centroid=True) \
            -> Tuple[np.ndarray, np.ndarray]:
        """ 分配每个数据点到最近的簇 """
        geo_indexer = GeoIndexer()
        rpc_client = RPCClient()
        points = self.points
        points_index = np.arange(points.shape[0])  # 点的索引
        labels = np.full(points.shape[0], -1)  # 初始化labels为-1，表示未分配
        count_clusters = centroids.shape[0]  # 簇的数量
        clusters_saturation_ratio = np.zeros(count_clusters)  # 簇的饱和度

        t1 = time.perf_counter()
        neighbor_number = round(points.shape[0] / count_clusters / 3)  # 每个点的邻域数量
        tree = KDTree(self.points)  # 构建KDTree索引
        distances, indices = tree.query(self.points, k=neighbor_number + 1)  # 查询每个点的邻近点，k+1是因为第一个点是自身
        neighbor_max_distances = distances[:, -1]  # 获取每个点的最大邻近距离
        # 计算点的密度权重
        points_density_weight = (neighbor_max_distances / np.mean(neighbor_max_distances)) ** (1 / 2)
        self.timer['t1'] += time.perf_counter() - t1

        t2 = time.perf_counter()
        # 将距离质心最近的几个点，直接分配给该簇
        n = min(int(points.shape[0] / count_clusters * 0.2), 10)
        centroid_distances = np.linalg.norm(centroids[:, np.newaxis] - points, axis=2)
        centroid_nearest = np.argsort(centroid_distances, axis=1)
        for i in range(count_clusters):
            labels[centroid_nearest[i][:n]] = i
        self.timer['t2'] += time.perf_counter() - t2

        while True:
            unallocated_indices = points_index[labels == -1]
            if len(unallocated_indices) == 0:
                break

            t3 = time.perf_counter()
            # 计算未分配点的权重
            weights = ClusteringAlgorithm.clac_unallocated_weight(points, labels,
                                                                  clusters_saturation_ratio, points_density_weight)
            self.timer['t3'] += time.perf_counter() - t3

            # 根据饱和度确定批量处理的数量
            max_saturation_ratio = np.max(clusters_saturation_ratio)
            allocate_number = count_clusters if max_saturation_ratio > 0.9 else \
                2 * count_clusters if max_saturation_ratio > 0.7 else \
                    3 * count_clusters if max_saturation_ratio > 0.4 else 5 * count_clusters

            # 估算簇的半径：计算第一个质心到最近质心的距离，除以2
            if count_clusters < 2:
                cluster_radius = 0.0
            else:
                centroid_distances = np.linalg.norm(centroids[0] - centroids[1:], axis=1)
                cluster_radius = np.min(centroid_distances) / 2

            t4 = time.perf_counter()
            # 分配
            allocated_points = points[labels != -1]  # 已分配的点
            allocated_labels = labels[labels != -1]  # 已分配的标签
            allocate_labels = []  # 本轮分配的标签列表
            allocate_points = []  # 本轮分配的点列表
            for i in np.argsort(weights)[::-1][:max(round(allocate_number), 1)]:  # 按照权重从大到小分配，共分配allocate_number个点
                current_index = unallocated_indices[i]  # 当前点的索引
                current_coord = points[current_index]  # 当前点的坐标
                # 当前点到本轮分配的点的距离小于簇半径时，当前点跳过本次分配
                # 原因是：距离相近的点，会影响当前的权重，需要在下一轮重新计算权重
                if (len(allocate_points) > 0 and cluster_radius > 0 and
                        np.any(np.linalg.norm(current_coord - allocate_points, axis=1) < cluster_radius)):
                    continue

                curr_to_allocated_distances = np.linalg.norm(current_coord - allocated_points, axis=1)  # 计算当前点到已分配点的距离
                nearest_index = np.argmin(curr_to_allocated_distances)  # 找到距离最近的已分配点
                current_label = allocated_labels[nearest_index]  # 当前点分配的标签
                labels[current_index] = current_label  # 将当前点分配给最近的已分配点的标签
                allocate_labels.append(current_label)  # 将当前点分配的标签添加到分配的标签列表
                allocate_points.append(current_coord)  # 将当前点添加到分配的点列表
            self.timer['t4'] += time.perf_counter() - t4

            t5 = time.perf_counter()
            # 计算饱和度
            update_labels = np.unique(allocate_labels)
            update_saturation_ratio_list = self.saturation_ratio(labels, update_labels, count_clusters)
            for allocate_label, saturation_ratio in zip(update_labels, update_saturation_ratio_list):
                clusters_saturation_ratio[allocate_label] = saturation_ratio
            self.timer['t5'] += time.perf_counter() - t5

        # 改变簇数量
        if can_add_centroid:
            saturation_sum = np.sum(clusters_saturation_ratio)
            if saturation_sum > count_clusters + 1.5:
                t6 = time.perf_counter()
                add_count = round(saturation_sum - count_clusters)
                # 饱和度最高的簇中添加质心
                allocate_labels = np.argsort(clusters_saturation_ratio)[::-1][:add_count]
                add_points = []
                for l in allocate_labels:
                    if np.sum(labels == l) > 0:
                        cluster_points = points[labels == l]
                        current_centroid = centroids[l]  # 当前簇的质心
                        # 从簇中选择一个不是当前质心的点
                        max_attempts = 3  # 最大尝试次数，避免无限循环
                        for attempt in range(max_attempts):
                            candidate_point = random_choice(cluster_points, 1)[0]
                            # 检查候选点是否是当前簇的质心
                            distance_to_centroid = np.linalg.norm(current_centroid - candidate_point)
                            if distance_to_centroid > 1e-10:  # 不是质心
                                add_points.append(candidate_point)
                                break
                self.timer['t6'] += time.perf_counter() - t6
                if add_points:  # 只有当成功添加了新点时才进行递归调用
                    centroids = np.vstack([centroids, np.array(add_points)])
                    return self.assign_clusters(centroids, can_add_centroid=False)
            elif saturation_sum < count_clusters - 1.5:
                t7 = time.perf_counter()
                remove_count = round(count_clusters - saturation_sum)
                # 饱和度最低的簇中删除质心
                remove_labels = np.argsort(clusters_saturation_ratio)[:remove_count]
                centroids = np.delete(centroids, remove_labels, axis=0)
                self.timer['t7'] += time.perf_counter() - t7
                return self.assign_clusters(centroids, can_add_centroid=False)
            elif np.max(clusters_saturation_ratio) > self.split_cluster_threshold:
                t8 = time.perf_counter()
                # 在饱和度最高的簇，添加一个质心，将其拆分
                max_label = np.argmax(clusters_saturation_ratio)
                cluster_points = points[labels == max_label]
                current_centroid = centroids[max_label]  # 当前簇的质心

                # 从簇中选择一个不是当前质心的点
                max_attempts = 3  # 最大尝试次数，避免无限循环
                new_centroid = None
                for attempt in range(max_attempts):
                    candidate_point = random_choice(cluster_points, 1)[0]
                    # 检查候选点是否是当前簇的质心
                    distance_to_centroid = np.linalg.norm(current_centroid - candidate_point)
                    if distance_to_centroid > 1e-10:  # 不是质心
                        new_centroid = candidate_point
                        break
                self.timer['t8'] += time.perf_counter() - t8
                if new_centroid is not None:
                    centroids = np.vstack([centroids, [new_centroid]])
                    # TODO: 优化递归逻辑
                    return self.assign_clusters(centroids, can_add_centroid=False)

        return labels, centroids

    @staticmethod
    def clac_unallocated_weight(points, labels, clusters_saturation_ratio, points_density_weight):
        unallocated_mask = (labels == -1)  # 未分配的点
        unallocated_points = points[unallocated_mask]  # 未分配的点坐标
        # unallocated_points_density_weight = points_density_weight[unallocated_mask]  # 未分配点的密度权重，保留备用
        weights = np.zeros(unallocated_points.shape[0])  # 初始化权重

        # 获取已分配点及其标签
        allocated_mask = (labels != -1)
        allocated_points = points[allocated_mask]
        allocated_labels = labels[allocated_mask]

        # 如果没有已分配点，直接返回零权重
        if len(allocated_points) == 0:
            return weights

        # 计算最大簇的点数n
        unique_labels, counts = np.unique(allocated_labels, return_counts=True)
        max_cluster_size = np.max(counts) if len(counts) > 0 else 1
        k = min(max_cluster_size + 1, len(allocated_points))  # 查询点数上限

        # 构建KDTree加速最近邻搜索
        kdtree = KDTree(allocated_points)

        # 批量查询所有未分配点的最近邻
        distances, indices = kdtree.query(unallocated_points, k=k)
        neighbor_labels = allocated_labels[indices]  # 每个未分配点的邻居标签

        # 处理每个未分配点
        for i in range(unallocated_points.shape[0]):
            # 初始化簇距离字典
            cluster_dist = {}

            # 遍历邻居点，记录不同簇的最短距离
            for j in range(k):
                label = neighbor_labels[i, j]
                dist = distances[i, j]

                # 如果该簇尚未记录或找到更近距离，更新距离
                if label not in cluster_dist or dist < cluster_dist[label]:
                    cluster_dist[label] = dist

            # 按距离排序并取前2个簇
            sorted_clusters = sorted(cluster_dist.items(), key=lambda x: x[1])[:2]
            dist_arr = [dist for _, dist in sorted_clusters]
            labels_arr = [label for label, _ in sorted_clusters]

            # 确保有2个距离值（不足时用大数填充）
            while len(dist_arr) < 2:
                dist_arr.append(1e12)
                labels_arr.append(-1)  # 无效标签

            # 提取最近簇的距离和标签
            first_distance = dist_arr[0]
            second_distance = dist_arr[1]
            first_label = labels_arr[0]

            # 饱和度权重计算
            sat_ratio = clusters_saturation_ratio[first_label] if first_label >= 0 else 1.0
            weight_1 = (-(sat_ratio + 0.1) ** 10 - (sat_ratio + 0.1) + 8) ** 3

            # 距离权重计算
            ratio = second_distance / np.maximum(first_distance, 1e-12)  # 防止除零
            weight_2 = ratio ** (1 / 3)

            # 点密度权重，先保留备用
            # weight_3 = unallocated_points_density_weight[i]

            # 计算总权重
            weights[i] = weight_1 * weight_2

        return weights

    def clustering(self, init_centroids: np.ndarray, step: int = 10, max_iter: int = 100):
        """ 聚类计算 """
        checkpoint_step = 0
        checkpoint_labels = None
        checkpoint_centroids = None
        centroids = init_centroids.copy()
        points = self.points

        for i in range(max_iter):
            self.current_step = i

            t9 = time.perf_counter()
            # 分配每个数据点到最近的簇
            labels, centroids1 = self.assign_clusters(centroids)
            self.timer['t9'] += time.perf_counter() - t9

            t10 = time.perf_counter()
            # 计算新的簇中心
            centroids2 = ClusteringAlgorithm.update_centroids(points, labels)
            self.timer['t10'] += time.perf_counter() - t10

            centroids = centroids2

            t12 = time.perf_counter()
            available, savable = self.save_checkpoint(labels)
            self.timer['t12'] += time.perf_counter() - t12
            if available:
                checkpoint_step += 1
            if savable:
                checkpoint_step = 0
                checkpoint_labels = labels
                checkpoint_centroids = centroids1

            now = datetime.now().strftime("%H:%M:%S")
            click.echo(f'{i}, checkpoint_step:{checkpoint_step}, 路线数量：{len(centroids)} - {now}')

            if checkpoint_step >= step:
                click.echo("=" * 40)
                click.echo("方法累计用时统计:")
                for k, v in self.timer.items():
                    click.echo(f"{k:>10}: {v:.4f}秒")
                click.echo("=" * 40)
                return checkpoint_labels, checkpoint_centroids

        click.echo("=" * 40)
        click.echo("方法累计用时统计:")
        for k, v in self.timer.items():
            click.echo(f"{k:>10}: {v:.4f}秒")
        click.echo("=" * 40)

        if checkpoint_labels is None:
            raise Exception('无法找到合适的路线')

        return checkpoint_labels, checkpoint_centroids
