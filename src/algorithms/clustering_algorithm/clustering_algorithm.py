import click
import numpy as np
from datetime import datetime
import time
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
from sklearn.neighbors import KDTree

from src.services.direct_router.direct_router_rpc_client import DirectRouterRpcClient
from src.services.geo_indexer.geo_indexer import GeoIndexer
from src.utils.utils import random_choice, calc_weighted_centroid, find_nearest_points, max_distance


class ClusteringAlgorithm(ABC):

    def __init__(self, points: np.ndarray, split_cluster_threshold: float):
        self.current_step = 0
        self.points = points
        self.split_cluster_threshold = split_cluster_threshold

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
    def save_checkpoint(self, labels: np.ndarray, progress: float) -> Tuple[bool, bool]:
        pass

    @abstractmethod
    def saturation_ratio(self, labels: np.ndarray, update_labels: np.ndarray, count_clusters: int):
        pass

    def assign_clusters(self, centroids: np.ndarray, can_add_centroid=True) \
            -> Tuple[np.ndarray, np.ndarray]:
        """ 分配每个数据点到最近的簇 """
        points = self.points
        points_index = np.arange(points.shape[0])  # 点的索引
        labels = np.full(points.shape[0], -1)  # 初始化labels为-1，表示未分配
        count_clusters = centroids.shape[0]  # 簇的数量
        clusters_saturation_ratio = np.full(count_clusters, 0.01)  # 簇的饱和度

        # 将距离质心最近的几个点，直接分配给该簇
        n = min(int(points.shape[0] / count_clusters * 0.2), 10)
        centroid_distances = np.linalg.norm(centroids[:, np.newaxis] - points, axis=2)
        centroid_nearest = np.argsort(centroid_distances, axis=1)
        for i in range(count_clusters):
            labels[centroid_nearest[i][:n]] = i

        while True:
            unallocated_indices = points_index[labels == -1]
            if len(unallocated_indices) == 0:
                break

            # 计算未分配点的权重
            weights = ClusteringAlgorithm.clac_unallocated_weight(points, labels,
                                                                  clusters_saturation_ratio)

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

            # 计算饱和度
            update_labels = np.unique(allocate_labels)
            update_saturation_ratio_list = self.saturation_ratio(labels, update_labels, count_clusters)
            for allocate_label, saturation_ratio in zip(update_labels, update_saturation_ratio_list):
                clusters_saturation_ratio[allocate_label] = saturation_ratio

        # 改变簇数量
        if can_add_centroid:
            # 通过调整阈值，在迭代多轮还不收敛时，放宽簇数量限制
            cluster_threshold = 1 if self.current_step <= 10 else \
                1.25 if self.current_step <= 20 else 1.5
            # 计算饱和度总和，乘以0.95是因为当前状态还有优化空间，理想状态下会略低于当前的总饱和度。
            saturation_sum = np.sum(clusters_saturation_ratio) * 0.95
            if saturation_sum > count_clusters + cluster_threshold:
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

                if add_points:  # 只有当成功添加了新点时才进行递归调用
                    centroids = np.vstack([centroids, np.array(add_points)])
                    return self.assign_clusters(centroids, can_add_centroid=False)
            elif saturation_sum < count_clusters - cluster_threshold:
                remove_count = round(count_clusters - saturation_sum)
                # 至少保留两个质心
                remove_count = min(remove_count, count_clusters - 2)
                if remove_count > 0:
                    # 饱和度最低的簇中删除质心
                    remove_labels = np.argsort(clusters_saturation_ratio)[:remove_count]
                    centroids = np.delete(centroids, remove_labels, axis=0)
                    return self.assign_clusters(centroids, can_add_centroid=False)
            elif np.max(clusters_saturation_ratio) > self.split_cluster_threshold:
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

                if new_centroid is not None:
                    centroids = np.vstack([centroids, [new_centroid]])
                    # TODO: 优化递归逻辑
                    return self.assign_clusters(centroids, can_add_centroid=False)

        return labels, centroids

    @staticmethod
    def clac_unallocated_weight(points, labels, clusters_saturation_ratio):
        unallocated_mask = (labels == -1)  # 未分配的点
        unallocated_points = points[unallocated_mask]  # 未分配的点坐标
        weights = np.zeros(unallocated_points.shape[0])  # 初始化权重

        # 获取已分配点及其标签
        allocated_mask = (labels != -1)
        allocated_points = points[allocated_mask]
        allocated_labels = labels[allocated_mask]

        # 对已分配点，根据经纬度去重
        unique_coord_indices = np.unique(allocated_points, axis=0, return_index=True)[1]
        allocated_points = allocated_points[unique_coord_indices]
        allocated_labels = allocated_labels[unique_coord_indices]

        # 如果没有已分配点，直接返回零权重
        if len(allocated_points) == 0:
            return weights

        unique_labels, counts = np.unique(allocated_labels, return_counts=True)
        if len(unique_labels) < 2:
            raise Exception("已分配点的簇数量少于2，无法计算权重")
        # 计算最大簇的点数n
        max_cluster_size = np.max(counts)
        k = min(max_cluster_size + 1, len(allocated_points))  # 查询点数上限

        # 构建KDTree加速最近邻搜索
        kdtree = KDTree(allocated_points, leaf_size=10)

        # 批量查询所有未分配点的最近邻
        distances, indices = kdtree.query(unallocated_points, k=k)

        direct_router_client = DirectRouterRpcClient()
        geo_indexer = GeoIndexer()

        # 处理每个未分配点
        for i in range(unallocated_points.shape[0]):  # i 是未分配点的索引
            current_point = unallocated_points[i]
            cluster_dist = {}  # {label: distance}
            cluster_point = {}  # {label: point}

            # 遍历邻居点，记录不同簇的最短距离
            for j in range(k):  # j 排序邻居点的索引
                label = allocated_labels[indices[i, j]]  # 获取邻居点的标签
                dist = distances[i, j]

                # 如果该簇尚未记录或找到更近距离，更新距离
                if label not in cluster_dist or dist < cluster_dist[label]:
                    cluster_dist[label] = dist
                    cluster_point[label] = allocated_points[indices[i, j]]  # 记录最近邻点

            # 按距离排序并取前2个簇
            sorted_clusters = sorted(cluster_dist.items(), key=lambda x: x[1])[:2]
            dist_arr = [dist for _, dist in sorted_clusters]
            labels_arr = [label for label, _ in sorted_clusters]

            # 确保有2个距离值（不足时用大数填充）
            while len(dist_arr) < 2:
                raise Exception("未分配点的最近簇不足2个，请查询逻辑是否正确")

            first_distance = dist_arr[0]
            second_distance = dist_arr[1]
            first_label = labels_arr[0]
            second_label = labels_arr[1]
            first_point = cluster_point[first_label]
            second_point = cluster_point[second_label]
            current_id = geo_indexer.get_nearest_node_id((current_point[0], current_point[1]))
            first_id = geo_indexer.get_nearest_node_id((first_point[0], first_point[1]))
            second_id = geo_indexer.get_nearest_node_id((second_point[0], second_point[1]))
            duration_arr = direct_router_client.batch_get_path_duration_from_cache(
                [(current_id, first_id), (current_id, second_id)]
            )
            first_duration = duration_arr[0] or 24 * 3600  # 如果没有缓存，使用24小时作为默认值
            second_duration = duration_arr[1] or 24 * 3600  # 如果没有缓存，使用24小时作为默认值

            # 计算最近点到附近已分配点的平均距离
            d, neighbor_indices = kdtree.query([first_point], k=3)  # 查询最近的3个点（包含自身）
            first_to_neighbors_distances = d[0][1:]  # 距离第一个点到其邻近点的距离
            first_to_neighbors_distances = first_to_neighbors_distances[
                ~np.isnan(first_to_neighbors_distances)]  # 去除NaN值
            first_to_neighbors_distance = np.mean(first_to_neighbors_distances)  # 计算平均距离
            # 计算最近点到附近已分配点的平均行驶时间
            nearest_neighbors = allocated_points[neighbor_indices[0][1:]]  # 获取除自身外最近的2个邻居点坐标
            # 将坐标转换为节点ID并构建查询列表
            first_node_id = geo_indexer.get_nearest_node_id((first_point[0], first_point[1]))
            duration_queries = []
            for neighbor_point in nearest_neighbors:
                neighbor_id = geo_indexer.get_nearest_node_id((neighbor_point[0], neighbor_point[1]))
                duration_queries.append((first_node_id, neighbor_id))
            # 批量查询路线行驶时间
            neighbor_durations = direct_router_client.batch_get_path_duration_from_cache(duration_queries)
            valid_durations = [duration for duration in neighbor_durations if duration is not None]
            first_to_neighbors_duration = np.mean(valid_durations) if valid_durations else 600  # 默认10分钟

            # 饱和度权重计算；当前簇饱和度越低，权重越高
            sat_ratio = clusters_saturation_ratio[first_label] if first_label >= 0 else 1.0
            weight_1 = (sat_ratio or 1e-12) ** -7

            # 距离权重计算；靠近最近的簇且远离第二近的簇，权重越高；目的是找到紧挨簇边缘的点
            duration_ratio = second_duration / np.maximum(first_duration, 1e-12)  # 防止除零
            weight_2_1 = duration_ratio if duration_ratio < 10 else 10

            distance_ratio = second_distance / np.maximum(first_distance, 1e-12)  # 防止除零
            weight_2_2 = distance_ratio if distance_ratio < 10 else 10

            weight_2 = weight_2_1 * weight_2_2

            # 跨度权重计算
            # 当前跨度：当前待分配点到最近点已分配点的距离
            # 边界处的跨度：最近点已分配点到其邻近点的平均距离
            # 跨度权重 = 当前跨度 / 边界处的跨度；当前跨度越大，权重越小;
            span_ratio = first_duration / np.maximum(first_to_neighbors_duration, 1e-12)  # 防止除零
            weight_3_1 = 1 if span_ratio < 2 else (span_ratio - 1) ** -0.4

            span_ratio = first_distance / np.maximum(first_to_neighbors_distance, 1e-12)  # 防止除零
            weight_3_2 = 1 if span_ratio < 2 else (span_ratio - 1) ** -1.0

            weight_3 = weight_3_1 * weight_3_2

            # 计算总权重
            weights[i] = weight_1 * weight_2 * weight_3

        return weights

    def clustering(self, init_centroids: np.ndarray, step: int = 10, max_iter: int = 100, zone: int = 1):
        """ 聚类计算 """
        checkpoint_step = 0
        checkpoint_labels = None
        checkpoint_centroids = None
        centroids = init_centroids.copy()
        points = self.points

        for i in range(max_iter):
            self.current_step = i

            # 分配每个数据点到最近的簇
            labels, centroids1 = self.assign_clusters(centroids)

            # 计算新的簇中心
            centroids2 = ClusteringAlgorithm.update_centroids(points, labels)

            centroids = centroids2

            available, savable = self.save_checkpoint(labels, i / max_iter)
            if savable:
                checkpoint_step = 0
                checkpoint_labels = labels
                checkpoint_centroids = centroids1
            else:
                if checkpoint_labels is not None:
                    checkpoint_step += 1

            now = datetime.now().strftime("%H:%M:%S")
            click.echo(f'{zone}-{i}, checkpoint_step:{checkpoint_step}, 路线数量：{len(centroids)} - {now}')

            if checkpoint_step >= step:
                return checkpoint_labels, checkpoint_centroids

        if checkpoint_labels is None:
            raise Exception('无法找到合适的路线')

        return checkpoint_labels, checkpoint_centroids
