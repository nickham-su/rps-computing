import click
import numpy as np
from datetime import datetime
from typing import Tuple
from abc import ABC, abstractmethod
from src.services.direct_router.direct_router import DirectRouter
from src.services.geo_indexer.geo_indexer import GeoIndexer
from src.utils.utils import find_nearest_point, random_choice, calc_weighted_centroid, find_nearest_points, max_distance


class ClusteringAlgorithm(ABC):

    def __init__(self, points: np.ndarray):
        self.current_step = 0
        self.points = points

    @staticmethod
    def preprocessing_route(points: np.ndarray, centroids: np.ndarray, limit: int):
        geo_indexer = GeoIndexer()
        """ 预处理点到簇的路线 """
        point_pairs_list = []
        for i in range(len(points)):
            distances = np.linalg.norm(points[i] - centroids, axis=1)
            nearest = np.argsort(distances)
            lat, lon = points[i]
            point_id = geo_indexer.get_nearest_node_id((lat, lon))
            for j in nearest[:limit]:
                lat, lon = centroids[j]
                centroid_id = geo_indexer.get_nearest_node_id((lat, lon))
                point_pairs_list.append((point_id, centroid_id))
        DirectRouter().batch_calc_path_duration(point_pairs_list)

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
    def saturation_ratio(self, cluster_points: np.ndarray, cluster_label: int, count_clusters: int):
        pass

    def assign_clusters(self, centroids: np.ndarray, can_add_centroid=True) \
            -> Tuple[np.ndarray, np.ndarray]:
        """ 分配每个数据点到最近的簇 """
        points = self.points
        points_index = np.arange(points.shape[0])
        labels = np.full(points.shape[0], -1)
        limit = 3  # 只尝试最近几个簇是否可以添加
        count_clusters = centroids.shape[0]
        clusters_saturation_ratio = np.zeros(count_clusters)

        neighbor_radius = np.zeros(points.shape[0])
        neighbor_number = round(points.shape[0] / count_clusters / 3)
        for i in range(points.shape[0]):
            lat, lon = points[i]
            neighbor_points, _ = find_nearest_points((lat, lon), points, neighbor_number)
            center = np.mean(neighbor_points, axis=0)
            radius, _ = max_distance(center, neighbor_points)
            neighbor_radius[i] = radius
        points_density_weight = (neighbor_radius / np.mean(neighbor_radius)) ** (1 / 2)

        # 预处理点到簇的路线
        ClusteringAlgorithm.preprocessing_route(points, centroids, limit)

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
                                                                  clusters_saturation_ratio, points_density_weight)
            # 根据饱和度确定批量处理的数量
            max_saturation_ratio = np.max(clusters_saturation_ratio)
            allocate_number = 0.5 * count_clusters if max_saturation_ratio > 0.9 else \
                count_clusters if max_saturation_ratio > 0.8 else \
                    2 * count_clusters if max_saturation_ratio > 0.7 else \
                        3 * count_clusters if max_saturation_ratio > 0.4 else 5 * count_clusters

            # 分配
            allocated_points = points[labels != -1]
            allocated_labels = labels[labels != -1]
            allocate_labels = []
            for i in np.argsort(weights)[::-1][:max(round(allocate_number), 1)]:
                current_index = unallocated_indices[i]
                current_coord = points[current_index]
                curr_to_allocated_distances = np.linalg.norm(current_coord - allocated_points, axis=1)
                nearest_index = np.argmin(curr_to_allocated_distances)
                current_label = allocated_labels[nearest_index]
                labels[current_index] = current_label
                allocate_labels.append(current_label)

            # 计算饱和度
            for allocate_label in np.unique(allocate_labels):
                # 计算饱和度
                cluster_points = points[labels == allocate_label]
                clusters_saturation_ratio[allocate_label] = self.saturation_ratio(
                    cluster_points, int(allocate_label), count_clusters)

        # 改变簇数量
        if can_add_centroid:
            saturation_sum = np.sum(clusters_saturation_ratio)
            if saturation_sum > count_clusters + 0.5:
                add_count = round(saturation_sum - count_clusters)
                # 饱和度最高的簇中添加质心
                allocate_labels = np.argsort(clusters_saturation_ratio)[::-1][:add_count]
                add_points = []
                for l in allocate_labels:
                    if np.sum(labels == l) > 0:
                        add_points.append(random_choice(points[labels == l], 1)[0])
                centroids = np.vstack([centroids, np.array(add_points)])
                return self.assign_clusters(centroids, can_add_centroid=False)
            elif saturation_sum < count_clusters - 0.5:
                remove_count = round(count_clusters - saturation_sum)
                # 饱和度最低的簇中删除质心
                remove_labels = np.argsort(clusters_saturation_ratio)[:remove_count]
                centroids = np.delete(centroids, remove_labels, axis=0)
                return self.assign_clusters(centroids, can_add_centroid=False)
            elif np.max(clusters_saturation_ratio) > 1.4:
                # 在饱和度最高的簇，添加一个质心，将其拆分
                max_label = np.argmax(clusters_saturation_ratio)
                cluster_points = points[labels == max_label]
                centroids = np.vstack([centroids, random_choice(cluster_points, 1)])
                return self.assign_clusters(centroids, can_add_centroid=False)

        return labels, centroids

    @staticmethod
    def clac_unallocated_weight(points, labels, clusters_saturation_ratio, points_density_weight):
        unallocated_mask = (labels == -1)
        unallocated_points = points[unallocated_mask]
        unallocated_points_density_weight = points_density_weight[unallocated_mask]
        weights = np.zeros(unallocated_points.shape[0])

        # 预计算所有簇中心
        max_label = np.max(labels)
        centroids = np.zeros((max_label + 1, 2))
        for l in range(max_label + 1):
            cluster_points = points[labels == l]
            if len(cluster_points) > 0:
                centroids[l] = np.mean(cluster_points, axis=0)
            else:
                centroids[l] = np.array([0.0, 0.0])

        for i in range(unallocated_points.shape[0]):
            point = unallocated_points[i]
            # 计算到所有簇中心的距离
            point_to_centroids = np.linalg.norm(point - centroids, axis=1)
            # 获取最近的3个簇
            candidate_labels = np.argsort(point_to_centroids)[:3]
            c1_points = points[labels == candidate_labels[0]]
            c2_points = points[labels == candidate_labels[1]]
            c3_points = points[labels == candidate_labels[2]]
            # 计算到候选点的距离
            c1_distance = np.min(np.linalg.norm(point - c1_points, axis=1))
            c2_distance = np.min(np.linalg.norm(point - c2_points, axis=1))
            c3_distance = np.min(np.linalg.norm(point - c3_points, axis=1))

            first_distance = min(c1_distance, c2_distance, c3_distance)
            second_distance = sorted([c1_distance, c2_distance, c3_distance])[1]
            first_label = candidate_labels[np.argmin([c1_distance, c2_distance, c3_distance])]

            # 计算权重
            sat_ratio = clusters_saturation_ratio[first_label]
            weight_1 = (-(sat_ratio + 0.1) ** 10 - (sat_ratio + 0.1) + 8) ** 3

            ratio = second_distance / np.maximum(first_distance, 1e-12)
            weight_2 = ratio ** (1 / 3)

            weight_3 = unallocated_points_density_weight[i]

            weights[i] = weight_1 * weight_2 * weight_3
        return weights

    def filter_centroids(self, centroids: np.ndarray, labels: np.ndarray):
        """ 过滤质心 """
        label_arr = np.unique(labels)
        saturation_ratio_arr = np.array([self.saturation_ratio(self.points[labels == l], int(l), len(centroids))
                                         for l in label_arr])

        if np.sum(saturation_ratio_arr < 0.6) > 0:
            min_index = np.argmin(saturation_ratio_arr)
            return np.delete(centroids, min_index, axis=0)
        return centroids

    def clustering(self, init_centroids: np.ndarray, step: int = 10, max_iter: int = 100, print_step: bool = False):
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
            # 过滤质心
            centroids3 = self.filter_centroids(centroids2, labels)
            centroids = centroids3

            available, savable = self.save_checkpoint(labels)
            if available:
                checkpoint_step += 1
            if savable:
                checkpoint_step = 0
                checkpoint_labels = labels
                checkpoint_centroids = centroids1

            if print_step:
                now = datetime.now().strftime("%H:%M:%S")
                click.echo(f'{i}, checkpoint_step:{checkpoint_step}, 路线数量：{len(centroids)} - {now}')

            if checkpoint_step >= step:
                return checkpoint_labels, checkpoint_centroids

        if checkpoint_labels is None:
            raise Exception('无法找到合适的路线')
        return checkpoint_labels, checkpoint_centroids
