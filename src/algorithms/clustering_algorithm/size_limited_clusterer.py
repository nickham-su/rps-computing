import math
from typing import Tuple
import numpy as np
from src.algorithms.clustering_algorithm.clustering_algorithm import ClusteringAlgorithm


class SizeLimitedClusterer(ClusteringAlgorithm):
    def __init__(self, points: np.ndarray, max_size: int):
        super().__init__(points)
        self.max_size = max_size
        self.min_std = math.inf

    def save_checkpoint(self, labels: np.ndarray) -> Tuple[bool, bool]:
        size_list = [len(self.points[labels == label]) for label in np.unique(labels)]
        size_std = np.std(size_list)
        if size_std < self.min_std:
            self.min_std = size_std
            return True, True
        return True, False

    def saturation_ratio(self, cluster_points: np.ndarray, cluster_label: int, count_clusters: int):
        return len(cluster_points) / self.max_size

    def calculate_mean_intra_cluster_distance(self, labels: np.ndarray):
        num_clusters = np.max(labels) + 1
        mean_distances = np.zeros(num_clusters)
        for label in range(num_clusters):
            cluster_points = self.points[labels == label]
            num_points = len(cluster_points)
            if num_points < 2:
                mean_distances[label] = 0  # 如果簇中少于两个点，平均距离为0
            else:
                distances_sum = np.sum([
                    np.linalg.norm(point1 - point2)
                    for i, point1 in enumerate(cluster_points)
                    for point2 in cluster_points[i + 1:]
                ])
                num_distances = num_points * (num_points - 1) / 2
                mean_distances[label] = distances_sum / num_distances
        return np.mean(mean_distances)
