import math
from typing import Tuple
import numpy as np
from src.algorithms.clustering_algorithm.clustering_algorithm import ClusteringAlgorithm


class SizeLimitedClusterer(ClusteringAlgorithm):
    def __init__(self, points: np.ndarray, max_size: int):
        super().__init__(points, 10)
        self.max_size = max_size
        self.min_std = math.inf

    def save_checkpoint(self, labels: np.ndarray) -> Tuple[bool, bool]:
        size_list = [len(self.points[labels == label]) for label in np.unique(labels)]
        size_std = np.std(size_list)
        if size_std < self.min_std:
            self.min_std = size_std
            return True, True
        return True, False

    def saturation_ratio(self, labels: np.ndarray, update_labels: np.ndarray, count_clusters: int):
        results = []
        for cluster_label in update_labels:
            count_points = np.sum(labels == cluster_label)
            saturation_ratio = count_points / self.max_size
            results.append(saturation_ratio)
        return results
