import click
import numpy as np
from haversine import haversine


class ResultOptimizer:

    def __init__(self, points: np.ndarray, labels: np.ndarray):
        self.points = points
        self.labels = labels

    def optimize(self):
        unique_labels = np.unique(self.labels)
        centroids = np.zeros((np.max(unique_labels) + 1, self.points.shape[1]))
        for label in unique_labels:
            centroids[label] = np.mean(self.points[self.labels == label], axis=0)

        new_labels = self.labels.copy()
        # 计算每个点到其所属质心的距离
        distances_to_own_centroid = np.array([haversine(self.points[i], centroids[self.labels[i]])
                                              for i in range(self.points.shape[0])])
        # 按距离从远到近排序
        sorted_indices = np.argsort(distances_to_own_centroid)[::-1]
        for i in sorted_indices:
            # 当前所在label和到质心的距离
            current_label = self.labels[i]
            current_distance = distances_to_own_centroid[i]
            # 计算点到所有质心的距离，并按照从近到远排序
            distances = np.array([haversine(self.points[i], c) for c in centroids])
            sorted_centroid_distances = np.argsort(distances)
            # 最近的label和到最近质心的距离
            nearest_label = sorted_centroid_distances[0]
            nearest_distance = distances[nearest_label]
            # 如果当前所在簇不是最近的簇，且根据下面规则判断是否为异常点，修改归属。
            if current_label != nearest_label:
                current_points = self.labels == current_label
                current_points[i] = False
                current_border = np.percentile(distances_to_own_centroid[current_points], 90)
                nearest_border = np.percentile(distances_to_own_centroid[self.labels == nearest_label], 90)
                distance1 = current_distance - current_border
                distance2 = nearest_distance - nearest_border
                if distance1 > distance2 * 2 and current_distance > current_border * 1.4:
                    new_labels[i] = nearest_label
        click.echo(f'优化完成，共调整{np.sum(new_labels != self.labels)}个点')
        for i in np.arange(self.labels.shape[0])[new_labels != self.labels]:
            click.echo(f'{self.labels[i]}->{new_labels[i]}')
        return new_labels
