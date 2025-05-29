import numpy as np
from haversine import haversine
from typing import List, Tuple


def random_choice(points: np.ndarray, k: int) -> np.ndarray:
    """
    随机选择k个点
    points: np.ndarray, shape=(n, 2)
    """
    indices = np.random.choice(points.shape[0], k, replace=False)
    return points[indices]


def find_nearest_point(origin: Tuple[float, float], option_points: np.ndarray):
    """
    找到离origin最近的点
    origin: Tuple[float, float]
    option_points: np.ndarray, shape=(n, 2)

    return: Tuple[float, float], int
    """
    distances = np.array([haversine(origin, p) for p in option_points])
    nearest_index = np.argmin(distances)
    return option_points[nearest_index], nearest_index


def find_nearest_points(origin: Tuple[float, float], option_points: np.ndarray, k: int):
    """
    找到离origin最近的k个点
    origin: Tuple[float, float]
    option_points: np.ndarray, shape=(n, 2)
    k: int

    return: List[List[float, float]], List[int]
    """
    distances = np.array([haversine(origin, p) for p in option_points])
    nearest_indices = np.argsort(distances)[:k]
    return option_points[nearest_indices], nearest_indices


def max_distance(origin: Tuple[float, float], option_points: np.ndarray):
    """
    找到离origin最远点的距离
    origin: Tuple[float, float]
    option_points: np.ndarray, shape=(n, 2)

    return: float, int 距离, 索引
    """
    distances = np.array([haversine(origin, p) for p in option_points])
    max_index = np.argmax(distances)
    return distances[max_index], max_index


def min_distance(origin: Tuple[float, float], option_points: np.ndarray):
    """
    找到离origin最近点的距离
    origin: Tuple[float, float]
    option_points: np.ndarray, shape=(n, 2)

    return: float, int 距离, 索引
    """
    distances = np.array([haversine(origin, p) for p in option_points])
    min_index = np.argmin(distances)
    return distances[min_index], min_index


def calc_weighted_centroid(points: np.ndarray, distance_threshold=1.0):
    """
    计算一批点的加权质心，靠近质心的点具有更高权重。
    points: np.ndarray, shape=(n, 2)
    distance_threshold: float

    return: Tuple[float, float]
    """
    # 首先计算初始质心
    initial_centroid = points.mean(axis=0)
    # 计算每个点到初始质心的距离
    distances = np.linalg.norm(points - initial_centroid, axis=1)
    # 通过距离反比计算权重
    # 可以使用一个距离阈值来调整权重曲线，以避免权重过大
    weights = np.exp(-distances / distance_threshold)
    # 计算加权质心
    weighted_sum = np.sum(points.T * weights, axis=1)
    total_weight = np.sum(weights)
    weighted_centroid = weighted_sum / total_weight
    return weighted_centroid
