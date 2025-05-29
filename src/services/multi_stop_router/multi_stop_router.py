import numpy as np
import networkx as nx
from typing import Tuple, List, Optional, Dict
from itertools import pairwise
from src.services.direct_router.direct_router import DirectRouter
from src.services.geo_indexer.geo_indexer import GeoIndexer


class MultiStopRouter:

    @staticmethod
    def calc_route_duration(waypoints: np.ndarray, start_coord: Tuple[float, float]) -> float:
        """ 计算路径用时 """
        _, route_node_ids = MultiStopRouter.routing(waypoints, start_coord)
        return MultiStopRouter.get_route_duration(route_node_ids, start_coord)

    @staticmethod
    def get_route_duration(route_node_ids: List[int], start_coord: Tuple[float, float]):
        start_id = GeoIndexer().get_nearest_node_id(start_coord)
        ids = [start_id] + route_node_ids
        point_pairs_list = list(pairwise(ids))
        duration_list = DirectRouter().batch_calc_path_duration(point_pairs_list)
        return sum(duration_list)

    @staticmethod
    def routing(waypoints: np.ndarray, start_coord: Tuple[float, float]):
        """ 计算多点路径 """
        id_to_index: Dict[int, List[int]] = {}
        for i in range(len(waypoints)):
            lat, lon = waypoints[i]
            node_id = GeoIndexer().get_nearest_node_id((lat, lon))
            if node_id in id_to_index:
                id_to_index[node_id].append(i)
            else:
                id_to_index[node_id] = [i]

        node_ids = list(id_to_index.keys())
        start_id = GeoIndexer().get_nearest_node_id(start_coord)
        if start_id not in node_ids:
            node_ids.append(start_id)

        count_nodes = len(node_ids)
        point_pairs_list = []
        added_set = set()
        for i in range(count_nodes - 1):
            p1 = node_ids[i]
            for j in range(i + 1, count_nodes):
                p2 = node_ids[j]
                key = f"{min(p1, p2)}_{max(p1, p2)}"
                if p1 == p2 or key in added_set:
                    continue
                point_pairs_list.append((p1, p2))
                added_set.add(key)
        DirectRouter().batch_calc_path_duration(point_pairs_list)

        # 构建图
        g = nx.Graph()
        for i in range(count_nodes - 1):
            p1 = node_ids[i]
            for j in range(i + 1, count_nodes):
                p2 = node_ids[j]
                duration = DirectRouter().get_path_duration_from_cache(p1, p2)
                g.add_edge(p1, p2, weight=duration)

        # 规划线路
        path = np.array(nx.approximation.christofides(g, weight='weight'))
        # 清除缓存
        del g

        index = np.where(path == start_id)[0][0]
        sorted_path = np.concatenate((path[index:], path[1:index]))
        route_node_ids = []
        route_indexes = []
        for node_id in sorted_path:
            if node_id not in id_to_index:
                continue  # 跳过起点
            route_node_ids.append(node_id)
            for i in id_to_index[node_id]:
                route_indexes.append(i)

        if len(route_indexes) != len(waypoints):
            raise ValueError('路径规划出错')

        # 按照waypoints的顺序返回每个点的序号
        return np.argsort(route_indexes), route_node_ids
