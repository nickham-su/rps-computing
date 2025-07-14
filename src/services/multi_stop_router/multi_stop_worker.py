import click
import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional
from itertools import pairwise

from src.services.direct_router.direct_router_rpc_client import DirectRouterRpcClient
from src.services.geo_indexer.geo_indexer import GeoIndexer


class MultiStopRouterWorker:
    # 添加类属性保存共享的DirectRouterProxy实例

    @staticmethod
    def calc_route_duration(waypoints: np.ndarray, start_coord: Optional[Tuple[float, float]] = None) \
            -> Tuple[float, List[int]]:
        """ 计算路径用时 """
        if len(waypoints) + (1 if start_coord else 0) < 2:
            return 0.0, []
        route_node_ids, indexes = MultiStopRouterWorker._routing(waypoints, start_coord)
        duration = MultiStopRouterWorker._get_route_duration(route_node_ids, start_coord)
        return duration, indexes

    @staticmethod
    def _get_route_duration(route_node_ids: List[int], start_coord: Optional[Tuple[float, float]] = None):
        """ 获取路径用时 """
        if start_coord is not None:
            start_id = GeoIndexer().get_nearest_node_id(start_coord)
            ids = [start_id, *route_node_ids]
        else:
            ids = route_node_ids
        if len(ids) < 2:
            return 0.0
        point_pairs_list = list(pairwise(map(int, ids)))
        duration_list = DirectRouterRpcClient().batch_calc_path_duration(point_pairs_list)
        return sum(duration_list)

    @staticmethod
    def _routing(waypoints: np.ndarray, start_coord: Optional[Tuple[float, float]] = None) \
            -> Tuple[List[int], List[int]]:
        """ 计算多点路径 """
        direct_router_rpc_client = DirectRouterRpcClient()
        geo_indexer = GeoIndexer()

        if len(waypoints) == 0:
            return [], []
        if len(waypoints) == 1:
            # 如果只有一个点，则直接返回该点的索引和节点ID
            lat, lon = waypoints[0]
            node_id = geo_indexer.get_nearest_node_id((lat, lon))
            return [node_id], [0]

        id_to_index: Dict[int, List[int]] = {}
        for i in range(len(waypoints)):
            lat, lon = waypoints[i]
            node_id = geo_indexer.get_nearest_node_id((lat, lon))
            if node_id in id_to_index:
                id_to_index[node_id].append(i)
            else:
                id_to_index[node_id] = [i]

        node_ids = list(id_to_index.keys())

        if len(node_ids) == 1:  # 不可能为0，因为至少有一个点；为1的原因是所有点都在同一个节点上，想到于他们直接没有距离，任意排序即可。
            return list(id_to_index.keys()), list(range(len(waypoints)))
        elif len(node_ids) == 2:
            # 如果只有两个节点，则直接返回这两个节点的索引和ID
            route_indexes = []
            for node_id in node_ids:
                route_indexes.extend(id_to_index[node_id])
            return node_ids, list(map(int, np.argsort(route_indexes)))

        start_id = None
        if start_coord is not None:
            # 如果提供了起点坐标，则获取起点的节点ID
            start_id = geo_indexer.get_nearest_node_id(start_coord)
            if start_id not in node_ids:
                # 如果起点节点ID不在waypoints中，则添加到node_ids中
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
                point_pairs_list.append((int(p1), int(p2)))
                added_set.add(key)

        direct_router_rpc_client.batch_calc_path_duration(point_pairs_list)

        # 构建图
        g = nx.Graph()
        for i in range(count_nodes - 1):
            p1 = node_ids[i]
            for j in range(i + 1, count_nodes):
                p2 = node_ids[j]
                duration = direct_router_rpc_client.get_path_duration_from_cache(p1, p2)
                g.add_edge(p1, p2, weight=duration)

        # 规划线路，path是一个节点ID的列表，且线路是一个闭合的环，即起点和终点是同一个节点
        path = np.array(nx.approximation.christofides(g, weight='weight'))
        # 清除缓存
        del g

        if start_id:
            # 如果有起点坐标，则需要将路径调整为从起点开始
            index = np.where(path == start_id)[0][0]
            sorted_path = np.concatenate((path[index:], path[1:index]))
        else:
            # 如果没有起点坐标，则直接使用计算出的路径
            sorted_path = path[:-1]  # 去掉最后一个节点，因为它是起点的重复
        route_node_ids = []
        route_indexes = []
        for node_id in sorted_path:
            if node_id not in id_to_index:
                continue  # 跳过起点
            route_node_ids.append(node_id)
            route_indexes.extend(id_to_index[node_id])

        if len(route_indexes) != len(waypoints):
            raise ValueError('路径规划出错')

        # 按照waypoints的顺序返回每个点的序号
        return route_node_ids, list(map(int, np.argsort(route_indexes)))
