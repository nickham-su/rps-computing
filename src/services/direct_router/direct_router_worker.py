from typing import Tuple


class DirectRouterWorker:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            import networkx as nx
            from haversine import haversine
            from src.data.data_loader import get_map_data, get_node_coord
            g = nx.DiGraph()
            map_df = get_map_data()
            g.add_edges_from((row["start"], row["end"], {'weight': row["duration"]}) for _, row in map_df.iterrows())

            def heuristic(node_id1: int, node_id2: int) -> float:
                coord1 = get_node_coord(node_id1)
                coord2 = get_node_coord(node_id2)
                return haversine(coord1, coord2) / 60 * 3600

            def _find_path_duration(start_id: int, end_id: int) -> float:
                return float(nx.astar_path_length(g, start_id, end_id, weight='weight', heuristic=heuristic))

            cls._instance._calc_path_duration = _find_path_duration

        return cls._instance

    def calc_path_duration(self, point_pairs: Tuple[int, int]) -> float:
        """ 查询最短路径用时 """
        start_id, end_id = point_pairs
        return self._calc_path_duration(start_id, end_id)
