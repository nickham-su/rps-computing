from typing import Tuple
from sklearn.neighbors import KDTree

from src.data.data_loader import get_all_nodes


class GeoIndexer:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.nearest_node_cache = {}
            node_df = get_all_nodes()
            cls._instance.node_ids = node_df['node_id'].copy()
            cls._instance.kd_tree = KDTree(node_df[['lat', 'lon']].values, leaf_size=30)
        return cls._instance

    def get_nearest_node_id(self, point: Tuple[float, float]) -> int:
        """ 获取最近节点ID """
        if self.kd_tree is None:
            raise RuntimeError("Geo index not initialized")
        key = (point[0], point[1])
        if cached := self.nearest_node_cache.get(key):
            return cached
        _, ind = self.kd_tree.query([point], k=1)
        nearest_node_id = int(self.node_ids.iloc[ind[0][0]])
        self.nearest_node_cache[key] = nearest_node_id
        return nearest_node_id
