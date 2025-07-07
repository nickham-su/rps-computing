import os
from typing import Optional, Dict, Tuple, List
import click
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import networkx as nx
from haversine import haversine, Unit

map_data: Optional[pd.DataFrame] = None  # 地图数据
node_data: Optional[pd.DataFrame] = None  # 节点数据
coord_cache: Optional[Dict[int, Tuple[float, float]]] = None  # 节点坐标缓存

# 道路类型的速度
way_type_to_speed = {
    "motorway": 122,
    "motorway_link": 90,
    "trunk": 100,
    "trunk_link": 60,
    "primary": 58,
    "primary_link": 45,
    "secondary": 52,
    "secondary_link": 35,
    "tertiary": 51,
    "tertiary_link": 33,
    "unclassified": 40,
    "residential": 42,
}
cache_dir = './cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


def get_bbox(warehouse_coords: List[Tuple[float, float]], points: np.ndarray) -> Tuple[float, float, float, float]:
    # 计算所有坐标的边界(订单点 + 所有仓库)
    all_coords = np.vstack([points, np.array(warehouse_coords)])
    min_latitude = np.min(all_coords[:, 0])
    max_latitude = np.max(all_coords[:, 0])
    min_longitude = np.min(all_coords[:, 1])
    max_longitude = np.max(all_coords[:, 1])

    # 计算范围的宽度和高度（公里）
    # 纬度范围（南北方向）
    lat_range_km = (max_latitude - min_latitude) * 111.32
    # 经度范围（东西方向）- 使用中心纬度来计算
    center_latitude = (min_latitude + max_latitude) / 2
    lon_range_km = (max_longitude - min_longitude) * (111.32 * np.cos(np.radians(center_latitude)))
    click.echo(f"范围: 纬度 {min_latitude} ~ {max_latitude}, 经度 {min_longitude} ~ {max_longitude}")
    click.echo(f"范围: 纬度范围 {lat_range_km:.2f}km, 经度范围 {lon_range_km:.2f}km")
    # 检查是否超过300km
    if lat_range_km > 300 or lon_range_km > 300:
        raise ValueError("范围超过300km，请检查数据")

    # 向四周扩大范围
    expansion_distance_km = 10
    # 纬度方向上的扩展（纬度每度约等于111.32km）
    lat_expansion = expansion_distance_km / 111.32
    expanded_min_latitude = min_latitude - lat_expansion
    expanded_max_latitude = max_latitude + lat_expansion
    # 经度方向上的扩展（与纬度有关）
    # 使用中心纬度来计算经度扩展
    center_latitude = (min_latitude + max_latitude) / 2
    # 经度每度对应的距离约等于 111.32 * cos(latitude)km
    lon_expansion = expansion_distance_km / (111.32 * np.cos(np.radians(center_latitude)))
    expanded_min_longitude = min_longitude - lon_expansion
    expanded_max_longitude = max_longitude + lon_expansion

    # 定义边界框
    return (expanded_min_longitude, expanded_min_latitude,
            expanded_max_longitude, expanded_max_latitude)


def _load_map_from_gpkg(bbox: Tuple[float, float, float, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 取环境变量
    map_data_dir = os.getenv('MAP_DATA_DIR', 'map_data')
    map_data_dir = map_data_dir.rstrip('/')
    road_network_path = f'{map_data_dir}/road_network.gpkg'
    if not os.path.exists(road_network_path):
        raise FileNotFoundError(f"地图数据文件 {road_network_path} 不存在")

    # 直接使用查询读取数据
    roads = gpd.read_file(road_network_path, layer='lines', bbox=box(*bbox))
    # 提取所有LineString中的坐标点
    all_points = []
    for line in roads.geometry:
        # 对每条线获取所有坐标点
        for coord in line.coords:
            # coord 是 (longitude, latitude) 格式
            # 将经纬度乘以10^7并转换为整数
            lon_int = int(round(coord[0] * 10 ** 7))
            lat_int = int(round(coord[1] * 10 ** 7))
            all_points.append({'lon': lon_int, 'lat': lat_int})

    # 创建DataFrame并去重
    nodes_df = pd.DataFrame(all_points)
    nodes_df_unique = nodes_df.drop_duplicates(subset=['lon', 'lat']).copy()
    # 重建索引，使得 node_id 从 0 开始
    nodes_df_unique.reset_index(drop=True, inplace=True)
    # 将lat、lon作为索引
    nodes_df_unique['node_id'] = nodes_df_unique.index
    nodes_df_indexed = nodes_df_unique.set_index(['lat', 'lon'])

    # 初始化一个有向图
    G = nx.DiGraph()
    # 遍历每一条道路
    for index, road_attributes in roads.iterrows():
        line = road_attributes.geometry  # LineString 对象
        highway_type = road_attributes.highway
        oneway_status = road_attributes.oneway  # 可能为 'yes', 'no', None 等
        junction_type = road_attributes.junction

        node_ids_in_current_road = []
        # 提取当前道路的所有坐标点，并查找对应的 node_id
        for lon, lat in line.coords:
            lon_int = int(round(lon * 10 ** 7))
            lat_int = int(round(lat * 10 ** 7))
            node_id = nodes_df_indexed.loc[(lat_int, lon_int), 'node_id']
            node_ids_in_current_road.append(node_id)

        # 遍历当前道路的节点对 (路段)，以添加边
        for i in range(len(node_ids_in_current_road) - 1):
            u_node = node_ids_in_current_road[i]
            v_node = node_ids_in_current_road[i + 1]
            # 判断是否为单行道
            # 条件1: oneway 属性明确为 'yes'
            is_oneway_explicit = (oneway_status == 'yes')
            # 条件2: oneway 属性为 None (或 NaN) 且 junction 类型为 'roundabout'
            is_oneway_roundabout_implicit = (pd.isna(oneway_status) or oneway_status is None) and \
                                            (junction_type == 'roundabout')

            if is_oneway_explicit or is_oneway_roundabout_implicit:
                # 单向添加边: u -> v
                G.add_edge(u_node, v_node, highway=highway_type)
            else:
                # 双向添加边: u -> v 和 v -> u
                # 这包括 oneway == 'no', oneway 为 None 且非环岛等情况
                G.add_edge(u_node, v_node, highway=highway_type)
                G.add_edge(v_node, u_node, highway=highway_type)

    # 找到所有强连通分量
    strongly_connected_components = list(nx.strongly_connected_components(G))

    if len(strongly_connected_components) == 0:
        raise ValueError("没有找到强连通分量")

    # 找到最大的强连通分量
    largest_component_nodes = max(strongly_connected_components, key=len)

    # 从原图中提取最大强连通子网
    largest_scc_subnet = G.subgraph(largest_component_nodes).copy()  # 使用 .copy() 确保得到一个独立的图副本

    _node_data = nodes_df_unique.loc[list(largest_scc_subnet.nodes())].copy()
    _node_data['lat'] = _node_data['lat'] / 10 ** 7
    _node_data['lon'] = _node_data['lon'] / 10 ** 7
    _node_data['node_id'] = _node_data['node_id'].astype(int)

    # 遍历所有的边
    edges = []
    for u, v, data in largest_scc_subnet.edges(data=True):
        u_row = _node_data.loc[u]
        v_row = _node_data.loc[v]
        start_coord = (u_row['lat'], u_row['lon'])
        end_coord = (v_row['lat'], v_row['lon'])
        distance = haversine(start_coord, end_coord, unit=Unit.METERS)
        speed = way_type_to_speed[data['highway']] / 3600 * 1000  # 转换为米/秒
        edges.append({
            'start': u,  # 起点id
            'end': v,  # 终点id
            'start_lat': start_coord[0],  # 起点纬度
            'start_lon': start_coord[1],  # 起点经度
            'end_lat': end_coord[0],  # 终点纬度
            'end_lon': end_coord[1],  # 终点经度
            'distance': distance,  # 距离（米）
            'duration': distance / speed,  # 用时（秒）
        })
    _map_data = pd.DataFrame(edges)
    _node_data.reset_index(drop=True, inplace=True)
    return _map_data, _node_data


def load_map_data(bbox: Tuple[float, float, float, float], _: Optional[int] = None) -> None:
    global map_data
    global node_data
    global coord_cache
    if map_data is not None:
        return

    map_data_path = f'{cache_dir}/map_data_{round(bbox[0], 3)}_{round(bbox[1], 3)}_{round(bbox[2], 3)}_{round(bbox[3], 3)}.pkl'
    node_data_path = f'{cache_dir}/node_data_{round(bbox[0], 3)}_{round(bbox[1], 3)}_{round(bbox[2], 3)}_{round(bbox[3], 3)}.pkl'
    # 检查缓存
    if os.path.exists(map_data_path) and os.path.exists(node_data_path):
        click.echo("加载pkl缓存数据")
        map_data = pd.read_pickle(map_data_path)
        node_data = pd.read_pickle(node_data_path)
    else:
        # 使用gpkg数据，并保存到缓存
        click.echo("加载gpkg地图数据")

        map_data, node_data = _load_map_from_gpkg(bbox)
        map_data.to_pickle(map_data_path)
        node_data.to_pickle(node_data_path)

    coord_cache = {
        int(row['node_id']): (float(row['lat']), float(row['lon']))
        for _, row in node_data.iterrows()
    }

    click.echo(f"地图数据加载完成，共 {len(map_data)} 条边，{len(node_data)} 个节点")


def get_map_data() -> pd.DataFrame:
    """ 获取地图数据 """
    global map_data
    if map_data is None:
        raise Exception('地图数据未加载')
    return map_data


def get_all_nodes() -> pd.DataFrame:
    """ 获取节点数据 """
    global node_data
    if node_data is None:
        raise Exception('节点数据未加载')
    return node_data


def get_node_coord(node_id: int) -> Tuple[float, float]:
    """ 获取节点坐标 """
    global coord_cache
    if coord_cache is None:
        raise Exception('节点数据未加载')
    return coord_cache[node_id]
