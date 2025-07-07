import traceback
from typing import Tuple, List

import click
import pandas as pd
import numpy as np
import math
from src.algorithms.clustering_algorithm.duration_limited_clusterer import batch_duration_limited_cluster, \
    DurationLimitedClusterer, duration_limited_cluster
from src.algorithms.clustering_algorithm.size_limited_clusterer import SizeLimitedClusterer
from src.services.direct_router.direct_router_rpc_server import DirectRouterRpcServer
from src.services.multi_stop_router.multi_stop_router_rpc_server import MultiStopRouterRPCServer
from src.utils.utils import random_choice
from src.data.data_loader import load_map_data, get_bbox
from src.algorithms.result_visualizer.result_visualizer import ResultVisualizer
from src.algorithms.data_preprocessor.preprocess_data import preprocess_data


def parse_warehouse_coords(coords_str: str) -> List[Tuple[float, float]]:
    """解析多仓库坐标字符串，格式: 'lat1,lon1 lat2,lon2'"""
    coords = []
    for pair in coords_str.strip().split():
        try:
            lat, lon = map(float, pair.split(','))
            coords.append((lat, lon))
        except ValueError:
            raise click.ClickException(f"坐标格式错误: '{pair}'，应为 'lat,lon'")

    if not coords:
        raise click.ClickException("至少需要提供一个仓库坐标")

    return coords


@click.command()
@click.option(
    '--warehouse-coord',
    type=click.Tuple([float, float]),  # 定义为一个包含两个浮点数的元组
    help='单个仓库的经纬度坐标 (格式: 纬度 经度)'  # 参数的帮助说明
)
@click.option(
    '--warehouse-coords',
    type=str,
    help='多个仓库坐标，格式: "lat1,lon1 lat2,lon2"'  # 参数的帮助说明
)
@click.option(
    '--orders-excel',
    type=click.Path(exists=True, dir_okay=False, readable=True),  # 必须是存在且可读的文件路径
    required=True,  # 此参数为必填项
    help='包含订单数据的Excel文件路径'  # 参数的帮助说明
)
@click.option(
    '--per-delivery-duration',
    type=int,  # 参数类型为整数
    default=5 * 60,  # 默认值设置为 300 秒 (5分钟)
    show_default=True,  # 在帮助信息中显示默认值
    help='每单配送时间（秒）'  # 参数的帮助说明
)
@click.option(
    '--work-duration',
    type=int,  # 参数类型为整数
    default=10 * 60 * 60,  # 默认值设置为 36000 秒 (10小时)
    show_default=True,  # 在帮助信息中显示默认值
    help='允许的总工作时长（秒）'  # 参数的帮助说明
)
def main(warehouse_coord, warehouse_coords, orders_excel, per_delivery_duration, work_duration):
    """
    基于订单数据进行聚类分析，考虑配送时长和工作时间限制。
    """
    # 处理仓库坐标参数
    if warehouse_coords and warehouse_coord:
        # 两个参数都存在时优先使用 warehouse_coords
        coords_list = parse_warehouse_coords(warehouse_coords)
    elif warehouse_coords:
        coords_list = parse_warehouse_coords(warehouse_coords)
    elif warehouse_coord:
        coords_list = [warehouse_coord]
    else:
        raise click.ClickException("必须提供 --warehouse-coord 或 --warehouse-coords 参数")

    try:
        # 读取订单数据
        click.echo(f"正在读取订单数据...")
        df = pd.read_excel(orders_excel, dtype={'business_code': str, 'latitude': float, 'longitude': float})
        # 提取经纬度坐标
        points = df[['latitude', 'longitude']].values
        click.echo(f"共读取 {len(points)} 个运单。")

        # 根据仓库坐标获取边界框
        bbox = get_bbox(coords_list, points)

        # 读取地图数据（假设此函数使用仓库坐标）
        click.echo(f"加载地图数据...")
        load_map_data(bbox)

        # 启动DirectRouterRpcServer
        direct_router_rpc_server = DirectRouterRpcServer()
        direct_router_rpc_server.start_server(bbox=bbox)

        # 启动MultiStopRouterRPCServer
        multi_stop_router_rpc_server = MultiStopRouterRPCServer()
        multi_stop_router_rpc_server.start_server(bbox=bbox)

        # 预处理数据
        click.echo("正在预处理数据...")
        preprocess_data(points, 50)

        if points.shape[0] < 2000:
            click.echo(f"运单数量: {len(points)}，进行单进程聚类...")
            labels = duration_limited_cluster(points, coords_list, per_delivery_duration, work_duration)
        else:
            click.echo(f"运单数量: {len(points)}，进行并行聚类...")
            labels = batch_cluster(bbox, points, coords_list, per_delivery_duration, work_duration)

        # 可视化并导出结果
        click.echo("正在导出和可视化结果...")
        result_visualizer = ResultVisualizer(points, labels, coords_list, per_delivery_duration)
        # 将聚类标签添加到原始DataFrame中并导出
        output_filename = result_visualizer.export_json(df)
        click.echo(f"结果已导出至 {output_filename}")

        click.echo(f'<done>{output_filename}</done>')  # 使用 click.echo 输出完成信息

        # 停止RPC服务器
        direct_router_rpc_server.stop_server()
        multi_stop_router_rpc_server.stop_server()
    except Exception as e:
        tb_str = traceback.format_exc()
        click.echo(tb_str)
        click.echo(f'<error>{str(e)}</error>')  # 使用 click.echo 输出错误信息


def batch_cluster(bbox: Tuple[float, float, float, float], points: np.ndarray,
                  warehouse_coords: List[Tuple[float, float]], per_delivery_duration: int,
                  work_duration: int) -> np.ndarray:
    """ 拆分区域，并行聚类 """
    click.echo(f"运单数量: {len(points)}，进行区域拆分并行聚类...")
    mini_cluster_size = 10  # 每个微簇的点数
    zone_size = min(1500, round(points.shape[0] / 3))  # 每个区域的点数
    # 1.聚类微簇
    click.echo("正在进行微簇聚类...")
    num_clusters = math.ceil(points.shape[0] / mini_cluster_size)
    centroids = random_choice(points, num_clusters)
    clusterer = SizeLimitedClusterer(points, mini_cluster_size)
    mini_cluster_labels, _ = clusterer.clustering(centroids, step=1, max_iter=30)

    # 2.每个微簇取一个点
    mini_cluster_points = []
    for l in np.unique(mini_cluster_labels):
        ps = points[mini_cluster_labels == l]
        mini_cluster_center = np.mean(points[mini_cluster_labels == 0], axis=0)
        center_index = np.argmin(np.linalg.norm(mini_cluster_center - ps, axis=1))
        mini_cluster_points.append(ps[center_index])
    mini_cluster_points = np.array(mini_cluster_points)

    # 3.区域聚类
    click.echo("正在进行区域聚类...")
    num_clusters = math.ceil(mini_cluster_points.shape[0] / round(zone_size / mini_cluster_size))
    centroids = random_choice(mini_cluster_points, num_clusters)
    clusterer = SizeLimitedClusterer(mini_cluster_points, round(zone_size / mini_cluster_size))
    mini_cluster_to_zone_labels, _ = clusterer.clustering(centroids, step=2, max_iter=30)

    # 4.将微簇标签映射到区域标签，得到 zone_labels
    un_mini_cluster_labels = np.unique(mini_cluster_labels)
    zone_labels = np.full(len(mini_cluster_labels), -1)
    for l in mini_cluster_to_zone_labels:
        zone_labels[np.isin(mini_cluster_labels, un_mini_cluster_labels[mini_cluster_to_zone_labels == l])] = l

    # 5.批量聚类
    click.echo(f"正在从 {len(np.unique(zone_labels))} 个区域中进行批量聚类...")
    result_labels = batch_duration_limited_cluster(bbox, points, zone_labels, warehouse_coords, per_delivery_duration,
                                                   work_duration)
    return result_labels


if __name__ == '__main__':
    main()  # click 会自动处理命令行参数并调用 main 函数
