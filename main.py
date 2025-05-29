import click
import pandas as pd
import math
from src.algorithms.clustering_algorithm.duration_limited_clusterer import DurationLimitedClusterer
from src.utils.utils import random_choice
from src.data.data_loader import load_map_data
from src.algorithms.result_visualizer.result_visualizer import ResultVisualizer
from src.algorithms.data_preprocessor.preprocess_data import preprocess_data


@click.command()
@click.option(
    '--warehouse-coord',
    type=click.Tuple([float, float]),  # 定义为一个包含两个浮点数的元组
    required=True,  # 此参数为必填项
    help='仓库的经纬度坐标 (格式: 纬度 经度)'  # 参数的帮助说明
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
def main(warehouse_coord, orders_excel, per_delivery_duration, work_duration):
    """
    基于订单数据进行聚类分析，考虑配送时长和工作时间限制。
    """
    try:
        # 读取订单数据
        click.echo(f"正在读取订单数据...")
        df = pd.read_excel(orders_excel, dtype={'business_code': str, 'latitude': float, 'longitude': float})
        # 提取经纬度坐标
        points = df[['latitude', 'longitude']].values
        click.echo(f"共读取 {len(points)} 个运单。")

        # 读取地图数据（假设此函数使用仓库坐标）
        click.echo(f"加载地图数据...")
        load_map_data(warehouse_coord, points)

        # 预处理数据
        click.echo("正在预处理数据...")
        preprocess_data(warehouse_coord, points, 200)

        # 随机初始化聚类中心
        num_clusters = math.ceil(points.shape[0] / 60)
        click.echo(f"正在为 {num_clusters} 个聚类随机初始化中心点...")
        centroids = random_choice(points, num_clusters)

        # 执行带有时长限制的聚类算法
        click.echo("开始执行带有时长限制的聚类算法...")
        clusterer = DurationLimitedClusterer(
            warehouse_coord, points, per_delivery_duration, work_duration
        )
        labels, final_centroids = clusterer.clustering(centroids, step=3, max_iter=100, print_step=True)
        click.echo("聚类完成。")

        # 可视化并导出结果
        click.echo("正在导出和可视化结果...")
        result_visualizer = ResultVisualizer(points, labels, warehouse_coord, per_delivery_duration)
        # 将聚类标签添加到原始DataFrame中并导出
        output_filename = result_visualizer.export_json(df)
        click.echo(f"结果已导出至 {output_filename}")

        click.echo(f'<done>{output_filename}</done>')  # 使用 click.echo 输出完成信息
    except Exception as e:
        click.echo(f'<error>{str(e)}</error>')  # 使用 click.echo 输出错误信息


if __name__ == '__main__':
    main()  # click 会自动处理命令行参数并调用 main 函数
