import click
import numpy as np
from scipy.spatial import KDTree
import itertools
import networkx as nx
from src.services.direct_router.direct_router_rpc_client import DirectRouterRpcClient
from src.services.geo_indexer.geo_indexer import GeoIndexer


def _build_cluster_graph(original_indices, point_index_to_node_id, router_client, min_cluster_size):
    """
    构建簇内完全图并计算最小生成树
    
    Args:
        original_indices: 簇内点的原始索引数组
        point_index_to_node_id: 点索引到路网节点ID的映射
        router_client: 路径计算客户端
        min_cluster_size: 最小簇大小
        
    Returns:
        tuple: (mst, original_indices, cluster_node_ids) 或 None（如果构建失败）
    """
    # 获取簇内所有点对应的路网节点ID
    cluster_node_ids = [point_index_to_node_id[i] for i in original_indices]

    # 创建NetworkX图，节点使用本地索引（0,1,2...）
    G = nx.Graph()
    for i, node_id in enumerate(cluster_node_ids):
        G.add_node(i, node_id=node_id)

    # 生成所有点对组合，用于计算两两之间的路径时间
    node_id_pairs = list(itertools.combinations(cluster_node_ids, 2))
    if not node_id_pairs:
        return None  # 只有一个点的簇无法构建图

    # 批量查询所有点对之间的路径时间
    durations = router_client.batch_calc_path_duration(node_id_pairs)

    # 建立路网节点ID到图节点索引的映射
    node_id_to_index = {node_id: i for i, node_id in enumerate(cluster_node_ids)}
    for (id1, id2), duration in zip(node_id_pairs, durations):
        if duration is not None:
            # 将路网节点ID转换为图的本地节点索引
            idx1 = node_id_to_index[id1]
            idx2 = node_id_to_index[id2]
            # 添加边，权重为路径时间（允许无穷大值）
            G.add_edge(idx1, idx2, weight=duration)

    # 处理孤立节点（无法达到的点）
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        # 移除孤立节点，因为它们无法与其他点连通
        G.remove_nodes_from(isolated_nodes)
        if len(G.nodes()) < min_cluster_size:
            return None  # 剩余节点太少

        # 重新映射节点索引：将剩余节点重新编号为连续的 0,1,2...
        remaining_nodes = sorted(G.nodes())
        node_mapping = {old_node: new_node for new_node, old_node in enumerate(remaining_nodes)}

        # 创建新图，使用重新编号的节点
        G_new = nx.Graph()
        for new_node in range(len(remaining_nodes)):
            G_new.add_node(new_node)

        # 复制所有边到新图，使用新的节点编号
        for old_u, old_v, data in G.edges(data=True):
            new_u = node_mapping[old_u]
            new_v = node_mapping[old_v]
            G_new.add_edge(new_u, new_v, weight=data['weight'])

        G = G_new

        # 同步更新相关的数据结构
        original_indices = original_indices[remaining_nodes]
        cluster_node_ids = [cluster_node_ids[i] for i in remaining_nodes]

    # 检查和修复图的连通性
    if not nx.is_connected(G):
        # 如果图不连通，尝试使用仅有有限权重的边
        finite_edges = [(u, v, d) for u, v, d in G.edges(data=True) if np.isfinite(d['weight'])]
        if len(finite_edges) == 0:
            return None  # 没有有效连接

        # 创建仅包含有限权重边的子图
        G_finite = nx.Graph()
        G_finite.add_nodes_from(G.nodes())
        for u, v, d in finite_edges:
            G_finite.add_edge(u, v, weight=d['weight'])

        if not nx.is_connected(G_finite):
            return None  # 即使只用有限边也不连通

        # 使用修复后的连通图
        G = G_finite

    # 使用NetworkX的Kruskal算法计算最小生成树
    mst = nx.minimum_spanning_tree(G, weight='weight')

    return mst, original_indices, cluster_node_ids


def _find_small_groups_from_mst(mst, original_indices, cluster_node_ids, small_group_threshold, moved_points):
    """
    从最小生成树中找到可分离的小团体
    
    Args:
        mst: 最小生成树
        original_indices: 簇内点的原始索引数组
        cluster_node_ids: 簇内点对应的路网节点ID列表
        small_group_threshold: 小团体大小阈值（相对于原簇大小的比例）
        moved_points: 已移动的点集合
        
    Returns:
        list: 小团体信息列表，每个元素包含 (keep_cost, sub_group_original_indices, sub_group_node_ids)
    """
    # 获取MST的所有边，并按权重降序排列（最长的边在前）
    edges = list(mst.edges(data=True))
    edges.sort(key=lambda x: x[2]['weight'], reverse=True)

    # 决定要评估的边数量（只评估最长的10%的边）
    n_edges_to_evaluate = max(1, int(len(edges) * 0.1))

    small_groups = []

    # 尝试移除最长的边来分离小团体
    for i in range(min(n_edges_to_evaluate, len(edges))):
        edge = edges[i]
        u, v, data = edge
        keep_cost = data['weight']  # 保持这条边的成本

        # 创建临时MST并移除当前边，模拟分离操作
        temp_mst = mst.copy()
        temp_mst.remove_edge(u, v)

        # 检查移除边后的连通分量
        components = list(nx.connected_components(temp_mst))
        if len(components) < 2:
            continue  # 移除边后仍然连通，说明有其他路径

        # 找到较小的连通分量（小团体）
        smaller_component = min(components, key=len)

        # 检查小团体大小是否合适（不能太大）
        if len(smaller_component) > len(original_indices) * small_group_threshold:
            continue  # 小团体太大，不适合移动

        # 将图的本地节点索引转换为原始点索引
        sub_group_local_indices = np.array(list(smaller_component))
        sub_group_original_indices = original_indices[sub_group_local_indices]


        # 获取小团体的路网节点ID
        sub_group_node_ids = [cluster_node_ids[i] for i in sub_group_local_indices]

        small_groups.append((keep_cost, sub_group_original_indices, sub_group_node_ids))

    return small_groups


def _calculate_move_cost(sub_group_node_ids, sub_group_original_indices, points,
                         refined_labels, point_index_to_node_id, router_client,
                         kdtree, avg_cluster_size, cid):
    """
    计算小团体移动到目标簇的成本
    
    Args:
        sub_group_node_ids: 小团体的路网节点ID列表
        sub_group_original_indices: 小团体的原始点索引
        points: 所有点的坐标数组
        refined_labels: 当前聚类标签
        point_index_to_node_id: 点索引到路网节点ID的映射
        router_client: 路径计算客户端
        kdtree: KD树用于邻近搜索
        avg_cluster_size: 平均簇大小
        cid: 当前簇ID
        
    Returns:
        tuple: (min_move_cost, best_target_cluster) 或 (None, None)（如果没有合适目标）
    """
    # 计算小团体中心
    sub_group_points = points[sub_group_original_indices]
    sub_group_centroid = sub_group_points.mean(axis=0)

    # 使用KD树在小团体中心附近寻找邻近点
    distances, indices = kdtree.query(sub_group_centroid, k=min(avg_cluster_size, len(points)))
    # 获取这些邻近点所在的簇作为候选目标
    candidate_clusters = set(refined_labels[indices])
    candidate_clusters.discard(cid)  # 排除当前簇
    candidate_clusters.discard(-1)  # 排除噪声点

    if not candidate_clusters:
        return None, None  # 没有合适的候选目标簇

    min_move_cost = float('inf')  # 记录最小移动成本
    best_target_cluster = -1  # 记录最优目标簇

    # 遍历所有候选目标簇，计算移动成本
    for target_cid in candidate_clusters:
        # 获取目标簇的所有点和对应的路网节点ID
        target_cluster_indices = np.where(refined_labels == target_cid)[0]
        target_node_ids = [point_index_to_node_id[i] for i in target_cluster_indices]

        # 生成小团体与目标簇之间的所有点对组合
        cross_cluster_pairs = list(itertools.product(sub_group_node_ids, target_node_ids))
        if not cross_cluster_pairs:
            continue  # 目标簇为空

        # 批量计算跨簇路径时间
        move_durations = router_client.batch_calc_path_duration(cross_cluster_pairs)

        # 过滤有效的路径时间
        valid_durations = [d for d in move_durations if d is not None]
        if not valid_durations:
            continue  # 没有有效路径

        # 找到最小的跨簇连接成本
        current_move_cost = min(valid_durations)
        if current_move_cost < min_move_cost:
            min_move_cost = current_move_cost
            best_target_cluster = target_cid

    if not np.isfinite(min_move_cost):
        return None, None

    return min_move_cost, best_target_cluster


def _execute_optimal_moves(candidate_moves, refined_labels, moved_points, move_count, max_concurrent_moves, iteration_count):
    """
    执行收益最大的移动操作
    
    Args:
        candidate_moves: 候选移动操作列表
        refined_labels: 当前聚类标签数组
        moved_points: 已移动的点集合
        move_count: 每个点的移动次数统计字典
        max_concurrent_moves: 最大并发移动次数
        iteration_count: 当前迭代次数
        
    Returns:
        tuple: (moves_executed, iteration_profit, iteration_points_moved)
    """
    if not candidate_moves:
        return 0, 0, 0  # 没有任何有收益的移动

    # 按平均每点收益降序排列，优先执行收益最高的移动
    candidate_moves.sort(key=lambda x: x['avg_profit'], reverse=True)

    # 如果最好的移动收益也很小，则停止优化
    if candidate_moves[0]['profit'] < 20:  # 20秒阈值
        return 0, 0, 0

    # 初始化本轮迭代的统计变量
    processed_clusters = set()  # 记录已处理的簇，避免冲突
    moves_executed = 0  # 已执行的移动次数
    iteration_profit = 0  # 本轮迭代的总收益
    iteration_points_moved = 0  # 本轮迭代移动的总点数

    # 逐个执行候选移动，避免冲突
    for move in candidate_moves:
        # 检查是否超过本轮允许的最大移动次数
        if moves_executed >= max_concurrent_moves:
            break  # 防止过度变动影响稳定性

        source_cluster = move['source_cluster']
        target_cluster = move['target_cluster']

        # 检查相关簇是否已经被处理过（避免同一簇多次变动）
        if source_cluster in processed_clusters or target_cluster in processed_clusters:
            continue  # 跳过可能引起冲突的移动

        # 执行移动：更新点的簇标签
        refined_labels[move['indices_to_move']] = move['target_cluster']
        # 记录已移动的点，避免后续重复移动
        moved_points.update(move['indices_to_move'])
        # 更新每个移动点的移动次数统计
        for point_idx in move['indices_to_move']:
            move_count[point_idx] = move_count.get(point_idx, 0) + 1
        # 标记相关簇已处理
        processed_clusters.add(source_cluster)
        processed_clusters.add(target_cluster)

        # 更新统计信息
        iteration_profit += move['profit']
        iteration_points_moved += len(move['indices_to_move'])
        moves_executed += 1

    # 只有在实际执行了移动时才输出进度信息
    if moves_executed > 0:
        click.echo(
            f"第 {iteration_count} 轮: 移动 {moves_executed} 次, 收益 {iteration_profit:.1f}s, 点数 {iteration_points_moved}")

    return moves_executed, iteration_profit, iteration_points_moved


def refine_clusters(points, labels, min_cluster_size=3, small_group_threshold=0.1, max_concurrent_moves=5):
    """
    基于NetworkX最小生成树的聚类优化算法
    
    该函数通过迭代优化的方式，将聚类中的小团体重新分配到更合适的簇中，
    以减少整体路径成本。算法的核心思想是：对每个簇构建最小生成树，
    然后移除成本最高的边来分离小团体，并将其重新分配到成本更低的目标簇。
    
    Args:
        points (np.ndarray): 所有配送点的坐标数组，形状为 (n_points, 2)
        labels (np.ndarray): 每个点的初始聚类标签，形状为 (n_points,)
        min_cluster_size (int): 最小簇大小，小于此大小的簇不参与优化
        small_group_threshold (float): 小团体阈值，分离出的小团体大小不能超过原簇的此比例
        max_concurrent_moves (int): 每轮迭代最大允许的并发移动次数
        
    Returns:
        np.ndarray: 优化后的聚类标签数组
        
    算法流程：
        1. 初始化路径计算服务和地理索引器
        2. 迭代优化：
           - 为每个簇构建完全图并计算最小生成树
           - 移除MST中成本最高的边来分离小团体
           - 计算将小团体移动到其他簇的成本
           - 执行收益最大的移动操作
        3. 输出优化统计信息
    """
    # ====== 1. 初始化阶段 ======
    refined_labels = np.copy(labels)  # 复制标签数组，避免修改原始数据
    moved_points = set()  # 记录已移动的点，避免重复移动
    move_count = {}  # 记录每个点的移动次数，用于收益衰减计算

    # 初始化路径计算和地理索引服务
    router_client = DirectRouterRpcClient()
    router_client.init()
    geo_indexer = GeoIndexer()

    # 为每个点找到最近的路网节点ID，用于路径计算
    all_node_ids = [geo_indexer.get_nearest_node_id(tuple(pt)) for pt in points]
    point_index_to_node_id = {i: node_id for i, node_id in enumerate(all_node_ids)}

    # ====== 2. 迭代参数设置 ======
    iteration_count = 0  # 当前迭代次数
    total_profit = 0.0  # 累计优化收益（时间减少量）
    total_points_moved = 0  # 累计移动的点数
    max_iterations = 100  # 最大迭代次数，防止无限循环

    click.echo("开始基于NetworkX的聚类优化迭代...")

    # 获取有效的簇ID列表（排除噪声点标签-1）
    initial_cluster_ids = np.unique(refined_labels)
    if -1 in initial_cluster_ids:
        initial_cluster_ids = initial_cluster_ids[initial_cluster_ids != -1]

    if len(initial_cluster_ids) == 0:
        click.echo("没有有效的簇，停止优化")
        return refined_labels

    # 计算平均簇大小，用于邻近搜索的范围控制
    avg_cluster_size = max(2, min(int(len(points) / len(initial_cluster_ids)), len(points) // 10))
    kdtree = KDTree(points)  # 构建KD树用于快速邻近搜索

    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            click.echo(f"达到最大迭代次数 {max_iterations}，停止优化")
            break

        # ====== 3. 每轮迭代开始 ======
        # 获取当前所有有效簇ID
        cluster_ids = np.unique(refined_labels)
        if -1 in cluster_ids:
            cluster_ids = cluster_ids[cluster_ids != -1]  # 排除噪声点

        if len(cluster_ids) == 0:
            break  # 没有有效簇时退出

        candidate_moves = []  # 存储所有候选移动操作

        # ====== 4. 遍历每个簇进行优化分析 ======
        for cid in cluster_ids:
            # 获取当前簇的所有点索引
            cluster_mask = (refined_labels == cid)
            original_indices = np.where(cluster_mask)[0]

            # 跳过过小的簇（不满足最小簇大小要求）
            if len(original_indices) < min_cluster_size:
                continue

            # ====== 4.1 构建簇内图并计算MST ======
            graph_result = _build_cluster_graph(original_indices, point_index_to_node_id,
                                                router_client, min_cluster_size)
            if graph_result is None:
                continue  # 构建图失败，跳过这个簇

            mst, original_indices, cluster_node_ids = graph_result

            # ====== 4.2 从MST中找到可分离的小团体 ======
            small_groups = _find_small_groups_from_mst(mst, original_indices, cluster_node_ids,
                                                       small_group_threshold, moved_points)

            # ====== 4.3 为每个小团体计算移动成本和收益 ======
            for keep_cost, sub_group_original_indices, sub_group_node_ids in small_groups:
                # 计算移动到目标簇的成本
                move_result = _calculate_move_cost(sub_group_node_ids, sub_group_original_indices,
                                                   points, refined_labels, point_index_to_node_id,
                                                   router_client, kdtree, avg_cluster_size, cid)
                if move_result[0] is None:
                    continue  # 没有合适的移动目标

                min_move_cost, best_target_cluster = move_result

                # 计算移动收益：保持当前连接的成本 - 移动到新簇的成本
                profit = keep_cost - min_move_cost
                
                # 基于移动次数的收益衰减：找到小团体中移动次数最多的点
                max_moves = max(move_count.get(idx, 0) for idx in sub_group_original_indices)
                # 应用衰减公式：收益 * (0.8)^最大移动次数
                profit *= (0.8 ** max_moves)
                
                if profit > 0:  # 只有有收益的移动才值得考虑
                    candidate_moves.append({
                        'profit': profit,  # 总收益（时间减少）
                        'avg_profit': profit / len(sub_group_original_indices),  # 平均每点收益
                        'indices_to_move': sub_group_original_indices,  # 要移动的点索引
                        'target_cluster': best_target_cluster,  # 目标簇ID
                        'source_cluster': cid,  # 源簇ID
                        'keep_cost': keep_cost,  # 保持现状的成本
                        'move_cost': min_move_cost  # 移动的成本
                    })

        # ====== 5. 执行最优移动操作 ======
        moves_executed, iteration_profit, iteration_points_moved = _execute_optimal_moves(
            candidate_moves, refined_labels, moved_points, move_count, max_concurrent_moves, iteration_count)

        if moves_executed == 0:
            break  # 没有执行任何移动，结束优化

        # ====== 5.1 更新累计统计 ======
        total_profit += iteration_profit
        total_points_moved += iteration_points_moved

    # ====== 6. 输出优化结果统计 ======
    click.echo(f"\n基于NetworkX的聚类优化完成!")
    click.echo(f"总迭代次数: {iteration_count}")
    click.echo(f"总收益: {total_profit:.2f}s")  # 总计节省的时间
    click.echo(f"总移动点数: {total_points_moved}")  # 总计移动的配送点数量
    if total_points_moved > 0:
        click.echo(f"平均每点收益: {total_profit / total_points_moved:.2f}s")  # 平均每点节省时间

    return refined_labels  # 返回优化后的聚类标签
