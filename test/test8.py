import networkx as nx
import random

def generate_random_graph(num_nodes, probability):
    """
    生成一个随机图
    :param num_nodes: 节点数量
    :param probability: 边的存在概率
    :return: 随机图
    """
    G = nx.Graph()
    G.add_nodes_from(range(1, num_nodes + 1))
    for i in range(1, num_nodes + 1):
        for j in range(i + 1, num_nodes + 1):
            if random.random() < probability:
                G.add_edge(i, j, weight=random.randint(1, 10))
    return G

def find_shortest_path(graph, source, target):
    """
    查找最短路径
    :param graph: 图
    :param source: 起始节点
    :param target: 目标节点
    :return: 最短路径
    """
    try:
        path = nx.shortest_path(graph, source=source, target=target, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return None

# 设置节点数量和边的存在概率
num_nodes = 10
edge_probability = 0.4

# 生成随机图
graph = generate_random_graph(num_nodes, edge_probability)

# 随机选择起始节点和目标节点
source_node = random.randint(1, num_nodes)
target_node = random.randint(1, num_nodes)

# 查找最短路径
shortest_path = find_shortest_path(graph, source_node, target_node)

# 打印结果
print("随机生成的图:")
print(f"节点数量: {num_nodes}")
print(f"边的存在概率: {edge_probability}")
print(f"起始节点: {source_node}")
print(f"目标节点: {target_node}")
print("最短路径:")
if shortest_path:
    print(shortest_path)
else:
    print("无法找到最短路径")