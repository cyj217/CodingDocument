import heapq
import networkx as nx
import matplotlib.pyplot as plt

def dijkstra(graph, start):
    # 初始化距离字典，用于存储起始节点到其他节点的最短距离
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # 使用优先队列（最小堆）存储待处理的节点
    heap = [(0, start)]

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        # 如果当前节点已经有更短的距离，则跳过
        if current_distance > distances[current_node]:
            continue

        # 遍历当前节点的邻居节点
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            # 如果通过当前节点前往邻居节点的距离更短，则更新最短距离
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))

    return distances

# 创建图形
graph = {
    '1': {'2': 5, '3': 7, '4': 4},
    '2': {'3': 6, '4': 4, '5': 5},
    '3': {'5': 5},
    '4': {'3': 3, '5': 9},
    '5': {},
}

# 指定起始节点
start_node = '1'

# 调用Dijkstra算法
distances = dijkstra(graph, start_node)

# 输出最短距离
for node, distance in distances.items():
    print(f"最短距离从节点 {start_node} 到节点 {node}：{distance}")

# 构建最短路径有向图形
shortest_path_graph = nx.DiGraph()
for node, distance in distances.items():
    shortest_path_graph.add_node(node, distance=distance)
    if node in graph:
        for neighbor, weight in graph[node].items():
            shortest_path_graph.add_edge(node, neighbor, weight=weight)

# 绘制最短路径有向图形
pos = nx.spring_layout(shortest_path_graph)
edge_labels = nx.get_edge_attributes(shortest_path_graph, 'weight')
node_labels = nx.get_node_attributes(shortest_path_graph, 'distance')
nx.draw_networkx(shortest_path_graph, pos, with_labels=True, node_size=500, arrows=True)
nx.draw_networkx_edge_labels(shortest_path_graph, pos, edge_labels=edge_labels)
nx.draw_networkx_labels(shortest_path_graph, pos, labels=node_labels)
plt.title('Shortest Path Graph')
plt.axis('off')
plt.show()