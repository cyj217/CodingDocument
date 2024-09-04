import pulp
import networkx as nx
import matplotlib.pyplot as plt

# 创建问题实例
problem = pulp.LpProblem("Branch and Bound Example", pulp.LpMaximize)

# 定义决策变量
x = pulp.LpVariable("x", lowBound=0, cat="Integer")
y = pulp.LpVariable("y", lowBound=0, cat="Integer")

# 定义目标函数
problem += 3 * x + 5 * y

# 添加约束条件
problem += x + 2 * y <= 10
problem += 3 * x + y <= 12

# 定义决策图
decision_graph = nx.DiGraph()  # 使用有向图

# 定义一个辅助函数，用于绘制决策图
def draw_decision_graph(node, parent_node, level):
    node_name = f"Node {level}"
    decision_graph.add_node(node_name)
    decision_graph.add_edge(parent_node, node_name)

    # 求解子问题并细化变量范围
    sub_problem = problem.copy()
    if node["var_name"] == "x":
        sub_problem += node["var"] >= node["value"] + 1
    elif node["var_name"] == "y":
        sub_problem += node["var"] <= node["value"]

    sub_problem.solve(pulp.PULP_CBC_CMD())

    # 检查子问题是否可行
    if sub_problem.status == pulp.LpStatusOptimal:
        sub_x_value = pulp.value(x)
        sub_y_value = pulp.value(y)
        decision_graph.add_node((sub_x_value, sub_y_value))
        decision_graph.add_edge(node_name, (sub_x_value, sub_y_value))

        # 递归调用绘制子问题的决策图
        for var in [x, y]:
            if var.name != node["var_name"]:
                draw_decision_graph({"var_name": var.name, "var": var, "value": int(var.value())}, node_name, level+1)


# 添加初始节点
initial_node = "Initial"
decision_graph.add_node(initial_node)

# 求解整数规划问题
problem.solve(pulp.PULP_CBC_CMD())

# 输出结果
print("Optimization Results:")
print(f"Objective Value: {pulp.value(problem.objective)}")
for var in problem.variables():
    print(f"{var.name}: {pulp.value(var)}")

# 绘制决策图
draw_decision_graph({"var_name": "x", "var": x, "value": int(x.value())}, initial_node, 1)
draw_decision_graph({"var_name": "y", "var": y, "value": int(y.value())}, initial_node, 1)

# 绘制最优解节点
optimal_node = "Optimal"
decision_graph.add_node(optimal_node)
x_optimal = pulp.value(x)
y_optimal = pulp.value(y)
decision_graph.add_edge((x_optimal, y_optimal), optimal_node)

# 绘制决策图
pos = nx.spring_layout(decision_graph)
nx.draw_networkx(decision_graph, pos, with_labels=True, node_size=500, arrows=True)
plt.title('Branch and Bound Decision Graph')
plt.axis('off')
plt.show()