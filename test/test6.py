import gurobipy as grb # 导入gurobi模块
from gurobipy import GRB  # 导入gurobi常量
import numpy as np # 导入numpy模块
import matplotlib.pyplot as plt # 导入matplotlib模块
import time # 导入time模块

# 定义城市数量
num_cities = 30

# 生成随机的城市坐标
city_coordinates = np.random.rand(num_cities, 2)  # 生成大小为(num_cities, 2)的随机坐标矩阵，范围在0到1之间

# 创建模型
model = grb.Model("TSP")

# 创建变量
num_cities = len(city_coordinates)
x = model.addVars(num_cities, num_cities, vtype=GRB.BINARY, name="x")

# 添加约束：每个城市只能被访问一次
model.addConstrs(x[i, j] == 0 for i in range(num_cities) for j in range(num_cities) if i == j)
model.addConstrs(grb.quicksum(x[i, j] for j in range(num_cities) if j != i) == 1 for i in range(num_cities))
model.addConstrs(grb.quicksum(x[i, j] for i in range(num_cities) if i != j) == 1 for j in range(num_cities))

# 添加约束：DFJ禁止子回路
for i in range(1, num_cities):
    for j in range(1, num_cities):
        if i != j:
            model.addConstr(x[i, j] + x[j, i] + x[j, 0] <= x[i, 0])

# 设置目标函数：最小化总距离
model.setObjective(grb.quicksum(grb.LinExpr(np.sqrt((city_coordinates[i][0] - city_coordinates[j][0])**2 +
(city_coordinates[i][1] - city_coordinates[j][1])**2), x[i, j])
for i in range(num_cities) for j in range(num_cities)), GRB.MINIMIZE)

# 记录开始时间
start_time = time.time()

# 求解模型
model.optimize()

# 记录结束时间
end_time = time.time()

# 提取最优解
optimal_path = [(i, j) for i in range(num_cities) for j in range(num_cities) if x[i, j].x > 0.5]

# 绘制城市坐标图
plt.scatter(city_coordinates[:, 0], city_coordinates[:, 1], color='b', marker='o', label='Cities')

# 绘制最优路径
for i, j in optimal_path:
    plt.plot([city_coordinates[i, 0], city_coordinates[j, 0]], [city_coordinates[i, 1], city_coordinates[j, 1]], color='r')

plt.title('TSP Solution')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.legend()
plt.show()

# 打印最优解
if model.status == GRB.OPTIMAL:
    print("最短路径长度:", model.objVal)
    print("最短路径:")
    for v in model.getVars():
        if v.x > 0:
            print(v.varName)

# 打印计算时间
elapsed_time = end_time - start_time
print("计算时间:", elapsed_time, "秒")