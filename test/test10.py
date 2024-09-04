import numpy as np
from scipy.optimize import linear_sum_assignment
import time

rows = 1000
cols = 1000

cost_matrix = np.random.randint(low=10, high=100, size=(rows, cols))

start_time = time.time()

row_ind, col_ind = linear_sum_assignment(cost_matrix)

end_time = time.time()

print(cost_matrix)

if row_ind.size == rows and col_ind.size == cols:
    for i in range(rows):
        print(f"任务 {i} 分配给工人 {col_ind[i]}，成本为 {cost_matrix[i, col_ind[i]]}")

        total_cost = cost_matrix[row_ind, col_ind].sum()
    print(f"总成本: {total_cost}")
else:
    print("无法找到最佳指派")

execution_time = end_time - start_time
print(f"计算时间: {execution_time} 秒")