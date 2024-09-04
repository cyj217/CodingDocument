import numpy as np
from scipy.optimize import linear_sum_assignment

# 创建分配问题的成本矩阵
cost_matrix = np.array([[5, 1000, 9, 7],
                       [6, 4, 9, 8],
                       [1000, 7, 6, 3],
                       [5, 4, 11, 6]])

# 解决分配问题
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# 打印最佳分配方案
for i in range(len(row_ind)):
    print(f"任务 {row_ind[i]} 分配给人员 {col_ind[i]}")