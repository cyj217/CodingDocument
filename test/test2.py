import numpy as np

# 设置随机数种子，以确保结果可重复
np.random.seed(42)

# 定义回归函数
def regression_function(a, b, c):
    return 1*a + 2*b + 0.1*c

a = np.array([22.29, 10.64, 15.84, 26.89, 18.57, 13.89, 20.96, 16.38])
b = np.array([14.78, 7.19, 8.58, 12.43, 13.66, 11.97, 6.59, 10.32])
c = np.array([2.19, 9.46, 4.18, 8.3, 5.56, 7.3, 2.81, 1.26])

# 生成输出数据 y，通过回归函数添加噪声
y_true = regression_function(a, b, c)
noise = np.random.randn(len(a))  # 根据输入数据长度生成服从标准正态分布的噪声
y = y_true + noise

print("y:")
print(y)