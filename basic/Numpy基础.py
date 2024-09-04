import numpy as np
from numpy import random

a = np.array([[1,2,3],
              [4,5,6]]) # 创建一个二维数组
print (a.shape) # 打印数组的维度
a_one = np.ones((2,3)) # 创建一个2行3列的全1数组
print (a_one) # 打印数组
a_zero = np.zeros((2,3)) # 创建一个2行3列的全0数组
print (a_zero) # 打印数组
a_eye = np.eye(3) # 创建一个3行3列的单位矩阵
print (a_eye) # 打印数组
a_rand = np.random.random((2,3)) # 创建一个2行3列的随机数组
print (a_rand) # 打印数组
print (a_rand.T) # 打印数组的转置
print (a_rand.reshape(3,2)) # 打印数组的重塑

# 重塑数组形状
a = np.array([[1,2,3],
                [4,5,6]]) # 创建一个二维数组
print (a.ravel()) # 将二维数组转成一维数组
print (a.reshape(3,2)) # 改变二维数组形状
print (a.reshape((-1,1))) # 将二维数组转成列向量

# 数组合并
a = np.array([[1,2,3],
                [4,5,6]]) # 创建一个二维数组
b = np.array([[7,8,9],
                [10,11,12]]) # 创建一个二维数组
print (np.vstack((a,b))) # 垂直合并数组
print (np.hstack((a,b))) # 水平合并数组

# 数组分割
a = np.array([[1,2,3,4,5,6],
              [4,5,6,7,8,9],
              [7,8,9,10,11,12],
              [10,11,12,13,14,15]])
print (a[0:3,1:4]) # 打印数组的第1-3行，第2-4列
print (a[:3,-4:]) # 打印数组的第1-3行，倒数第4列到最后一列

# 随机数
random.seed(1234) # 设置随机数种子
random.rand(3,2) # 产生均匀分布的随机数，维度是3*2
random.randn(3,2) # 产生标准正态分布的随机数，维度是3*2
random.random((3,2)) # 在[0,1)区间产生随机数，维度是3*2
random.randint(low=2,high=10,size=(3,2)) # 产生[2,10)区间的随机整数，维度是3*2
random.normal(loc=0.0,scale=1.0,size=(3,2)) # 产生均值为0，标准差为1的正态分布随机数，维度是3*2
random.poisson(lam=100,size=(3,2)) # 产生随机泊松分布的随机数，维度是3*2
random.uniform(low=3,high=10,size=(3,2)) # 产生[3,10)区间的均匀分布的随机数，维度是3*2
random.beta(a=3,b=5,size=(3,2)) # 产生beta分布的随机数，维度是3*2
random.binomial(n=10,p=0.5,size=(3,2)) # 产生二项分布的随机数，维度是3*2
random.chisquare(df=2,size=(3,2)) # 产生卡方分布的随机数，维度是3*2
random.gamma(shape=2,scale=2,size=(3,2)) # 产生gamma分布的随机数，维度是3*2
random.logistic(loc=0,scale=1,size=(3,2)) # 产生logistic分布的随机数，维度是3*2
random.exponential(scale=2,size=(3,2)) # 产生指数分布的随机数，维度是3*2
random.f(dfnum=2,dfden=3,size=(3,2)) # 产生F分布的随机数，维度是3*2

# 随机采样
random.seed(1234) # 设置随机数种子
samples = [1,2,3,4,5,6,7,8,9] # 创建一个列表
random.choice(samples,size=5,replace=True) # 从列表中有放回的随机采样5个元素
random.choice(samples,size=5,replace=False) # 从列表中无放回的随机采样5个元素
random.shuffle(samples) # 将列表元素随机打乱
print(samples) # 打印列表

# 矩阵运算
a1 = np.array([[4,5,6],[1,2,3]]) # 创建一个二维数组
a2 = np.array([[6,5,4],[3,2,1]]) # 创建一个二维数组
print (a1+a2) # 矩阵加法
print (a1-a2) # 矩阵减法
print (a1*a2) # 矩阵对应元素相乘
print (a1/a2) # 矩阵除法
print (a1%a2) # 矩阵求余
print (a1**3) # 矩阵求幂
print (a1*3) # 矩阵数乘
a3 = a2.T # 矩阵转置
print(np.dot(a1,a3)) # 矩阵相乘
a = np.array([[1,2,3],[4,5,6],[5,4,3]]) # 创建一个二维数组
print (np.linalg.inv(a)) # 矩阵求逆
eigenValues,eigenVectors = np.linalg.eig(a) # 矩阵特征值和特征向量
print (eigenValues) # 打印特征值
print (eigenVectors) # 打印特征向量
print (np.linalg.det(a)) # 矩阵行列式
U,sigma,VT = np.linalg.svd(a,full_matrices=False) # 矩阵奇异值分解

# 线性代数
A = np.array([
    [1,-2,1],
    [0,2,-8],
    [-4,5,9]       
])
B = np.array([0,8,-9])
result = np.linalg.solve(A,B) # 求解线性方程组
print('x=',result[0]) # 打印x
print('y=',result[1]) # 打印y
print('z=',result[2]) # 打印z
