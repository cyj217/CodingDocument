import numpy as np # 导入numpy库
import matplotlib.pyplot as plt # 导入matplotlib库
from sklearn import datasets # 导入sklearn库中的datasets模块
import warnings # 忽略警告
warnings.filterwarnings('ignore') # 忽略警告
from sklearn.cluster import KMeans # 导入KMeans聚类算法
from sklearn.cluster import MiniBatchKMeans # 导入MiniBatchKMeans聚类算法
from sklearn.cluster import Birch # 导入Birch聚类算法
from sklearn.cluster import DBSCAN # 导入DBSCAN聚类算法

# 数据合并并可视化
x1,y1 = datasets.make_circles(n_samples=5000,
                              factor=.6,
                              noise=0.05) # 生成环形数据
x2,y2 = datasets.make_blobs(n_samples=1000,
                            n_features=2,
                            centers=[[1.2,1.2]],
                            cluster_std=[[.1]],
                            random_state=9) # 生成聚类数据
y2 = y2+2 # 修改标签
x = np.vstack((x1,x2)) # 纵向合并数据
y = np.hstack((y1,y2)) # 横向合并数据
plt.figure(figsize=(5,5)) # 设置画布大小
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False # 设置显示负号
plt.title('原始数据分布') # 设置标题
plt.scatter(x[:,0],x[:,1],c=y,marker='o') # 绘制散点图
plt.show() # 显示图像

# K-Means算法分类
result = KMeans(n_clusters=3,random_state=9).fit_predict(x) # 聚类并预测
plt.figure(figsize=(5,5)) # 设置画布大小
plt.title("KMeans") # 设置标题
plt.scatter(x[:,0],x[:,1],c=result) # 绘制散点图
plt.show() # 显示图像

# 小批量K-Means算法分类
result = MiniBatchKMeans(n_clusters=3,random_state=9).fit_predict(x) # 聚类并预测
plt.figure(figsize=(5,5)) # 设置画布大小
plt.title("MiniBatchKMeans") # 设置标题
plt.scatter(x[:,0],x[:,1],c=result) # 绘制散点图
plt.show() # 显示图像

# Birch层次分类
result = Birch(n_clusters=3).fit_predict(x) # 聚类并预测
plt.figure(figsize=(5,5)) # 设置画布大小
plt.title("Birch") # 设置标题
plt.scatter(x[:,0],x[:,1],c=result) # 绘制散点图
plt.show() # 显示图像

# DBSCAN聚类
result = DBSCAN(eps=0.1).fit_predict(x) # 聚类并预测
plt.figure(figsize=(5,5)) # 设置画布大小
plt.title("DBSCAN") # 设置标题
plt.scatter(x[:,0],x[:,1],c=result) # 绘制散点图
plt.show() # 显示图像