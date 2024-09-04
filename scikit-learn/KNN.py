from sklearn.neighbors import KNeighborsClassifier # 导入KNN分类器
from sklearn.model_selection import train_test_split  # 导入数据集划分工具
from sklearn import datasets # 导入数据集
import numpy as np # 导入numpy库
from mpl_toolkits.mplot3d import Axes3D # 导入3D坐标轴
import matplotlib.pyplot as plt # 导入matplotlib库
from sklearn import neighbors # 导入KNN分类器

# k近邻算法分类分析
iris = datasets.load_iris() # 导入鸢尾花数据集
X = iris["data"] # 获取数据集的特征
y = iris["target"] # 获取数据集的标签  
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2) # 划分数据集
knn = KNeighborsClassifier() # 实例化KNN分类器
knn.fit(X_train,y_train) # 训练模型
iris_y_predict = knn.predict(X_test) #得到预测结果
probility = knn.predict_proba(X_test) # 得到预测概率
neighborpoint = knn.kneighbors([X_test[-1]],5,False) # 得到最近的5个点
score = knn.score(X_test,y_test,sample_weight=None) # 得到模型的准确率
print('iris_y_test=') # 打印真实标签
print(y_test) # 打印真实标签
print('Accuracy:',score) # 打印准确率
print('neighborpoint of last test sample:',neighborpoint) # 打印最近的5个点
print('probility:',probility) # 打印预测概率
print(X_test[-1].reshape(-1,1)) # 打印最后一个测试样本

# k近邻算法回归分析
np.random.seed(0) # 设置随机种子
X = np.sort(5 * np.random.rand(40,1),axis=0) # 生成40个随机数
T = np.linspace(0,5,500)[:,np.newaxis] # 生成500个随机数
y = np.sin(X).ravel() # 生成正弦函数
y[::5] += 1 * (0.5 - np.random.rand(8)) # 生成噪声
n_neighbors = 5 # 设置最近邻的个数
for i, weights in enumerate(['uniform','distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors,weights=weights) # 实例化KNN回归器
    y_ = knn.fit(X,y).predict(T) # 训练模型并预测
    plt.subplot(2,1,i+1) # 绘制子图
    plt.scatter(X,y,color='darkorange',label='data') # 绘制散点图
    plt.plot(T,y_,color='navy',label='prediction') # 绘制预测曲线
    plt.axis('tight') # 设置坐标轴
    plt.legend() # 设置图例
    plt.title("KNeighborsRegressor(k = %i,weights = '%s')"%(n_neighbors,weights)) # 设置标题
plt.tight_layout() # 设置子图间距
plt.show() # 显示图像

# 绘制二维/三维散点图
iris = datasets.load_iris() # 导入鸢尾花数据集
X = iris["data"] # 获取数据集的特征
y = iris["target"] # 获取数据集的标签  
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1) # 划分数据集
knn = KNeighborsClassifier() # 实例化KNN分类器
knn.fit(X_train,y_train) # 训练模型
plt.scatter(X_train[:,0],X_train[:,1],marker='o',c=y_train) # 绘制二维散点图
plt.scatter(X_test[:,0],X_test[:,1],marker='+',c=y_test) # 绘制二维散点图
plt.show() # 显示图像
fig = plt.figure() # 实例化图像
ax = fig.add_subplot(111,projection='3d') # 绘制三维坐标轴
ax.scatter(X_train[:,1],X_train[:,2],X_train[:,3],c=y_train) # 绘制三维散点图
ax.scatter(X_test[:,1],X_test[:,2],X_test[:,3],c=y_test,marker='+') # 绘制三维散点图
ax.set_xlabel('X',fontdict={'size':15,'color':'red'}) # 设置X轴标签
ax.set_ylabel('Y',fontdict={'size':15,'color':'red'}) # 设置Y轴标签
ax.set_zlabel('Z',fontdict={'size':15,'color':'red'}) # 设置Z轴标签
# 获取数据集的最小值和最大值
x_min, x_max = X_train[:, 1].min() - .1, X_train[:, 1].max() + .1
y_min, y_max = X_train[:, 2].min() - .1, X_train[:, 2].max() + .1
z_min, z_max = X_train[:, 3].min() - .1, X_train[:, 3].max() + .1
# 设置三维坐标轴范围
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)
plt.show() # 显示图像
