import numpy as np # 导入numpy库
import matplotlib.pyplot as plt # 导入绘图库
from sklearn import tree # 导入决策树模型
from sklearn.datasets import load_iris # 导入数据集iris
from sklearn.tree import DecisionTreeClassifier, plot_tree # 导入决策树分类器和绘图函数

# 训练决策树模型
iris = load_iris() # 加载数据集
X = iris.data[:,[1,2]] # 为方便绘图，仅选取数据集的两个特征
y = iris.target # 获取数据集的标签
clf = tree.DecisionTreeClassifier(max_depth=4) # 设定决策树的最大深度为4
clf=clf.fit(X,y) # 训练模型
x_min,x_max = X[:,0].min()-1,X[:,0].max()+1 # 获取x轴的最大值和最小值
y_min,y_max = X[:,1].min()-1,X[:,1].max()+1 # 获取y轴的最大值和最小值

# 决策树分类结果可视化
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1)) # 生成网格点坐标矩阵
print(xx) # 第一列花萼长度数据按h取等分作为行，并复制多行得到xx网格矩阵
print(yy) # 第二列花萼宽度数据按h取等分作为列，并复制多列得到yy网格矩阵
Z = clf.predict(np.c_[xx.ravel(),yy.ravel()]) # 调用ravel()函数将xx和yy的两个矩阵变成一维数组，并用np.c_[]函数组合成二维数组进行预测
print('111',xx.ravel()) # 查看xx的一维数组
print('222',yy.ravel()) # 查看yy的一维数组
Z = Z.reshape(xx.shape) # reshape()函数修改形状，将其Z转换为两个特征(长度和宽度)
C = plt.contourf(xx,yy,Z,alpha=0.75,cmap=plt.cm.cool) #plt.contourf绘制等高线填充图
plt.scatter(X[:,0],X[:,1],c=y,alpha=0.8) # 绘制散点图
plt.show() # 显示图像

# 决策树可视化
iris = load_iris() # 加载数据集
X = iris.data[:,[1,2]] # 为方便绘图，仅选取数据集的两个特征
y = iris.target # 获取数据集的标签
clf = tree.DecisionTreeClassifier(max_depth=4) # 设定决策树的最大深度为4
clf=clf.fit(X,y) # 训练模型
plt.figure(figsize=(10,8)) # 设置图像大小
plot_tree(clf,filled=True) # 绘制决策树
plt.show() # 显示图像
