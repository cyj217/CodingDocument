import numpy as np # 导入numpy库
import matplotlib.pyplot as plt # 导入matplotlib库
from sklearn import svm, datasets # 导入sklearn库中的svm,datasets模块
from sklearn.preprocessing import StandardScaler # 导入数据预处理模块
from sklearn.model_selection import GridSearchCV,train_test_split # 导入网格搜索、数据分割模块
from sklearn.svm import SVC # 导入支持向量机SVC模块

# SVM算法实现
iris = datasets.load_iris() # 导入数据集 
X = iris.data[:,:2] # 取前两维特征
y = iris.target # 标签
xlist1 = np.linspace(X[:,0].min(),X[:,0].max(),200) # 生成200个x1坐标点
xlist2 = np.linspace(X[:,1].min(),X[:,1].max(),200) # 生成200个x2坐标点
XGrid1,XGrid2 = np.meshgrid(xlist1,xlist2) # 生成网格点坐标矩阵
svc = svm.SVC(kernel='rbf',C=1,gamma=0.5,tol=1e-5,cache_size=1000).fit(X,y) # 非线性SVM:RBF核，超参数0.5，正则化系数1.SMO迭代精度1e-5，缓存大小1000MB
Z = svc.predict(np.vstack([XGrid1.ravel(),XGrid2.ravel()]).T) # 预测网格点坐标矩阵
Z = Z.reshape(XGrid1.shape) # 使之与输入的形状相同
plt.contourf(XGrid1,XGrid2,Z,cmap=plt.cm.hsv) # 填充等高线
plt.contour(XGrid1,XGrid2,Z,colors=('k',)) # 绘制等高线
plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k',linewidth=1.5,cmap=plt.cm.hsv) # 绘制样本点
plt.show() # 显示图像

# SVM算法优化
iris = datasets.load_iris() # 导入数据集
X = iris["data"] # 取特征
y = iris["target"] # 取标签
print(X[0:5,0:]) # 打印前5行数据
scaler = StandardScaler() # 标准化数据
X_std = scaler.fit_transform(X) # 特征标准化
print(X_std[0:5,0:]) # 打印标准化后的前5行数据
X_train,X_test,y_train,y_test = train_test_split(X_std,y,test_size=0.3) # 数据分割
svc = SVC(kernel='rbf',class_weight='balanced',) # 非线性SVM:RBF核，类别权重平衡
c_range = np.logspace(-5,15,11,base=2) # 生成等比数列
gamma_range = np.logspace(-9,3,13,base=2) # 生成等比数列
param_grid = [{'kernel':['rbf'],'C':c_range,'gamma':gamma_range}] # 设置网格搜索参数范围，cv=3,3折交叉验证
grid = GridSearchCV(svc,param_grid,cv=3) # 网格搜索
clf=grid.fit(X_train,y_train) # 训练模型
score = grid.score(X_test,y_test) # 计算测试集精度
print('accuracy is %s'%score) # 打印测试集精度