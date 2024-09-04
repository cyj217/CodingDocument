import matplotlib.pyplot as plt # 画图工具
from sklearn import datasets # 数据集
from sklearn.decomposition import PCA # 导入PCA算法包
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # 导入LDA算法包
from sklearn.datasets import make_swiss_roll # 导入瑞士卷数据集生成器
from mpl_toolkits.mplot3d import Axes3D # 导入3D坐标轴
from sklearn.decomposition import KernelPCA # 导入KPCA算法包
from sklearn.model_selection import GridSearchCV # 导入网格搜索包
import numpy as np # 数组工具
from sklearn.pipeline import Pipeline # 导入管道包
from sklearn.linear_model import LogisticRegression # 导入逻辑回归包
from sklearn.manifold import MDS # 导入MDS算法包
from sklearn.manifold import Isomap # 导入Isomap算法包
from sklearn.manifold import TSNE # 导入TSNE算法包

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # 忽略警告

# LDA和PCA二维投影
iris = datasets.load_iris() # 加载数据集
X = iris.data # 获取特征向量
y = iris.target # 获取标签
target_names = iris.target_names # 获取标签名称
pca = PCA(n_components=2) # 加载PCA算法，设置降维后主成分数目为2
X_r = pca.fit(X).transform(X) # 对原始数据进行降维，保存在X_r中
lda = LinearDiscriminantAnalysis(n_components=2) # 加载LDA算法，设置降维后主成分数目为2
X_r2 = lda.fit(X, y).transform(X) # 对原始数据进行降维，保存在X_r2中
print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_)) # 输出降维后各主成分的方差值占总方差值的比例
plt.figure() # 创建画布
colors = ['navy', 'turquoise', 'darkorange'] # 设置颜色
lw = 2 # 设置线宽
for color, i, target_name in zip(colors, [0, 1, 2], target_names): # 画出每种花的数据
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,label=target_name) # 画出每种花的数据
plt.legend(loc='best', shadow=False, scatterpoints=1) # 设置图例位置
plt.title('PCA of IRIS dataset') # 设置标题
plt.figure() # 创建画布
for color, i, target_name in zip(colors, [0, 1, 2], target_names): # 画出每种花的数据
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,label=target_name) # 画出每种花的数据
plt.legend(loc='best', shadow=False, scatterpoints=1) # 设置图例位置
plt.title('LDA of IRIS dataset') # 设置标题
plt.show() # 显示图像

# KPCA二维投影
X,t = make_swiss_roll(n_samples=1000,noise=0.2,random_state=42) # 生成瑞士卷数据集
lin_pca = KernelPCA(n_components=2,kernel='linear',fit_inverse_transform=True) # 线性核KPCA
rbf_pca = KernelPCA(n_components=2,kernel='rbf',gamma=0.0433,fit_inverse_transform=True) # 径向基核KPCA
sig_pca = KernelPCA(n_components=2,kernel='sigmoid',gamma=0.001,coef0=1,fit_inverse_transform=True) # sigmoid核KPCA
y = t > 6.9 # 设置标签
plt.figure(figsize=(11,4)) # 创建画布
for subplot, pca, title in ((131, lin_pca, 'Linear kernel'),(132, rbf_pca, 'RBF kernel, $\gamma=0.04$'),(133, sig_pca, 'Sigmoid kernel, $\gamma=10^{-3}, r=1$')): # 画出每种核的KPCA
    X_reduced = pca.fit_transform(X) # 对原始数据进行降维
    if subplot == 132: # 画出瑞士卷数据集
        X_reduced_rbf = X_reduced
    plt.subplot(subplot) # 设置子图
    plt.title(title, fontsize=14) # 设置标题
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot) # 画出降维后的数据
    plt.xlabel('$z_1$', fontsize=18) # 设置x轴标签
    if subplot == 131: # 设置y轴标签
        plt.ylabel('$z_2$', fontsize=18, rotation=0)
    plt.grid(True) # 设置网格
plt.show() # 显示图像

# 通过GridSearch寻找合适的核函数与参数
clf = Pipeline([("kpca", KernelPCA(n_components=2)),("log_reg", LogisticRegression())]) # 构建管道
param_grid = [{"kpca__gamma": np.linspace(0.03, 0.05, 10), "kpca__kernel": ["rbf", "sigmoid"]}] # 设置参数
grid_search = GridSearchCV(clf, param_grid, cv=3) # 设置网格搜索
grid_search.fit(X, y) # 进行网格搜索
print(grid_search.best_params_) # 输出最佳参数

# MDS/Isomap/t-SNE二维投影
X,t = make_swiss_roll(n_samples=1000,noise=0.2,random_state=42) # 生成瑞士卷数据集
mds = MDS(n_components=2,random_state=42) # MDS算法
X_reduced_mds = mds.fit_transform(X) # 对原始数据进行降维
isomap = Isomap(n_components=2) # Isomap算法
X_reduced_isomap = isomap.fit_transform(X) # 对原始数据进行降维
tsne = TSNE(n_components=2,random_state=42) # t-SNE算法
X_reduced_tsne = tsne.fit_transform(X) # 对原始数据进行降维
titles = ["MDS", "Isomap", "t-SNE"] # 设置标题
plt.figure(figsize=(11,4)) # 创建画布
for subplot, title, X_reduced in zip((131, 132, 133), titles, (X_reduced_mds, X_reduced_isomap, X_reduced_tsne)): # 画出每种算法的降维结果
    plt.subplot(subplot) # 设置子图
    plt.title(title, fontsize=14) # 设置标题
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot) # 画出降维后的数据
    plt.xlabel("$z_1$", fontsize=18) # 设置x轴标签
    if subplot == 131: # 设置y轴标签
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True) # 设置网格
plt.show() # 显示图像
