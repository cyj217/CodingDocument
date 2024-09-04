import matplotlib.pyplot as plt #导入matplotlib库
import numpy as np #导入numpy库
import pandas as pd #导入pandas库
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

data1 = pd.read_excel('experiment.xlsx',sheet_name='Sheet3')
data2 = pd.read_excel('test.xlsx',sheet_name='Sheet3')

X_train = data1[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t']].values
y_train = data1['avg'].values

X_test = data2[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t']].values
y_test = data2['avg'].values

# 创建PCA对象，指定要保留的主成分个数
n_components = 2  # 假设你想要保留2个主成分
pca = PCA(n_components=n_components)

# 将输入数据进行主成分分析
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 查看主成分的方差解释比例
explained_variance_ratio = pca.explained_variance_ratio_
print("主成分的方差解释比例:")
for i, ratio in enumerate(explained_variance_ratio):
    print("主成分{}的方差解释比例: {:.2f}%".format(i+1, ratio * 100))

# 查看主成分的特征向量（主成分载荷）
components = pca.components_
print("主成分的特征向量:")
for i, component in enumerate(components):
    print("主成分{}的特征向量: {}".format(i+1, component))

# 创建多项式特征
poly_features = PolynomialFeatures(degree=2)
X_train_pca_poly = poly_features.fit_transform(X_train_pca)
X_test_pca_poly = poly_features.transform(X_test_pca)

# 创建线性回归模型
linear_reg = LinearRegression()

# 拟合多项式特征的训练集数据
linear_reg.fit(X_train_pca_poly, y_train)

# 预测测试集的输出
y_pred = linear_reg.predict(X_test_pca_poly)

# 计算测试集的均方误差
mse_test = mean_squared_error(y_test, y_pred)
print("测试集的均方误差（MSE）：", mse_test)

# 创建多项式特征
poly_features = PolynomialFeatures(degree=2)
X_train_pca_poly = poly_features.fit_transform(X_train_pca)

# 创建线性回归模型并拟合多项式特征的训练集数据
linear_reg = LinearRegression()
linear_reg.fit(X_train_pca_poly, y_train)

# 获取模型的系数和截距
coefficients = linear_reg.coef_
intercept = linear_reg.intercept_

# 创建多项式对象
polynomial = np.poly1d(coefficients[1:])  # 排除截距项

# 输出多项式表达式
expression = "y = " + str(intercept)
for i, coef in enumerate(coefficients[1:], start=1):
    expression += " + " + str(coef) + " * x" + str(i)

print("多项式表达式:")
print(expression)