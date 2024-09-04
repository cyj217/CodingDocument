import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

data = pd.read_excel('5019.xlsx',sheet_name='Sheet1')

X_train = data[['a', 'b', 'c']].values
y_train = data['f3'].values

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 创建多项式特征
poly_features = PolynomialFeatures(degree=1, interaction_only = False, include_bias = False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# 创建线性回归模型并拟合多项式特征的训练集数据
linear_reg = LinearRegression()
linear_reg.fit(X_train_poly, y_train)

# 预测测试集的输出
y_pred = linear_reg.predict(X_test_poly)

# 计算测试集的均方误差
mse_test = mean_squared_error(y_test, y_pred)
print("测试集的均方误差（MSE）：", mse_test)

# 获取模型的系数和截距
coefficients = linear_reg.coef_
intercept = linear_reg.intercept_

# 创建多项式对象
polynomial = np.poly1d(coefficients)  # 排除截距项

# 输出多项式表达式
expression = "y = " + str(intercept)
for i, coef in enumerate(coefficients, start=1):
    expression += " + " + str(coef) + " * x" + str(i)

print("多项式表达式:")
print(expression)

# 获取特征名称
feature_names = poly_features.get_feature_names_out(input_features=['a', 'b', 'c'])

# 打印输出特征名称
print("转换后的输出特征名称:")
print(feature_names)
