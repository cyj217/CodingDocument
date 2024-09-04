import matplotlib.pyplot as plt #导入matplotlib库
from sklearn.linear_model import LinearRegression #导入线性回归模型
from sklearn.preprocessing import PolynomialFeatures #导入多项式回归模型
import numpy as np #导入numpy库
import pandas as pd #导入pandas库

data = pd.read_excel('experiment5.xlsx',sheet_name='Sheet3')

X = data[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t']].values
y = data['avg'].values

# 创建多项式特征矩阵
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)

# 创建线性回归模型并拟合数据
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# 打印回归系数和截距
coefficients = lin_reg.coef_
intercept = lin_reg.intercept_

expression = f"{intercept:.2f}"
for i in range(1, len(coefficients)):
    expression += f" + {coefficients[i]:.2f} * x{i}"

print("回归模型的表达式：")
print(expression)