import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

data1 = pd.read_excel('experiment.xlsx',sheet_name='Sheet5')
data2 = pd.read_excel('test.xlsx',sheet_name='Sheet3')

X_train = data1[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t']].values
y_train = data1['avg'].values

X_test = data2[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t']].values
y_test = data2['avg'].values

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
feature_names = poly_features.get_feature_names_out(input_features=['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t'])

# 打印输出特征名称
print("转换后的输出特征名称:")
print(feature_names)

new_data = np.array([[20, 5, 40, 773, 27, 5, 53, 798, 20]])
new_data_poly = poly_features.transform(new_data)
predicted_y = linear_reg.predict(new_data_poly)
print("预测的输出:", predicted_y)