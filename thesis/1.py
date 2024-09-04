import matplotlib.pyplot as plt #导入matplotlib库
from sklearn.linear_model import LinearRegression #导入线性回归模型
from sklearn.preprocessing import PolynomialFeatures #导入多项式回归模型
import numpy as np #导入numpy库
import pandas as pd #导入pandas库
from sklearn.ensemble import RandomForestRegressor #导入随机森林回归模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_excel('experiment5.xlsx',sheet_name='Sheet2')

# 提取特征和目标变量
X = data[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down']]
y = data['stdv 30']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 拟合模型
rf.fit(X_train, y_train)

# 预测
y_pred = rf.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print("均方误差（MSE）:", mse)