import matplotlib.pyplot as plt #导入matplotlib库
from sklearn.linear_model import LinearRegression #导入线性回归模型
from sklearn.preprocessing import PolynomialFeatures #导入多项式回归模型
import numpy as np #导入numpy库
import pandas as pd #导入pandas库
from sklearn.ensemble import RandomForestRegressor #导入随机森林回归模型

data = pd.read_excel('experiment5.xlsx',sheet_name='Sheet3')

X = data[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t']].values
y = data['stdv'].values

X_max = np.max(X)
X_min = np.min(X)
y_max = np.max(y)
y_min = np.min(y)

X_normalized = (X - X_min) / (X_max - X_min)
y_normalized = (y - y_min) / (y_max - y_min)

print (X_normalized)
print (y_normalized)