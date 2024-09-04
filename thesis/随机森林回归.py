import matplotlib.pyplot as plt #导入matplotlib库
import numpy as np #导入numpy库
import pandas as pd #导入pandas库
from sklearn.ensemble import RandomForestRegressor #导入随机森林回归模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data1 = pd.read_excel('experiment.xlsx',sheet_name='Sheet5')
data2 = pd.read_excel('test.xlsx',sheet_name='Sheet3')

X_train = data1[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t']].values
y_train = data1['stdv'].values

X_test = data2[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t']].values
y_test = data2['stdv'].values

# 构建随机森林模型
rf = RandomForestRegressor(n_estimators=20, random_state=42)

# 拟合模型
rf.fit(X_train, y_train)

# 预测
y_pred = rf.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print("均方误差（MSE）:", mse)

# 打印特征重要性
importances = rf.feature_importances_
print("特征重要性：", importances)
