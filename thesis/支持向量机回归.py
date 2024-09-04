import matplotlib.pyplot as plt #导入matplotlib库
import numpy as np #导入numpy库
import pandas as pd #导入pandas库
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data1 = pd.read_excel('experiment.xlsx',sheet_name='Sheet5')
data2 = pd.read_excel('test.xlsx',sheet_name='Sheet3')

X_train = data1[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t']].values
y_train = data1['avg'].values

X_test = data2[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t']].values
y_test = data2['avg'].values

# 创建SVM回归模型
svm = SVR(kernel='poly', degree=2)

# 拟合训练数据
svm.fit(X_train, y_train)

# 预测测试集
y_pred = svm.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print("均方误差（Mean Squared Error）：", mse)