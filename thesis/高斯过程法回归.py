import numpy as np #导入numpy库
import pandas as pd #导入pandas库
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data1 = pd.read_excel('experiment.xlsx',sheet_name='Sheet5')
data2 = pd.read_excel('test.xlsx',sheet_name='Sheet3')

X_train = data1[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t']].values
y_train = data1['stdv'].values

X_test = data2[['x_up', 'z_up', 'l_up', 'p_up', 'x_down', 'z_down', 'l_down', 'p_down','t']].values
y_test = data2['stdv'].values

# 创建高斯过程回归模型
kernel_linear = ConstantKernel() * RBF() + WhiteKernel()
gpr_linear = GaussianProcessRegressor(kernel=kernel_linear, alpha=0.1)

# 拟合训练数据
gpr_linear.fit(X_train, y_train)

# 预测测试集的输出
y_pred, y_std = gpr_linear.predict(X_test, return_std=True)

# 计算测试集的均方误差
mse_test = mean_squared_error(y_test, y_pred)
print("测试集的均方误差（MSE）：", mse_test)
