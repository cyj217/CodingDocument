import matplotlib.pyplot as plt #导入matplotlib库
from sklearn.linear_model import LinearRegression #导入线性回归模型
from sklearn.preprocessing import PolynomialFeatures #导入多项式回归模型
import numpy as np #导入numpy库
import pandas as pd #导入pandas库
from sklearn.ensemble import RandomForestRegressor #导入随机森林回归模型

data = pd.read_excel('experiment5.xlsx',sheet_name='Sheet2')

print (data)
print (data.columns)
