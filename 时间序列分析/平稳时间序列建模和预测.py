import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.tsa.api as smtsa
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# 定义绘制自相关图的函数
from statsmodels.graphics.tsaplots import acf
def ACF(ts, lag=20):
    lag_acf = acf(ts, nlags=lag, fft=False)
    plt.vlines(x=list(range(lag+1)), ymin=np.zeros(lag+1), ymax=lag_acf, linewidth=2.0, color='black')
    plt.axhline(y=0, linestyle=':', color='blue')
    plt.axhline(y=-1.96/np.sqrt(len(ts)), linestyle='--', color='red')
    plt.axhline(y=1.96/np.sqrt(len(ts)), linestyle='--', color='red')
    plt.title('')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("lag", fontsize=17)
    plt.ylabel("ACF", fontsize=17)
    plt.tight_layout()

# 定义绘制偏自相关图的函数
from statsmodels.graphics.tsaplots import pacf
def PACF(ts, lag=20):
    lag_pacf = smtsa.pacf(ts, nlags=lag)
    plt.vlines(x=list(range(lag+1)), ymin=np.zeros(lag+1), ymax=lag_pacf, linewidth=2.0, color='black')
    plt.axhline(y=0, linestyle=':', color='blue')
    plt.axhline(y=-1.96/np.sqrt(len(ts)), linestyle='--', color='red')
    plt.axhline(y=1.96/np.sqrt(len(ts)), linestyle='--', color='red')
    plt.title('')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("lag", fontsize=17)
    plt.ylabel("PACF", fontsize=17)
    plt.tight_layout()

# 绘制时序图
qhcpi_df = pd.read_csv('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第3章数据资源/cpi.csv', usecols = ['Date', 'QHCPI'], index_col = 0)
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ax.plot(qhcpi_df, marker="o", linestyle="-", color='blue')
ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
ax.set_ylabel("青海省居民消费指数", fontsize=17)
ax.set_xlabel("时间", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()
# 白噪声检验
acorr_ljungbox(qhcpi_df, lags = [6, 12], boxpierce = False, return_df = True)
print(acorr_ljungbox(qhcpi_df, lags = [6, 12], boxpierce = False, return_df = True))
# 自相关图和偏相关图定阶
fig = plt.figure(figsize=(12,4), dpi=150)
ax1 = fig.add_subplot(121)
ACF(qhcpi_df, lag=16)
ax2 = fig.add_subplot(122)
PACF(qhcpi_df, lag=8)
plt.show() # 由自相关图和偏自相关图可知，该序列为AR(2)序列
# 参数估计
qh_est = ARIMA(qhcpi_df, order=(2,0,0)).fit()
print(qh_est.summary().tables[1])

# 绘制时序图
jtsgs_df = pd.read_csv('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第3章数据资源/SGS.csv', usecols = ['year', 'JTSGS'], index_col = 0)
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ax.plot(jtsgs_df, marker="o", linestyle="-", color='blue')
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.set_ylabel("交通事故数", fontsize=17)
ax.set_xlabel("年份", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()
# 白噪声检验
acorr_ljungbox(jtsgs_df, lags = [1, 2, 3], boxpierce = True, return_df = True)
print(acorr_ljungbox(jtsgs_df, lags = [1, 2, 3], boxpierce = True, return_df = True))
# 自相关图和偏相关图定阶
fig = plt.figure(figsize=(12,4), dpi=150)
ax1 = fig.add_subplot(121)
ACF(jtsgs_df, lag=24)
ax2 = fig.add_subplot(122)
PACF(jtsgs_df, lag=24)
plt.show() # 由自相关图和偏自相关图可知，该序列为MA(1)序列
# 参数估计
jtsgs_est = ARIMA(jtsgs_df, order=(0,0,1)).fit()
print(jtsgs_est.summary().tables[1])

# 绘制时序图
huozai_df = pd.read_csv('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第3章数据资源/huozhai.csv', usecols = ['year', 'fire'], index_col = 0)
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ax.plot(huozai_df, marker="o", linestyle="-", color='blue')
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.set_ylabel("火灾数", fontsize=17)
ax.set_xlabel("年份", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()
# 白噪声检验
acorr_ljungbox(huozai_df, lags = [5, 10], boxpierce = True, return_df = True)
print(acorr_ljungbox(huozai_df, lags = [5, 10], boxpierce = True, return_df = True))
# 自相关图和偏相关图定阶
fig = plt.figure(figsize=(12,4), dpi=150)
ax1 = fig.add_subplot(121)
ACF(huozai_df, lag=24)
ax2 = fig.add_subplot(122)
PACF(huozai_df, lag=24)
plt.show() # 由自相关图和偏自相关图可知，该序列为ARMA(2,1)序列
# 参数估计
huozai_est = ARIMA(huozai_df, order=(2,0,1)).fit()
print(huozai_est.summary().tables[1])
