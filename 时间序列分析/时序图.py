import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

# 字体背景设置
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = "simsun"
plt.style.use('ggplot')

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
    plt.show()

# 中国GDP时序图
GDP = pd.read_csv('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/ChinaGDP.csv', index_col = 0, squeeze = True)
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ax.plot(GDP, marker="o", linestyle="-", color='blue')
ax.set_ylabel("国内生产总值（亿元）", fontsize=17)
ax.set_xlabel("年份", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()
# 相邻年度散点图
GDPy = GDP[1::]
GDPx = GDP[:-1]
fig=plt.figure(figsize=(12,4), dpi=150)
ax=fig.add_subplot(111)
ax.scatter(x=GDPx,y=GDPy,marker="o",color='blue')
ax.set_ylabel("当年GDP", fontsize=17)
ax.set_xlabel("上一年GDP", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()

# Dubic气温时序图
Dubic = np.loadtxt('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/DubicCity.txt')
Index = pd.date_range(start="1964-01", end="1976-01", freq="M")
Dubic_ts = pd.Series(Dubic, index=Index) # 建立时间序列
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ax.plot(Dubic_ts, marker="o", linestyle="-", color='blue')
ax.set_ylabel("气温", fontsize=17)
ax.set_xlabel("年份", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()

# 洛杉矶年降水量时序图
LosAngeles = np.loadtxt('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/LosAngeles.txt')
LostTime = pd.date_range(start="1880", end="1995", freq="Y")
LosAngeles_ts = pd.Series(LosAngeles, index=LostTime)
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ax.plot(LosAngeles_ts, marker="o", linestyle="-", color='blue')
ax.set_ylabel("降水量", fontsize=17)
ax.set_xlabel("年份", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()
# 相邻年度散点图
Losx = LosAngeles_ts[1::]
Losy = LosAngeles_ts[:-1]
fig=plt.figure(figsize=(12,4), dpi=150)
ax=fig.add_subplot(111)
ax.scatter(x=Losx,y=Losy,marker="o",color='blue')
ax.set_ylabel("当年降水量", fontsize=17)
ax.set_xlabel("上一年降水量", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()
# 自相关图
LosRain = np.loadtxt('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/LosAngeles.txt')
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ACF(LosRain, lag=100)

# 新西兰出国旅游目的地时序图
nzt = pd.read_csv('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/NZTravellersDestination.csv')
nzt.head(2)
print(nzt.head(2))
# 中国游客
NtoC = pd.read_csv('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/NZTravellersDestination.csv', usecols=['Date','China'], parse_dates=['Date'], index_col='Date')
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ax.plot(NtoC, marker="o", linestyle="-", color='blue')
ax.set_ylabel("游客数", fontsize=17)
ax.set_xlabel("年份", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()
# 其他国家游客
ucls = ['Date', 'China', 'India', 'UK', 'US']
NtoF = pd.read_csv('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/NZTravellersDestination.csv', usecols=ucls, parse_dates=['Date'], index_col='Date')
Date = NtoF.index
NtoC = NtoF.China.values
NtoI = NtoF.India.values
NtoU = NtoF.UK.values
NtoUS = NtoF.US.values
nex,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True,figsize=(12,6), dpi=150)
ax1.plot(Date, NtoC, marker="o", linestyle="-", color='red')
ax1.set_ylabel("中国", fontsize=17)
ax2.plot(Date, NtoI, marker="o", linestyle="-", color='blue')
ax2.set_ylabel("印度", fontsize=17)
ax3.plot(Date, NtoU, marker="o", linestyle="-", color='yellow')
ax3.set_ylabel("英国", fontsize=17)
ax4.plot(Date, NtoUS, marker="o", linestyle="-", color='green')
ax4.set_ylabel("美国", fontsize=17)
ax4.set_xlabel("年份", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()
# 自相关图
NtoCh = pd.read_csv('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/NZTravellersDestination.csv', usecols=['China'])
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ACF(NtoCh['China'], lag=100)

# 北京商品住宅施工面积累计值时序图
bjch = pd.read_csv('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/BJCH.csv', encoding = 'utf-8')
for f in bjch: # 线性插值
    bjch[f] = bjch[f].interpolate()
    bjch.dropna(inplace=True)
xlabs = bjch.Time
ticker_spacing = xlabs
ticker_spacing = 5
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ax.plot(xlabs, bjch.CCA, color='blue', marker='o')
ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_spacing))
ax.vlines(x=['2020/04','2020/09'], ymin=5230, ymax=6250, color="green", linestyle='--')
ax.hlines(y=[5250,6250], xmin='2020/04', xmax='2020/09', color="green", linestyle='--')
ax.set_ylabel("施工面积累计值", fontsize=17)
ax.set_xlabel("时间", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()
# 自相关图
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ACF(bjch['CCA'], lag=30)

# 宁夏回族自治区生产总值时序图
NingXiaGDP = pd.read_excel('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/ningxiaGDP.xlsx', index_col = 0, squeeze = True)
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ax.plot(NingXiaGDP, marker="o", linestyle="-", color='blue')
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.set_ylabel("宁夏生产总值(单位: 亿元)", fontsize=17)
ax.set_xlabel("年份", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()
# 自相关图
NxGDP = pd.read_excel('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/ningxiaGDP.xlsx')
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ACF(NxGDP['Ningxia'], lag=15)

# 洛杉矶最高最低月平均气温时序图
col = ["Date", "LosAngelesMax", "LosAngelesMin"]
LosTemp = pd.read_csv('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/LosTemp.csv', usecols=col, index_col=0)
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
LosTemp.plot(ax=ax, color=['red', 'blue'], marker='o')
ax.set_ylabel("温度", fontsize=17)
ax.set_xlabel("时间", fontsize=17)
ax.legend(loc=2, fontsize=12)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()
# 自相关图
LosTemp = pd.read_csv('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/LosTemp.csv')
fig = plt.figure(figsize=(12,4), dpi=150)
ax = fig.add_subplot(111)
ACF(LosTemp['LosAngelesMax'], lag=100)

# 白噪声检验/纯随机性检验
np.random.seed(15)
white_noise = np.random.randn(500)
fig = plt.figure(figsize=(12,4), dpi=150)
ax1 = fig.add_subplot(121)
ax1.plot(white_noise, color='blue', marker='o', linestyle='--')
ax1.set_ylabel("white_noise", fontsize=17)
ax1.set_xlabel("time", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax2 = fig.add_subplot(122)
ACF(white_noise, lag=100)
# Q统计量
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(white_noise, lags=[6,12], return_df=True)
print(acorr_ljungbox(white_noise, lags=[6,12], return_df=True))

# 时序图和自相关图进行平稳性的图检验，然后纯随机性检验
ratio = pd.read_csv('C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/时序数据集/第1章数据资源/Centenarians.csv', usecols=['Year','ratio'], parse_dates=['Year'], index_col='Year')
fig = plt.figure(figsize=(12,4), dpi=150)
ax1 = fig.add_subplot(121)
ax1.plot(ratio, color='blue', marker='o', linestyle='--')
ax1.set_ylabel("ratio", fontsize=17)
ax1.set_xlabel("year", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax2 = fig.add_subplot(122)
ACF(ratio['ratio'], lag=10)
acorr_ljungbox(ratio, lags=[5,10], boxpierce=True, return_df=True)
print(acorr_ljungbox(ratio, lags=[5,10], boxpierce=True, return_df=True))