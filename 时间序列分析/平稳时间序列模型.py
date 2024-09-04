import statsmodels.tsa.api as smtsa # 导入时间序列模块
import numpy as np
import matplotlib.pyplot as plt

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
n = 100
ma = np.r_[1,0]
ar11 = np.r_[1, -0.6]
ar12 = np.r_[1, -1]
ar13 = np.r_[1, 1.8]
ar14 = np.r_[1, -1, -0.3]
np.random.seed(231)
ar1 = smtsa.arma_generate_sample(ar=ar11, ma=ma, nsample=n)
np.random.seed(232)
ar2 = smtsa.arma_generate_sample(ar=ar12, ma=ma, nsample=n)
np.random.seed(233)
ar3 = smtsa.arma_generate_sample(ar=ar13, ma=ma, nsample=n)
np.random.seed(234)
ar4 = smtsa.arma_generate_sample(ar=ar14, ma=ma, nsample=n)
fig = plt.figure(figsize=(12,4), dpi=150)
ax1 = fig.add_subplot(221)
ax1.plot(ar1, color='blue',  linestyle='-')
ax1.set_xlabel("a", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax2 = fig.add_subplot(222)
ax2.plot(ar2, color='blue',  linestyle='-')
ax2.set_xlabel("b", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax3 = fig.add_subplot(223)
ax3.plot(ar3, color='blue',  linestyle='-')
ax3.set_xlabel("c", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax4 = fig.add_subplot(224)
ax4.plot(ar4, color='blue',  linestyle='-')
ax4.set_xlabel("d", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.tight_layout()
plt.show()

# 绘制自相关图
n = 200
ma = np.r_[1,0]
ar11 = np.r_[1, -0.8]
ar12 = np.r_[1, 0.7]
ar13 = np.r_[1, 0.2, -0.3]
ar14 = np.r_[1, -0.2, 0.3]
np.random.seed(281)
ar1 = smtsa.arma_generate_sample(ar=ar11, ma=ma, nsample=n)
np.random.seed(282)
ar2 = smtsa.arma_generate_sample(ar=ar12, ma=ma, nsample=n)
np.random.seed(283)
ar3 = smtsa.arma_generate_sample(ar=ar13, ma=ma, nsample=n)
np.random.seed(284)
ar4 = smtsa.arma_generate_sample(ar=ar14, ma=ma, nsample=n)
fig = plt.figure(figsize=(12,4), dpi=150)
ax1 = fig.add_subplot(221)
ACF(ar1, lag=30)
ax1.set_xlabel("1", fontsize=17)
ax2 = fig.add_subplot(222)
ACF(ar2, lag=30)
ax2.set_xlabel("2", fontsize=17)
ax3 = fig.add_subplot(223)
ACF(ar3, lag=30)
ax3.set_xlabel("3", fontsize=17)
ax4 = fig.add_subplot(224)
ACF(ar4, lag=30)
ax4.set_xlabel("4", fontsize=17)
fig.tight_layout()
plt.show()

# 绘制偏自相关图
fig = plt.figure(figsize=(12,6), dpi=150)
ax1 = fig.add_subplot(221)
PACF(ar1, lag=30)
ax1.set_xlabel("1", fontsize=17)
ax2 = fig.add_subplot(222)
PACF(ar2, lag=30)
ax2.set_xlabel("2", fontsize=17)
ax3 = fig.add_subplot(223)
PACF(ar3, lag=30)
ax3.set_xlabel("3", fontsize=17)
ax4 = fig.add_subplot(224)
PACF(ar4, lag=30)
ax4.set_xlabel("4", fontsize=17)
fig.tight_layout()
plt.show()

# 绘制偏自相关图
n =100
ar = np.r_[1,0]
ma1 = np.r_[1, -0.5]
ma2 = np.r_[1, -0.25, 0.5]
np.random.seed(216)
ma11 = smtsa.arma_generate_sample(ar=ar, ma=ma1, nsample=n)
np.random.seed(217)
ma22 = smtsa.arma_generate_sample(ar=ar, ma=ma2, nsample=n)
fig = plt.figure(figsize=(12,4), dpi=150)
ax1 = fig.add_subplot(121)
PACF(ma11, lag=30)
ax1.set_xlabel("1", fontsize=17)
ax2 = fig.add_subplot(122)
PACF(ma22, lag=30)
ax2.set_xlabel("2", fontsize=17)
fig.tight_layout()
plt.show()
