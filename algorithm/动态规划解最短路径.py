import pandas as pd # 导入pandas库
import numpy as np # 导入numpy库

# 定义前后两阶段的节点距离
df1 = pd.DataFrame(np.array([[10,20]]),index=["A"],columns=['B1','B2']) # A--B的距离
df2 = pd.DataFrame(np.array([[30,10],[5,20]]),index=['B1','B2'],columns=['C1','C2']) # B--C的距离
df3 = pd.DataFrame(np.array([[20],[10]]),index=['C1','C2'],columns=['D']) # C--D的距离

# 定义动态规划函数
def dp(df_from,df_to):
    from_node = df_to.index
    f = pd.Series(dtype='float64')
    g = []
    for j in from_node:
        m1 = df_to.loc[j]
        m2 = m1 + df_from
        m3 = m2.sort_values()
        f[j] = m3[0]
        g.append(m3.index[0])
    dc = pd.DataFrame()
    dc['v'] = f.values
    dc['n'] = g
    dc.index = f.index
    cv.append(dc)
    if len(start) > 0:
        df = start.pop()
        t = dp(dc['v'],df)
    else:
        return dc

# 定义起始节点
start = [df1]
cv = []
t1 = df3['D']
h1 = dp(df3['D'],df2)

# 输出最短路径
for m in range(len(cv)):
    xc = cv.pop()
    x1 = xc.sort_values(by='v')
    print(x1['n'].values[0],end='->')
