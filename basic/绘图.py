import matplotlib.pyplot as plt # 绘图库
import matplotlib as mpl # 绘图库
import numpy as np # 数组操作
from mpl_toolkits.mplot3d import Axes3D # 3D绘图
from matplotlib import cm # 颜色映射
from matplotlib import animation # 动画

mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 正常显示图像中的负号

# 绘制散点图
x = np.random.randint(low=2,high=10,size=10) # 生成10个2-10之间的随机整数
y = np.random.randint(low=2,high=10,size=10) # 生成10个2-10之间的随机整数
plt.scatter(x,y) # 绘制散点图
plt.title('散点图') # 设置标题
plt.xlabel('x轴') # 设置x轴标签
plt.ylabel('y轴') # 设置y轴标签
plt.show() # 显示图像

# 绘制折线图
x = np.linspace(start=0,stop=30,num=300) # 生成0-30之间的300个等差数
y = np.sin(x) # 计算x对应的sin值
plt.plot(x,y) # 绘制折线图
plt.title('折线图') # 设置标题
plt.xlabel('x轴') # 设置x轴标签
plt.ylabel('y轴') # 设置y轴标签
plt.show() # 显示图像

# 绘制柱状图
x = ['a','b','c','d'] # x轴数据
y = [3,5,7,9]  # y轴数据
plt.bar(x,y,width=0.5) # 绘制柱状图
plt.title('柱状图') # 设置标题
plt.xlabel('x轴') # 设置x轴标签
plt.ylabel('y轴') # 设置y轴标签
plt.show() # 显示图像

# 绘制饼图
x = ['a','b','c','d'] # x轴数据
y = [3,5,7,9]  # y轴数据
plt.pie(y,labels=x) # 绘制饼图
plt.title('饼图') # 设置标题
plt.show() # 显示图像

# 绘制直方图
x = np.random.normal(loc=0,scale=1,size=1000) # 生成1000个符合正态分布的随机数
plt.hist(x=x,bins=50) # 绘制直方图
plt.title('直方图') # 设置标题
plt.xlabel('x轴') # 设置x轴标签
plt.ylabel('y轴') # 设置y轴标签
plt.show() # 显示图像

# 绘制箱线图
x = np.random.normal(loc=0,scale=1,size=1000) # 生成1000个符合正态分布的随机数
plt.boxplot(x) # 绘制箱线图
plt.title('箱线图') # 设置标题
plt.xlabel('x轴') # 设置x轴标签
plt.ylabel('y轴') # 设置y轴标签
plt.show() # 显示图像

# 绘制雷达图
x = np.linspace(start=0,stop=2*np.pi,num=5) # 生成0-2π之间的5个等差数
y = np.sin(x) # 计算x对应的sin值
plt.polar(x,y) # 绘制雷达图
plt.title('雷达图') # 设置标题
plt.show() # 显示图像

# 绘制正弦曲线
x = np.linspace(start=0,stop=30,num=300) # 生成0-30之间的300个等差数
y = np.sin(x) # 计算x对应的sin值
plt.plot(x,y,color='r',marker='d',linestyle='--',linewidth=2,alpha=0.8) # 绘制正弦曲线
plt.title('颜色: 红, 标记: 菱形, 线型: 虚线, 线宽: 2, 透明度: 0.8') # 设置标题
plt.xlabel('x轴') # 设置x轴标签
plt.ylabel('y轴') # 设置y轴标签
plt.show() # 显示图像

# 组合图
x1 = np.linspace(start=0,stop=30,num=300) # 生成0-30之间的300个等差数
y1 = np.sin(x1) # 计算x对应的sin值
x2 = np.random.randint(low=0,high=10,size=10) # 生成10个0-10之间的随机整数
y2 = np.random.randint(low=0,high=10,size=10)/10 # 生成10个0-1之间的随机数
plt.plot(x1,y1,color='b',label='line plot') # 绘制正弦曲线
plt.scatter(x2,y2,color='r',label='scatter plot') # 绘制散点图
plt.title('组合图') # 设置标题
plt.legend(loc='best') # 设置图例
plt.show() # 显示图像

# 子图
fig = plt.figure(figsize=(7,7)) # 创建画布
ax1 = fig.add_subplot(2,2,1) # 创建子图1
ax2 = fig.add_subplot(2,2,2) # 创建子图2
ax3 = fig.add_subplot(2,2,3) # 创建子图3
ax4 = fig.add_subplot(2,2,4) # 创建子图4
x = np.linspace(start=0,stop=30,num=300) # 生成0-30之间的300个等差数
y = np.sin(x) # 计算x对应的sin值
ax1.plot(x,y) # 绘制正弦曲线
ax1.set_title('子图1') # 设置标题
x = np.random.randint(low=2,high=10,size=10) # 生成10个2-10之间的随机整数
y = np.random.randint(low=2,high=10,size=10) # 生成10个2-10之间的随机整数
ax2.scatter(x,y) # 绘制散点图
ax2.set_title('子图2') # 设置标题
x = np.random.normal(loc=0,scale=1,size=1000) # 生成1000个符合正态分布的随机数
ax3.hist(x=x,bins=50) # 绘制直方图
ax3.set_title('子图3') # 设置标题
x1 = np.linspace(start=0,stop=30,num=300) # 生成0-30之间的300个等差数
y1 = np.sin(x1) # 计算x对应的sin值
x2 = np.random.randint(low=0,high=10,size=10) # 生成10个0-10之间的随机整数
y2 = np.random.randint(low=0,high=10,size=10)/10 # 生成10个0-1之间的随机数
ax4.plot(x1,y1,color='b',label='line plot') # 绘制正弦曲线
ax4.scatter(x2,y2,color='r',label='scatter plot') # 绘制散点图
ax4.set_title('子图4') # 设置标题
plt.show() # 显示图像

# 3D曲线图
fig = plt.figure() # 创建画布
ax = fig.add_subplot(111,projection='3d') # 创建3D坐标轴
theta = np.linspace(-4*np.pi,4*np.pi,100) # 生成-4π到4π之间的100个等差数
z = np.linspace(-2,2,100) # 生成-2到2之间的100个等差数
r = z**2+1 # 计算r的值
x = r*np.sin(theta) # 计算x的值
y = r*np.cos(theta) # 计算y的值
ax.plot(x,y,z) # 绘制3D曲线图
plt.show() # 显示图像

# 3D散点图
fig = plt.figure() # 创建画布
ax = fig.add_subplot(111,projection='3d') # 创建3D坐标轴
x1 = np.random.random(100)*20 # 生成100个0-20之间的随机数
y1 = np.random.random(100)*20 # 生成100个0-20之间的随机数
z1 = x1+y1 # 计算z1的值
ax.scatter(x1,y1,z1,c='r',marker='o') # 绘制3D散点图
x2 = np.random.random(100)*20 # 生成100个0-20之间的随机数
y2 = np.random.random(100)*20 # 生成100个0-20之间的随机数
z2 = x2+y2 # 计算z2的值
ax.scatter(x2,y2,z2,c='b',marker='^') # 绘制3D散点图
plt.show() # 显示图像

# 3D曲面图
fig = plt.figure() # 创建画布
ax = fig.add_subplot(111,projection='3d') # 创建3D坐标轴
x = np.arange(-5,5,0.25) # 生成-5到5之间的0.25步长的等差数
y = np.arange(-5,5,0.25) # 生成-5到5之间的0.25步长的等差数
x,y = np.meshgrid(x,y) # 生成网格点坐标矩阵
z = np.sin(np.sqrt(x**2+y**2)) # 计算z的值
surf = ax.plot_surface(x,y,z,cmap=cm.coolwarm) # 绘制3D曲面图
plt.show() # 显示图像

class car(): # 定义car类
    def __init__(self,marker): # 初始化函数
        self.x = 1 # 初始化x坐标
        self.y = 1 # 初始化y坐标
        self.marker = marker # 初始化标记
    def move(self): # 定义move方法
        self.x = self.x+np.random.randint(low=-1,high=2,size=1)[0] # 随机生成x坐标
        self.y = self.y+np.random.randint(low=-1,high=2,size=1)[0] # 随机生成y坐标
        self.x = self.x if self.x > 0 else 0 # 防止越界
        self.x = self.x if self.x < 10 else 10 # 防止越界
        self.y = self.y if self.y > 0 else 0 # 防止越界
        self.y = self.y if self.y < 10 else 10 # 防止越界
cars = [car(marker='o'),car(marker='^'),car(marker='*')] # 创建3个car对象
fig = plt.figure() # 创建画布
i = list(range(1,1000)) # 生成1-1000的列表
def update(i): # 定义update函数
    plt.clf() # 清除图层
    for c in cars: # 遍历car对象
        c.move() # 调用move方法，移动1步
        x = c.x # 获取x坐标
        y = c.y # 获取y坐标
        marker = c.marker # 获取标记
        plt.xlim(0,10) # 限制x区域
        plt.ylim(0,10) # 限制y区域
        plt.scatter(x,y,marker=marker) # 绘制卡车
    return # 返回
ani = animation.FuncAnimation(fig,update) # 创建动画
plt.show() # 显示图像
