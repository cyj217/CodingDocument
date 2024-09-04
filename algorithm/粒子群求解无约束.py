import numpy as np # 导入numpy库
import matplotlib.pyplot as plt # 导入matplotlib.pyplot库
from matplotlib import cm # 导入cm库
import matplotlib as mpl # 导入mpl库
from mpl_toolkits.mplot3d import Axes3D # 导入Axes3D库

# 绘制Rastrigin函数
X = np.arange(-5, 5, 0.1) # 生成-5到5的数据，步长为0.1
Y = np.arange(-5, 5, 0.1) # 生成-5到5的数据，步长为0.1
X, Y = np.meshgrid(X, Y) # 生成网格数据
A = 10 # 定义A
Z = 2*A+X**2-A*np.cos(2*np.pi*X)+Y**2-A*np.cos(2*np.pi*Y) # 目标函数
fig = plt.figure() # 创建一个绘图对象
ax = fig.add_subplot(111,projection='3d') # 创建3D坐标轴
surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm) # 绘制3D图像
plt.show() # 显示图像

# PSO算法解Rastrigin最小值
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体为黑体
mpl.rcParams['axes.unicode_minus'] = False # 设置正常显示符号
def fitness_func(X): # 计算粒子的适应度值/目标函数值，X的维度是size*2
    A = 10 # 定义A
    pi = np.pi # 定义pi
    x = X[:,0] # 获取X的第一列数据
    y = X[:,1] # 获取X的第二列数据
    return 2*A+x**2-A*np.cos(2*pi*x)+y**2-A*np.cos(2*pi*y) # 返回目标函数值
def velocity_update(V,X,pbest,gbest,c1,c2,w,max_val): #根据速度更新公式和每个粒子的速度
    size = X.shape[0] # 获取粒子群的大小
    r1 = np.random.random((size,1)) # 生成0-1之间的随机数
    r2 = np.random.random((size,1)) # 生成0-1之间的随机数
    V = w*V+c1*r1*(pbest-X)+c2*r2*(gbest-X) # 更新速度
    V[V>max_val] = max_val # 限制速度的最大值
    V[V<-max_val] = -max_val # 限制速度的最小值
    return V # 返回更新后的速度
def position_update(X,V): # 根据位置更新公式和每个粒子的速度
    return X+V # 返回更新后的位置
def pso(): # 定义PSO参数
    w = 1 # 惯性因子
    c1 = 2 # 自我认知学习因子
    c2 = 2 # 社会认知学习因子
    r1 = None # 生成0-1之间的随机数
    r2 = None # 生成0-1之间的随机数
    dim = 2 # 变量的个数
    size = 20 # 种群大小，即种群中粒子的个数
    iter_num = 1000 # 算法最大迭代次数
    max_val = 0.5 # 限制粒子的最大速度为0.5
    best_fitness = float(9e10) # 初始的适应度值，在迭代过程中不断减小这个值
    fitness_value_list = [] # 记录每次迭代过程中种群适应度值的变化
    X = np.random.uniform(-5,5,(size , dim)) # 初始化种群各个粒子的位置
    V = np.random.uniform(-0.5,0.5,size=(size , dim)) # 初始化种群各个粒子的速度
    p_fitness = fitness_func(X) # 计算种群各个粒子的初始适应度值
    g_fitness = p_fitness.min() # 计算种群的初始最优适应度值
    fitness_value_list.append(g_fitness) # 添加到记录中
    pbest = X # 初始化种群各个粒子的历史最优位置
    gbest = X[p_fitness.argmin()] # 初始化种群的最优位置
    for i in range(1,iter_num): # 开始迭代
        V = velocity_update(V,X,pbest,gbest,c1,c2,w,max_val) # 更新速度
        X = position_update(X,V) # 更新位置
        p_fitness = fitness_func(X) # 计算种群各个粒子的适应度值
        g_fitness = p_fitness.min() # 计算种群的最优适应度值
        for j in range(size): # 更新种群各个粒子的历史最优位置
            if p_fitness[j] > p_fitness[j]:
                pbest[j] = X[j]
                p_fitness[j] = p_fitness[j]
        if g_fitness > g_fitness: # 更新种群的最优位置
            gbest = X[p_fitness.argmin()]
            g_fitness = g_fitness
        fitness_value_list.append(g_fitness) # 记录最优迭代结果
        i += 1 # 迭代次数加1
    print("最优值是:%.5f" % fitness_value_list[-1]) # 输出迭代的结果
    print("最优解是: x=%.5f, y=%.5f" % gbest) # 输出迭代的结果
    plt.plot(fitness_value_list,color='r') # 绘制适应度值的变化曲线
    plt.title('迭代过程') # 设置图形标题
    plt.show() # 显示图形

