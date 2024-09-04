import matplotlib.pyplot as plt #导入matplotlib库
from sklearn.linear_model import LinearRegression #导入线性回归模型
from sklearn.preprocessing import PolynomialFeatures #导入多项式回归模型
import numpy as np #导入numpy库

#数据可视化
def runplt(): #创建绘图函数
    plt.figure() #创建绘图对象
    plt.title(u'diameter-cost curve') #设置图表标题
    plt.xlabel(u'diameter') #设置x轴标签
    plt.ylabel(u'cost') #设置y轴标签
    plt.axis([0, 25, 0, 70]) #设置x轴和y轴的最大最小值
    plt.grid(True) #显示网格
    return plt #返回绘图对象
plt = runplt() #创建绘图对象
X = [[6], [8], [10], [15], [18]] #披萨直径
y = [[20], [25], [35], [50], [60]] #披萨价格
plt.plot(X, y, 'ro-') #绘制披萨价格和直径的关系
plt.show() #显示图表

#创建并拟合模型
model = LinearRegression() #创建线性回归模型
model.fit(X, y) #训练模型
predict_data = np.array([12]).reshape(-1,1) #预测数据
predict_result = model.predict(predict_data) #预测结果
print('预测一张12英寸披萨价格: $%.2f'%predict_result) #输出预测结果

#残差预测(残差平方和)
plt = runplt() #创建绘图对象
plt.plot(X, y, 'k') #绘制训练数据点
X2 = [[0], [10], [14], [25]] #测试数据点
model = LinearRegression() #创建线性回归模型
model.fit(X, y) #训练模型
y2 = model.predict(X2) #预测结果
plt.plot(X, y, 'k') #绘制训练数据点
plt.plot(X2, y2, 'g-') #绘制预测数据点
yr = model.predict(X) #训练数据点的预测结果
for idx, x in enumerate(X): #打印输出训练数据点的残差
    plt.plot([x, x], [y[idx], yr[idx]], 'r-') #绘制残差
plt.show() #显示图表
print('残差平方和: %.2f'%np.mean((model.predict(X)-y)**2)) #输出残差平方和

#拟合的曲线方程(y=a0+a1x)/解一元线性回归的最小二乘法(x的方差和x与y的协方差)
X = [[6], [8], [10], [15], [18]] #披萨直径
y = [[20], [25], [35], [50], [60]] #披萨价格
Xmean = np.mean(X) #求解x的均值
ymean = np.mean(y) #求解y的均值
bate = np.cov([6, 8, 10, 15, 18], [20, 25, 35, 50, 60])[0][1]/np.var([6, 8, 10, 15, 18], ddof=1) #求解beta
alpha = ymean-bate*Xmean #求解alpha
print('拟合的曲线方程为: \n') #输出拟合的曲线方程
print('y=%.2f'%alpha,'+%.2fx'%bate) #输出拟合的曲线方程

#测试集(模型评估)
X_test = [[12], [14], [16]] #测试集
y_test = [[40], [45], [55]] #测试集的真实值
model = LinearRegression() #建立线性回归模型
model.fit(X,y) #训练模型
print(model.score(X_test, y_test)) #输出R_squared

#多元线性回归(y=a0+a1x1+a2x2)
X = [[6, 2], [8, 1], [10, 0], [15, 2], [18, 0]] #x1:直径，x2:份数
y = [[22], [25], [32], [55], [58]] #y:价格
model = LinearRegression() #建立线性回归模型
model.fit(X, y) #训练模型
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]] #测试集
y_test = [[28], [26], [36], [60], [36]] #测试集的真实值
predictions = model.predict(X_test) #预测值
for i, prediction in enumerate(predictions): #打印输出测试集的预测值和真实值
    print('Precited: %s, Target: %s'%(prediction, y_test[i])) #输出预测值和真实值
print('R_squared: %.2f'%model.score(X_test, y_test)) #输出R_squared

#多远线性回归的方程
X = np.array([[6, 2], [8, 1], [10, 0], [15, 2], [18, 0]]) #x1:直径，x2:份数
y = np.array([[22], [25], [32], [55], [58]]) #y:价格
X_mean = np.mean(X, axis=0) #axis=0表示按列求均值
y_mean = np.mean(y, axis=0) #axis=0表示按列求均值
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) #求解beta
alpha = y_mean - beta.T.dot(X_mean) #求解alpha
print('拟合的曲线方程为: \n') #输出拟合的曲线方程
print('y = %.2f + %.2fx1 + %.2fx2' % (alpha[0], beta[0][0], beta[1][0])) #输出拟合的曲线方程

#定义训练集和测试集
X_train = [[6], [8], [10], [14], [18]] #训练集的直径
y_train = [[7], [9], [13], [17.5], [18]] #训练集的价格
X_test = [[6], [8], [11], [16]] #测试集的直径
y_test = [[8], [12], [15], [18]] #测试集的价格

#建立线性回归，并用训练的模型绘图
regressor = LinearRegression() #建立线性回归对象
regressor.fit(X_train, y_train) #训练模型
xx = np.linspace(0, 26, 100) #在0-26之间均匀的取100个数
yy = regressor.predict(xx.reshape(xx.shape[0], 1)) #将xx转换成100行1列的数组
plt.plot(X_train, y_train, 'k.') #训练集的散点图
plt.plot(xx, yy) #拟合的直线

#二次多项式回归
quadratic_featurizer = PolynomialFeatures(degree=2) #实例化一个二次多项式特征实例
X_train_quadratic = quadratic_featurizer.fit_transform(X_train) #用二次多项式对样本X值做变换
X_test_quadratic = quadratic_featurizer.transform(X_test) #用二次多项式对测试值X值做变换
regressor_quadratic = LinearRegression() #创建一个线性回归实例
regressor_quadratic.fit(X_train_quadratic, y_train) #以多项式变换后的x值为输入，代入线性回归模型做训练
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1)) #把训练好X值代入模型，获得相应的输出
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-') #用训练好的模型作图
plt.show() #显示图像
print(X_train) #输出X_train
print(X_train_quadratic) #输出二次多项式变换后的X_train
print(X_test) #输出X_test
print(X_test_quadratic) #输出二次多项式变换后的X_test
print('1 r-squred', regressor.score(X_test, y_test)) #输出一元线性回归的R_squared
print('2 r-squred', regressor_quadratic.score(X_test_quadratic,y_test)) #输出二次多项式回归的R_squared
plt.plot(X_train, y_train, 'k.') #训练集的散点图
quadratic_featurizer = PolynomialFeatures(degree=2) #实例化一个二次多项式特征实例
X_train_quadratic = quadratic_featurizer.fit_transform(X_train) #用二次多项式对样本X值做变换
X_test_quadratic = quadratic_featurizer.transform(X_test) #用二次多项式对测试值X值做变换
regressor_quadratic = LinearRegression() #创建一个线性回归实例
regressor_quadratic.fit(X_train_quadratic, y_train) #以多项式变换后的x值为输入，代入线性回归模型做训练
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0],1)) #把训练好X值代入模型，获得相应的输出
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-') #用训练好的模型作图

#三次多项式回归
cubic_featurizer = PolynomialFeatures(degree=3) #实例化一个三次多项式特征实例
X_train_cubic = cubic_featurizer.fit_transform(X_train) #用三次多项式对样本X值做变换
X_test_cubic = cubic_featurizer.transform(X_test) #用三次多项式对测试值X值做变换
regressor_cubic = LinearRegression() #创建一个线性回归实例
regressor_cubic.fit(X_train_cubic, y_train) #以多项式变换后的x值为输入，代入线性回归模型做训练
xx_cubic = cubic_featurizer.transform(xx.reshape(xx.shape[0], 1)) #把训练好X值代入模型，获得相应的输出
plt.plot(xx, regressor_cubic.predict(xx_cubic)) #用训练好的模型作图
plt.show() #显示图像
print(X_train_cubic) #输出三次多项式变换后的X_train
print(X_test_cubic) #输出三次多项式变换后的X_test
print('2 r-squred', regressor_quadratic.score(X_test_quadratic, y_test)) #输出二次多项式回归的R_squared
print('3 r-squred', regressor_cubic.score(X_test_cubic, y_test)) #输出三次多项式回归的R_squared
plt.plot(X_train, y_train, 'k.') #训练集的散点图
quadratic_featurizer = PolynomialFeatures(degree=2) #实例化一个二次多项式特征实例
X_train_quadratic = quadratic_featurizer.fit_transform(X_train) #用二次多项式对样本X值做变换
X_test_quadratic = quadratic_featurizer.transform(X_test) #用二次多项式对测试值X值做变换
regressor_quadratic = LinearRegression() #创建一个线性回归实例
regressor_quadratic.fit(X_train_quadratic, y_train) #以多项式变换后的x值为输入，代入线性回归模型做训练
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0],1)) #把训练好X值代入模型，获得相应的输出
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-') #用训练好的模型作图

#七次多项式回归
seventh_featurizer = PolynomialFeatures(degree=7) #实例化一个七次多项式特征实例
X_train_seventh = seventh_featurizer.fit_transform(X_train) #用七次多项式对样本X值做变换
X_test_seventh = seventh_featurizer.transform(X_test) #用七次多项式对测试值X值做变换
regressor_seventh = LinearRegression() #创建一个线性回归实例
regressor_seventh.fit(X_train_seventh, y_train) #以多项式变换后的x值为输入，代入线性回归模型做训练
xx_seventh = seventh_featurizer.transform(xx.reshape(xx.shape[0], 1)) #把训练好X值代入模型，获得相应的输出
plt.plot(xx, regressor_seventh.predict(xx_seventh)) #用训练好的模型作图
plt.show() #显示图像
print('2 r-squared', regressor_quadratic.score(X_test_quadratic, y_test)) #输出二次多项式回归的R_squared
print('7 r-squared', regressor_seventh.score(X_test_seventh, y_test)) #输出七次多项式回归的R_squared
