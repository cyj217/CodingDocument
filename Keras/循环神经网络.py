import numpy # 导入numpy库
import matplotlib.pyplot as plt # 导入matplotlib.pyplot库
from keras.models import Sequential # 导入Sequential模型
from keras.layers import Dense # 导入Dense层
from keras.layers import LSTM # 导入LSTM层
import pandas as pd # 导入pandas库
import os # 导入os库
from keras.models import Sequential, load_model # 导入Sequential模型和load_model函数
from sklearn.preprocessing import MinMaxScaler # 导入MinMaxScaler函数
import os.path as path # 导入path库

# 读取数据集
dataframe = pd.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3) # 读取国际航空乘客数据集
dataset = dataframe.values # 将数据集转换为数组
dataset = dataset.astype('float32') # 将数据集转换为浮点型
scaler = MinMaxScaler(feature_range=(0, 1)) # 将数据归一化
dataset = scaler.fit_transform(dataset) # 将数据归一化
train_size = int(len(dataset) * 0.65) # 训练集大小
trainlist = dataset[:train_size] # 训练集
testlist = dataset[train_size:] # 测试集
scaler = MinMaxScaler(feature_range=(0, 1)) # 将数据归一化
datasets = scaler.fit_transform(dataset) # 将数据归一化

# 创建数据集
def create_dataset(dataset,look_back): # 创建数据集
    dataX,dataY=[],[] # 创建空列表
    for i in range(len(dataset)-look_back-1): # 遍历数据集
        a=dataset[i:(i+look_back)] # 创建a
        dataX.append(a) # 将a添加到dataX中
        dataY.append(dataset[i+look_back]) # 将dataset[i+look_back]添加到dataY中
    return numpy.array(dataX),numpy.array(dataY) # 返回dataX和dataY
look_back=1 # 设置look_back
trainX,trainY=create_dataset(trainlist,look_back) # 创建训练集
testX,testY=create_dataset(testlist,look_back) # 创建测试集

# 创建并拟合LSTM网络
trainX=numpy.reshape(trainX,(trainX.shape[0],trainX.shape[1],1)) # 将训练集转换为3维
testX=numpy.reshape(testX,(testX.shape[0],testX.shape[1],1)) # 将测试集转换为3维

# 创建LSTM网络
model=Sequential() # 创建Sequential模型
model.add(LSTM(4,input_shape=(None,1))) # 添加LSTM层
model.add(Dense(1)) # 添加Dense层
model.compile(loss='mean_squared_error',optimizer='adam') # 编译模型
model.fit(trainX,trainY,epochs=100,batch_size=1,verbose=2) # 拟合模型

# 预测
trainPredict=model.predict(trainX) # 预测训练集
testPredict=model.predict(testX) # 预测测试集

# 反归一化
trainPredict=scaler.inverse_transform(trainPredict) # 反归一化训练集
trainY=scaler.inverse_transform(trainY) # 反归一化训练集
testPredict=scaler.inverse_transform(testPredict) # 反归一化测试集
testY=scaler.inverse_transform(testY) # 反归一化测试集

# 查看结果
plt.plot(trainY) # 绘制训练集真实值
plt.plot(trainPredict[1:]) # 绘制训练集预测值
plt.show() # 显示图像
plt.plot(testY) # 绘制测试集真实值
plt.plot(testPredict[1:]) # 绘制测试集预测值
plt.show() # 显示图像
