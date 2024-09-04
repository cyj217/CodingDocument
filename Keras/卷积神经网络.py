from keras.models import Sequential # 导入序贯模型
from keras.layers import Conv2D,MaxPool2D # 导入卷积层和池化层
from keras.layers import Dense,Flatten # 导入全连接层和扁平化层
from keras.utils import to_categorical # 导入one-hot编码函数

from keras.datasets import mnist # 导入MNIST数据集
(x_train,y_train),(x_test,y_test) = mnist.load_data() # 加载数据集
import matplotlib.pyplot as plt # 导入matplotlib.pyplot模块
plt.imshow(x_train[0]) # 显示第一张图片
plt.show() # 显示图片

img_x,img_y = 28,28 # 图片的长和宽
x_train = x_train.reshape(x_train.shape[0],img_x,img_y,1) # 将训练集的数据格式转换为(60000,28,28,1)/(n,rows,cols,channels)
x_test = x_test.reshape(x_test.shape[0],img_x,img_y,1) # 将测试集的数据格式转换为(10000,28,28,1)/(n,rows,cols,channels)
x_train = x_train.astype('float32') # 将训练集的数据类型转换为float32
x_test = x_test.astype('float32') # 将测试集的数据类型转换为float32
x_train /= 255 # 将训练集的数据归一化
x_test /= 255 # 将测试集的数据归一化
y_train = to_categorical(y_train,10) # 将训练集的标签转换为one-hot编码
y_test = to_categorical(y_test,10) # 将测试集的标签转换为one-hot编码

model = Sequential()
model.add(Conv2D(32,kernel_size=(5,5),padding='same',activation='relu',input_shape=(img_x,img_y,1))) # 添加卷积层
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) # 添加池化层
model.add(Conv2D(64,kernel_size=(5,5),padding='same',activation='relu')) # 添加卷积层
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) # 添加池化层
model.add(Flatten()) # 添加扁平化层
model.add(Dense(1000,activation='relu')) # 添加全连接层
model.add(Dense(10,activation='softmax')) # 添加全连接层

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) # 编译模型

model.fit(x_train,y_train,batch_size=128,epochs=10) # 训练模型

score = model.evaluate(x_test,y_test) # 评估模型
print('acc',score[1]) # 打印模型的准确率
print('loss',score[0]) # 打印模型的损失值
