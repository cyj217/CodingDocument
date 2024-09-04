import numpy as np # 导入numpy库
import tensorflow as tf # 导入tensorflow库

labels = [] # 用于存储情感分类(1:积极, 0:消极)
vocab = set() # set类型，存放不重复的字符
context = [] # 存放文本列表

with open("C:/Users/陈语捷/Desktop/缓存/python_work/Machine Learning/data/深度学习案例/深度学习案例集源码/深度学习案例集源码/第1章/ChnSentiCorp.txt", mode="r", encoding="UTF-8") as emotion_file:
    for line in emotion_file.readlines(): # 逐行读取
        line = line.strip().split(",") # 将每行数据以逗号分隔
        labels.append(int(line[0])) # 读取分类label
        text = line[1] # 读取每行的文本
        context.append(text) # 存储文本内容
        for char in text: vocab.add(char) # 将字符依次读取到字库并确保不重复

voacb_list = list(sorted(vocab)) # 将set类型转换为list类型并排序
print(len(voacb_list)) # 输出字符个数

token_list = [] # 创建一个存储句子数字的列表
for text in context: # 依次读取存储的每个句子
    token = [voacb_list.index(char) for char in text] # 将句子中的每个字依次读取并查询字符中的序号
    token = token[:80] + [0]*(80 - len(token)) # 以80个字符为长度对句子进行截取或者填充
    token_list.append(token) # 存储在token_list中
token_list= np.array(token_list) # 将存储的数据集进行格式化处理
labels = np.array(labels) # 将存储的数据集进行格式化处理

input_token = tf.keras.Input(shape=(80,)) # 创建一个占位符，固定输入的格式
embedding = tf.keras.layers.Embedding(input_dim=3508, output_dim=128)(input_token) # 创建embedding层，将每个字符转换为128维的向量
embedding = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128))(embedding) # 使用双向GRU对数据特征进行提取
output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(embedding) # 使用全连接层做分类器对数据进行分类
model = tf.keras.Model(inputs=input_token, outputs=output) # 组合模型

model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"]) # 定义优化器，损失函数以及准确率
model.fit(token_list, labels, epochs=10, verbose=2) # 训练模型
