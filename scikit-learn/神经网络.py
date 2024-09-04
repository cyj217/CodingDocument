from sklearn.datasets import load_digits # 导入手写体数字加载器
from sklearn.model_selection import train_test_split # 导入数据分割器
from sklearn.neural_network import MLPClassifier # 导入多层感知器分类器
from sklearn.preprocessing import StandardScaler # 导入数据标准化模块

# 神经网络实现
digits = load_digits() # 加载数据
X= digits.data # 提取特征
y = digits.target # 提取标签
scaler = StandardScaler() # 标准化转换0-1   
scaler.fit(X) # 训练标准化对象
X = scaler.transform(X) # 转换数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2) # 分割数据集
mlp = MLPClassifier(solver='sgd',activation='relu',alpha=1e-4,hidden_layer_sizes=(50),random_state=1,max_iter=100,verbose=True,learning_rate_init=.1) # 建立神经网络模型
mlp.fit(X_train,y_train) # 训练模型
print('score: ',mlp.score(X_test,y_test)) # 打印准确率
print('n_layers_: ',mlp.n_layers_) # 打印神经网络层数
print('n_iter_: ',mlp.n_iter_) # 打印迭代次数
print('loss_: ',mlp.loss_) # 打印损失函数值
print('out_activation_: ',mlp.out_activation_) # 打印输出激活函数
