from sklearn.naive_bayes import GaussianNB # 导入高斯模型
from sklearn.naive_bayes import MultinomialNB # 导入多项式模型
from sklearn.naive_bayes import BernoulliNB # 导入伯努利模型
from sklearn import datasets # 导入数据集
from sklearn.model_selection import train_test_split # 导入数据集划分包
from sklearn import naive_bayes # 导入贝叶斯分类包
import numpy as np # 导入numpy库

# 高斯模型
x = np.array([[-3,7],[1,5],[1,2],[-2,0],[2,3],[-4,0],[-1,1],[1,1],[-2,2],[2,7],[-4,1],[-2,7]]) # 输入数据
Y = np.array([3,3,3,3,4,3,3,4,3,4,4,4]) # 标签
model = GaussianNB() # 创建高斯模型
model.fit(x,Y) # 模型训练
predicted = model.predict([[1,2],[3,4]]) # 预测新数据
print(predicted) # 打印预测结果

# 多项式模型
X = np.random.randint(5,size=(6,100)) # 输入数据
y = np.array([1,2,3,4,5,6]) # 标签
clf = MultinomialNB() # 创建多项式模型
clf.fit(X,y) # 模型训练
predicted = clf.predict(X[2:]) # 预测新数据
print(predicted) # 打印预测结果

# 伯努利模型
X = np.random.randint(2,size=(6,100)) # 输入数据
Y = np.array([1,2,3,4,4,5]) # 标签
clf = BernoulliNB() # 创建伯努利模型
clf.fit(X,Y) # 模型训练
print(clf.predict(X[2:])) # 预测新数据

# iris贝叶斯分类
iris = datasets.load_iris() # 导入数据集
X = iris["data"] # 输入数据
y = iris["target"] # 标签
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3) # 划分数据集
Model = naive_bayes.GaussianNB() # 样本特征的分布大部分是连续值，因此使用GaussianNB；如果样本是离散值，则选择MultinomialNB或BernoulliNB
Model.fit(X_train,y_train) # 模型训练
predict = Model.predict(X_test) # 预测新数据
print(predict) # 打印预测结果
print(y_test) # 打印真实结果
score = Model.score(X_test,y_test) # 计算准确率
print('the score is :',score) # 打印准确率
