import numpy as np # 导入numpy库
from sklearn.datasets import load_breast_cancer # 乳腺癌数据集
from sklearn.model_selection import train_test_split # 数据集划分
from sklearn.linear_model import LogisticRegression # Logistic回归
from sklearn.preprocessing import PolynomialFeatures # 多项式特征
from sklearn.pipeline import Pipeline # 管道
import time # 导入时间库
import matplotlib.pyplot as plt # 导入matplotlib库
from sklearn.model_selection import learning_curve # 导入学习曲线函数
from sklearn.model_selection import ShuffleSplit # 导入交叉检验函数

# 加载数据集
cancer = load_breast_cancer() # 加载数据集
X = cancer.data # 数据
y = cancer.target # 标签
print('data shape: {0}; no. positive: {1}; no. negative: {2}' .format(X.shape,y[y==1].shape[0],y[y==0].shape[0])) # 打印数据集信息
print(cancer.data[0]) # 打印第一个样本的数据
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2) # 划分数据集

# 建立模型
model = LogisticRegression() # 建立模型
model.fit(X_train,y_train) # 训练模型
train_score = model.score(X_train, y_train) # 训练集上的准确率
test_score = model.score(X_test, y_test) # 测试集上的准确率
print('train score: {train_score:.6f};test_score:{test_score:.6f}'.format(train_score=train_score,test_score=test_score)) # 打印准确率

# 观察模型在测试集上的表现
y_pred = model.predict(X_test) # 预测值
print('matchs:{0}/{1}'.format(np.equal(y_pred,y_test).shape[0],y_test.shape[0])) # 打印匹配情况

# 预测概率（找出预测概率低于90%的样本）
y_pred_proba = model.predict_proba(X_test) #计算每个测试样本的预测概率
print('sample of predict probability: {0}'.format(y_pred_proba[0])) # 打印第一个样本的预测概率
y_pred_proba_0 = y_pred_proba[:, 0]>0.1 # 找出预测为阴性的概率大于0.1的样本，
result = y_pred_proba[y_pred_proba_0] # 保存在result里
y_pred_proba_1 = result[:, 1]>0.1 # 找出预测为阳性的概率大于0.1的样本，保存在result里
print(result[y_pred_proba_1]) # 打印result

# Pipeline增加多项式特征
def polynomial_model(degree=1, **kwarg): # 定义多项式模型函数
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False) # 增加多项式特征
    logistic_regression = LogisticRegression(**kwarg) # 建立逻辑回归模型
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("logistic_regression",logistic_regression)]) # 建立管道
    return pipeline # 返回管道

# 建立模型
model = polynomial_model(degree=2,penalty='l1',solver='liblinear') # 建立模型
start = time.perf_counter() # 计时开始
model.fit(X_train, y_train) # 训练模型
train_score = model.score(X_train, y_train) # 训练集上的准确率
cv_score = model.score(X_test, y_test) # 测试集上的准确率
print('elaspe:{0:.6f};train_score:{1:0.6f};cv_score:{2:.6f}'.format(
    time.perf_counter()-start,train_score,cv_score)) # 打印准确率

# 模型参数
logistic_regression = model.named_steps['logistic_regression'] # 获取逻辑回归模型
print('model parameters shape: {0};count of non-zero element: {1}'.format(
    logistic_regression.coef_.shape,
    np.count_nonzero(logistic_regression.coef_))) # 打印模型参数形状和非零元素个数

# 学习曲线
start = time.perf_counter() # 计时开始
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0) # 交叉检验
title = 'Learning Curves(degree={0},penalty={1})' # 图像标题
degrees = [1,2] # 多项式次数
penalty = 'l1' # 正则化项(l1/l2范数作为正则项)

# 绘制学习曲线
plt.figure(figsize=(12,4),dpi=144) # 创建画布
for i in range(len(degrees)): 
    plt.subplot(1,len(degrees),i+1) # 创建子图
    train_sizes, train_scores, test_scores = learning_curve(polynomial_model(degree=degrees[i], penalty=penalty,
        solver='liblinear', max_iter=300), X_train, y_train, cv=cv) # 计算得分
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training score") # 绘制训练得分
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Cross-validation score") # 绘制交叉检验得分
    plt.title(title.format(degrees[i], penalty)) # 标题
    plt.ylim(0.8, 1.01) # 纵坐标
    plt.legend(loc="best") # 图例
print('elaspe:{0:.6f}'.format(time.process_time()-start)) # 打印计时信息
plt.show() # 显示图像
