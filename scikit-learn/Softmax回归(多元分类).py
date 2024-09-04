from sklearn.model_selection import train_test_split # 数据集划分

# 载入鸢尾花数据集
def load_dataset_flower(): 
    from sklearn import datasets # 导入数据集模块
    iris = datasets.load_iris() # 载入数据集
    return iris # 返回数据集

# Softmax函数回归多分类
def softmax_classify(iris):
    from sklearn.linear_model import LogisticRegression # 导入逻辑回归模块
    X = iris["data"] # 获取特征变量
    y = iris["target"] # 获取目标变量
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2) # 划分数据集
    softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs",C=10) # 创建逻辑回归模型
    softmax_reg.fit(X_train,y_train) # 训练模型
    predict = softmax_reg.predict(X_test) # 预测
    predict_pro = softmax_reg.predict_proba(X_test) # 预测概率
    print('softmax 回归预测为: \n',predict_pro) # 打印预测概率
    print(iris.target_names) # 打印目标变量名称
    print('三类概率分别为: \n', predict_pro) # 打印预测概率
if __name__ == '__main__':
    iris = load_dataset_flower() # 载入数据集
    softmax_classify(iris) # 调用函数
