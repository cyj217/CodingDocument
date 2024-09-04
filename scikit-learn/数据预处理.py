from sklearn import preprocessing #导入预处理库
import numpy as np #导入numpy库
from sklearn.discriminant_analysis import StandardScaler #导入标准化库
from sklearn.impute import SimpleImputer #导入填充缺失值库

#数据标准化scale函数
X_train = np.array([[1.,-1.,2.], 
                    [2.,0.,0.],
                    [0.,1.,-1.]])
X_scaled = preprocessing.scale(X_train)
print(X_scaled) 
print(X_scaled.mean(axis=0)) 
print(X_scaled.std(axis=0)) 

#数据标准化StandardScaler类
scaler = preprocessing.StandardScaler().fit(X_train) 
print(scaler) 
print(scaler.mean_) 
print(scaler.scale_)
print(scaler.transform(X_train)) 
StandardScaler(copy=True, with_mean=True, with_std=True) 

#设置特征值的范围
X_train = np.array([[1.,-1.,2.],
                    [2.,0.,0.],
                    [0.,1.,-1.]])
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) 
X_train_minmax = min_max_scaler.fit_transform(X_train) #特征值缩放
print(X_train_minmax) 

#归一化
X_train_normalized = preprocessing.normalize(X_train,norm='l2') 
print(X_train_normalized) 

#归一化Normalizer类
normalizer = preprocessing.Normalizer().fit(X_train) 
X_train_normalized = normalizer.transform(X_train) 
print(X_train_normalized)

#二值化
binarizer = preprocessing.Binarizer(threshold=1.1) 
X_train_binarized = binarizer.transform(X_train)
print(X_train_binarized) 

#二值化(自己设置阈值，传出参数threshold)
binarizer = preprocessing.Binarizer(threshold=1.5) 
X_train_binarized = binarizer.transform(X_train) 
print(X_train_binarized) 

#编码类别特征
enc = preprocessing.OneHotEncoder(categories='auto') 
enc.fit([[0,0,3],
            [1,1,0],
            [0,2,1],
            [1,0,2]]) #fit学习编码
print(enc.transform([[0,1,3]]).toarray()) #transform编码

#填补缺失值
imp = SimpleImputer(missing_values=np.nan, strategy='mean') #实例化
imp.fit([[1,2],
            [np.nan,3],
            [7,6]]) #fit学习缺失值
X = [[np.nan,2],
        [6,np.nan],
        [7,6]] #新数据
print(imp.transform(X)) #transform补全缺失值
