import pyDOE as doe
import csv
import numpy as np

level = 3  # 水平数
parameters = ["a", "b", "c"]  # 参数名

parameter_ranges = [[10, 30], [5, 15], [1, 10]]  # 参数范围

num_samples = 10 # 样本数

samples = doe.lhs(len(parameters), samples=num_samples, criterion='maximin')  # 使用lhs函数生成样本

scaled_samples = []

# 缩放和转换样本值
for sample in samples:
    scaled_sample = []
    for i in range(len(sample)):
        min_value, max_value = parameter_ranges[i]
        if parameters[i] == "n_down" or parameters[i] == "n_up":
            actual_value = np.random.randint(min_value, max_value + 1)
        else:
            actual_value = min_value + (max_value - min_value) * sample[i]
        scaled_sample.append(round(actual_value, 2))
    scaled_samples.append(scaled_sample)

filename = "samples-5019.txt"  # 文件名

# 写入参数名和样本到txt文件
with open(filename, mode='w') as file:
    # 写入参数名
    file.write('\t'.join(parameters))
    file.write('\n')

    # 写入样本
    for sample in scaled_samples:
        file.write('\t'.join(str(value) for value in sample))
        file.write('\n')