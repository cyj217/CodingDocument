# 基础数据类型
int_a = 1 # 整数变量
float_b = 1.2 # 浮点数变量
bool_t = True # 布尔变量中的真
bool_f = False # 布尔变量中的假
str_c = "hello world" # 字符串变量

# 基础运算
a = 1
b = 2
print (a + b) # 加法
print (a - b) # 减法
print (a * b) # 乘法
print (a / b) # 除法
print (a % b) # 取余
print (a ** b) # 幂运算
print (a // b) # 取整

# 列表/;ist
la = [1, 2, 3, 4] # 创建一个列表，包括四个元素
lb = ['a', 'b', 'c'] # 创建一个列表，包括三个元素
print (la[0]) # 打印列表中的第一个元素
print (la[1:3]) # 打印列表中的第二个到第三个元素
print (la[2:]) # 打印列表中的第三个到最后一个元素
print (len(la)) # 打印列表的长度
print (la[-1]) # 打印列表中的最后一个元素
la[1] = 5 # 修改列表中的第二个元素
print (la) # 打印修改后的列表
la.append(5) # 在列表的最后添加一个元素
la.pop(2) # 删除列表中的第三个元素
print (la) # 打印修改后的列表
# 列表迭代
for index, value in enumerate(la):
    print (index, value) # 打印列表中的每个元素的索引和值
# 使用二维列表来模拟矩阵(列表中嵌套列表)
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]
print (matrix) # 打印矩阵
print (matrix[1][1]) # 打印矩阵中的第二行第二列的元素

# 元组/tuple
ta = ('a','b','c') # 创建一个元组，包括三个元素
print (len(ta)) # 打印元组的长度
print (ta[1]) # 打印元组中的第二个元素
print (ta[1:3]) # 打印元组中的第二个到第三个元素
print (ta[2:]) # 打印元组中的第三个到最后一个元素
# 元组迭代
for v in ta:
    print (v) # 打印元组中的每个元素的值

# 字典/dict
da = {'a':123, 'b':456, 'c':789} # 创建一个字典，包括三个元素
db = {'a':[1,2,3], 'b':[4,5,6]} # 创建一个字典，包括两个元素
print (len(da)) # 打印字典的长度
print (da.get('b')) # 打印字典中键为'b'的元素的值
print ('d' in da) # 判断字典中是否有键为'd'的元素
print (da.keys()) # 打印字典中所有键的列表
print (da.values()) # 打印字典中所有值的列表
print (da.items()) # 打印字典中所有键值对的列表
# 查看映射关系
for key, value in da.items():
    print (key, '=', value) # 打印字典中每个键值对的键和值
# 添加或删除映射关系
da['d'] = 10 # 添加一个键值对
da.pop('a') # 删除一个键值对
print (da) # 打印修改后的字典

# 集合/set
sa = set(['a','b','c','d']) # 创建一个集合，包括四个元素
sb = set(['b','c','f']) # 创建一个集合，包括三个元素
print ('a' in sa) # 判断集合中是否有元素'a'
print (sa & sb) # 打印两个集合的交集
print (sa | sb) # 打印两个集合的并集
print (sa - sb) # 打印两个集合的差集
print (sa ^ sb) # 打印两个集合的对称差集

# 条件语句
a = [1,2,3] # 创建一个列表
if 1 in a: # 判断1是否在列表中
    print ('1 is in a') # 如果在，打印'1 is in a'
else: 
    print ('1 is not in a') # 如果不在，打印'1 is not in a'

# 循环语句/满足条件时跳出循环
a = [1,2,3,4,5] # 创建一个列表
for i in a: 
    if i > 3: # 判断i是否大于3
        break # 如果大于3，跳出循环
    print (i) # 如果大于3，打印i
# 循环语句/通过循环知道满足某个条件
a = 1 # 创建一个变量
while a < 10: # 判断a是否小于10
    a = a + 1 # 如果小于10，a加1
    print ('do something also.') # 如果小于10，打印'do something also...'
print (a) # 打印a

# 函数
def find_max(a,b,c):
    max_number = None 
    if a > b and a > c:
        max_number = a
    elif b > a and b > c:
        max_number = b
    elif c > a and c > b:
        max_number = c
    return max_number
max_num = find_max(a=1, b=2, c=3)
print ('最大的数是：', max_num) 

# 类与实例
class cat():
    def __init__(self,color,weight):
        self.color = color 
        self.weight = weight 
    def catch_mice(self): 
        print ('抓老鼠')
    def eat_mice(self):
        print ('吃老鼠')
# 类的实例化
my_cat = cat('yellow',10) 
# 调用类的方法
my_cat.catch_mice()
my_cat.eat_mice()
# 访问类的属性
print (my_cat.color)
print (my_cat.weight)

# 迭代
a = [1,2,3,4,5] # 创建一个列表
def my_func(x):
    print ('do some on ', x)
    return x + 1
# 迭代版本
b = [my_func(x) for x in a] # 对列表a中的每个元素调用my_func函数
print (b) # 打印列表b
# 循环版本
b2 = [] # 创建一个空列表
for x in a: # 对列表a中的每个元素
    t = x + 1
    b2.append(t) # 将t添加到列表b2中
print (b2) # 打印列表b2
