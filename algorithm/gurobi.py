import gurobipy as grb # 导入gurobi模块

# multidict函数用于创建字典
student,chinese,math,english = grb.multidict({
    'student1':[1,2,3],
    'student2':[2,3,4],
    'student3':[3,4,5],
    'student4':[4,5,6],
}) # 用字典存储数据
print(student) # 输出学生名字
print(chinese) # 输出语文成绩
print(math) # 输出数学成绩
print(english) # 输出英语成绩

# Tuplelist函数用于创建元组列表
tl = grb.tuplelist([(1,2),(1,3),(2,3),(2,5)]) # 用元组列表存储数据
print(tl.select(1,'*')) # 输出第一个值为1的元组
print(tl.select('*',3)) # 输出第二个值为3的元组
tl.append((3,5)) # 添加元组
print(tl.select(3,'*')) # 输出第一个值为3的元组
print(tl.select(1,'*')) # 使用迭代的方法实现select功能

# Tuplelist子集
model = grb.Model() # 创建模型
tl = [(1,1),(1,2),(1,3),
      (2,1),(2,2),(2,3),
      (3,1),(3,2),(3,3)] # 元组列表
vars = model.addVars(tl,name="d") # 定义变量的下标
print(sum(vars.select(1,'*'))) # 输出第一个值为1的元组的和

