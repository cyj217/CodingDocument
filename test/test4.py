from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.variable import Real, Integer
import numpy as np

# 定义多目标优化问题
class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, 
                         n_obj=3, 
                         n_constr=0, 
                         xl=np.array([10, 5, 1]),
                         xu=np.array([30, 15, 10]),
                         variable_type=[Real(), Real(), Real()]
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 39.46 - 0.18 * x[:, 0] + 0.64 * x[:, 1] - 1.1 * x[:, 2] + 0.07 * x[:, 0] * x[:, 0] - 0.24 * x[:, 0] * x[:, 1] + 0.36 * x[:, 0] * x[:, 2] + 0.41 * x[:, 1] * x[:, 1] - 0.44 * x[:, 1] * x[:, 2] + 3.88 * x[:, 2] * x[:, 2]
        f2 = 14.89 - 0.14 * x[:, 0] + 0.48 * x[:, 1] - 0.84 * x[:, 2] + 0.01 * x[:, 0] * x[:, 0] - 0.16 * x[:, 0] * x[:, 1] + 0.28 * x[:, 0] * x[:, 2] + 0.27 * x[:, 1] * x[:, 1] - 0.4 * x[:, 1] * x[:, 2] + 2.95 * x[:, 2] * x[:, 2]
        f3 = -0.03 + 1.15 * x[:, 0] + 1.82 * x[:, 1] + 0.05 * x[:, 2]
        out["F"] = np.column_stack([f1, f2, f3])
        
# 创建多目标优化问题实例
problem = MyProblem()

# 创建NSGA-II算法实例
algorithm = NSGA2(pop_size=100, crossover_prob=0.9)

# 执行多目标优化
res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=False)

# 输出优化结果
print("最优解：")
print(res.X)
print("最优目标函数值：")
print(res.F)

# 可视化结果
plot = Scatter()
plot.add(res.F, color="red")
plot.show()