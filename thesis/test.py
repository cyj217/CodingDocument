from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize
import numpy as np

# 定义多目标优化问题
class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=9, 
                         n_obj=3, 
                         n_constr=1, 
                         xl=np.array([20, 5, 5, 400, 20, 5, , 400, 20]),
                         xu=np.array([35, 12, 6, 800, 35, 9, 6, 800, 40]),
                         x1=np.array([20, 35]), 
                         x2=np.array([5, 12]),
                         x3=[3, 4, 5, 6],
                         x4=np.array([400, 800]),
                         x5=np.array([20, 35]),
                         x6=np.array([5, 9]),
                         x7=[4, 5, 6],
                         x8=np.array([400, 800]),
                         x9=np.array([20, 40]),
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] * 0.21299 - x[:, 1] * 0.21868 - ((160 - x[:, 0] * 2)/(x[:, 2] - 1)) * 0.00861 + x[:, 3] * 0.00075 + x[:, 4] * 0.12388 + x[:, 5] * 0.04369 - ((160 - x[:, 4] * 2)/(x[:, 6] - 1)) * 0.03714 + x[:, 7] * 0.038619 + x[:, 8] * 0.013052 - 16.21145
        f2 = x[:, 2]  * x[:, 3] + x[:, 5] * x[:, 6]
        f3 = x[:, 8]
        g1 = - x[:, 0] * 0.23571 + x[:, 1] * 1.28615 + ((160 - x[:, 0] * 2)/(x[:, 2] - 1)) * 0.14168 - x[:, 3] * 0.14604 - x[:, 4] * 0.34289 - x[:, 5] * 0.10964 - ((160 - x[:, 4] * 2)/(x[:, 6] - 1)) * 0.15042 - x[:, 7] * 0.46692 - x[:, 8] * 1.96395 - 528.40053
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1])


# 创建多目标优化问题实例
problem = MyProblem()

# 创建NSGA-II算法实例
algorithm = NSGA2(pop_size=100)

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
