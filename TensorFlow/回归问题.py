import numpy as np

# 生成数据
data = []
for i in range(100):
    x = np.random.uniform(-10., 10.)
    eps = np.random.normal(0., 0.01)
    y = 1.477 * x + 0.089 + eps
    data.append([x, y])
data = np.array(data)

# 计算误差
def mse(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

# 计算梯度
def step_gradient(b_current, w_current, point, lr):
    b_gradient = 0
    w_gradient = 0
    N = float(len(point))
    for i in range(0, len(point)):
        x = point[i, 0]
        y = point[i, 1]
        b_gradient += (2 / N) * ((w_current * x + b_current) - y)
        w_gradient += (2 / N) * ((w_current * x + b_current) - y) * x
    new_b = b_current - lr * b_gradient
    new_w = w_current - lr * w_gradient
    return [new_b, new_w]

# 梯度更新
def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    b = starting_b
    w = starting_w
    for step in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points)
        if step % 50 == 0:
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    return [b, w]

# 主函数
def main():
    lr = 0.01
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    [b, w] = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)
    print(f"Final loss:{loss}, w:{w}, b:{b}")
