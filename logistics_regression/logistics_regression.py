import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time


def sigmoid(z):
    # sigmoid 函数
    return 1 / (1 + np.exp(-z))


def model(x, theta):
    # h函数
    return sigmoid(np.dot(x, theta.T))


def cost(x, y, theta):
    # 损失的计算函数
    left = np.multiply(-y, np.log(model(x, theta)))
    right = np.multiply((1 - y), np.log(1 - model(x, theta)))
    return np.sum(left - right) / (len(x))


def gradient(x, y, theta):
    # 梯度下降
    grad = np.zeros(theta.shape)
    error = (model(x, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, x[:, j])
        grad[0, j] = np.sum(term) / len(x)
    return grad


STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


def stop_criterion(stop_type, value, threshold):
    # 设定三种不同的的停止策略
    if stop_type == STOP_ITER:
        return value > threshold
    elif stop_type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    elif stop_type == STOP_GRAD:
        return np.linalg.norm(value) < threshold


def shuffle_data(data):
    # 数据洗牌
    np.random.shuffle(data)
    columns = data.shape[1]
    x = data[:, 0:columns - 1]
    y = data[:, columns - 1:columns]
    return x, y


def plot_pic(data):
    print(data.shape)
    positive = data[pdData["Admitted"] == 1]
    negative = data[pdData["Admitted"] == 0]
    fix, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(positive["Exam1"], positive["Exam2"], s=30, marker="o", label="Admitted")
    ax.scatter(negative["Exam1"], negative["Exam2"], s=30, marker="x", label="Not Admitted")
    ax.legend()
    ax.set_xlabel("Exam1 score")
    ax.set_ylabel("Exam2 score")
    nums = np.arange(-10, 10, step=1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(nums, sigmoid(nums), 'r')
    plt.show()


def descent(data, theta, batch_size, stop_type, thresh, alpha):
    # 梯度下降求解
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    x, y = shuffle_data(data)
    grad = np.zeros(theta.shape)
    costs = [cost(x, y, theta)]

    while True:
        grad = gradient(x[k:k + batch_size], y[k:k + batch_size], theta)
        k += batch_size  # 取batch数量个数据
        if k >= n:
            k = 0
            x, y = shuffle_data(data)  # 重新洗牌
        theta = theta - alpha * grad  # 参数更新
        costs.append(cost(x, y, theta))
        i += 1
        if stop_type == STOP_ITER:
            value = i
        elif stop_type == STOP_COST:
            value = costs
        elif stop_type == STOP_GRAD:
            value = grad

        if stop_criterion(stop_type, value, thresh):
            break

    return theta, i - 1, costs, grad, time.time() - init_time


def run_expe(data, theta, batch_size, stop_type, thresh, alpha):
    theta, iter, costs, grad, dur = descent(data, theta, batch_size, stop_type, thresh, alpha)
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += " data - learning rate : {} - ".format(alpha)
    if batch_size == n:
        str_desc_type = "Gradient"
    elif batch_size == 1:
        str_desc_type = "Stochastic"
    else:
        str_desc_type = "Mini-batch({})".format(thresh)
    name += str_desc_type + "descent - stop"
    if stop_type == STOP_ITER:
        str_stop = "{} iterations".format(thresh)
    elif stop_type == STOP_COST:
        str_stop = "costs change < {}".format(thresh)
    else:
        str_stop = "gradient norm < {}".format(thresh)

    name += str_stop
    print("***{}\nTheta:{} - Iter: {} - Last cost :{:03.2f} - Duration: {:03.2f}s".format(name, theta, iter,
                                                                                          costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, "r")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    ax.set_title(name.upper() + '- Error vs. Iteration')

    return theta


if __name__ == '__main__':
    path = 'data' + os.sep + "LogiReg_data.txt"
    pdData = pd.read_csv(path, header=None, names=["Exam1", "Exam2", "Admitted"])
    # plot_pic(pdData)
    # 数据中增加一列
    pdData.insert(0, "ones", 1)

    # orig_data = pdData.as_matrix() as_matrix is out of date
    orig_data = pdData.values
    cols = orig_data.shape[1]
    # 原始的 X
    X_data = orig_data[:, 0:cols - 1]
    # 原始的 y
    y_data = orig_data[:, cols - 1:cols]
    theta_data = np.zeros([1, 3])
    # print(cost(X_data, y_data, theta_data))
    n = 100
    # run_expe(orig_data, theta_data, n, STOP_ITER, thresh=5000, alpha=0.00001)
    run_expe(orig_data, theta_data, n, STOP_COST, thresh=0.000001, alpha=0.001)
    plt.show()
