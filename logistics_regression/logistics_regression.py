import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def sigmoid(z):
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
    grad = np.zeros(theta.shape)
    error = (model(x, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, x[:, j])
        grad[0, j] = np.sum(term) / len(x)
    return grad


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


if __name__ == '__main__':
    path = 'data' + os.sep + "LogiReg_data.txt"
    pdData = pd.read_csv(path, header=None, names=["Exam1", "Exam2", "Admitted"])
    # plot_pic(pdData)
    # 数据中增加一列
    pdData.insert(0, "ones", 1)
    orig_data = pdData.as_matrix()
    cols = orig_data.shape[1]
    # 原始的 X
    X_data = orig_data[:, 0:cols - 1]
    # 原始的 y
    y_data = orig_data[:, cols - 1:cols]
    theta_data = np.zeros([1, 3])
    print(cost(X_data, y_data, theta_data))
