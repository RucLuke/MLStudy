import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    # sigmoid 函数
    return 1 / (1 + np.exp(-z))


def h_function(z):
    return -(z * np.log2(z) + (1 - z) * np.log2(1 - z))


if __name__ == '__main__':
    # x = np.arange(-10, 10)
    # y = sigmoid(x)

    x = np.arange(0.01, 1, 0.01)
    print(x)
    y = h_function(x)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y, 'r')
    plt.show()
