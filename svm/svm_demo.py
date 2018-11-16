import numpy as np
import matplotlib.pyplot as plt
# import seaborn_demo as sns
from scipy import stats
from sklearn.datasets.samples_generator import make_blobs


def split_line():
    x_fit = np.linspace(-1, 3.5)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap="autumn")
    plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

    for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
        y_fit = m * x_fit + b
        plt.plot(x_fit, y_fit, '-k')
        plt.fill_between(x_fit, y_fit - d, y_fit + d, edgecolor="none", color="#AAAAAA", alpha=0.4)
    plt.xlim(-1, 3.5)
    plt.show()


# sns.set()
# 支持向量机的基本原理：一个低纬不可分的问题转换为高维可分的问题
if __name__ == '__main__':
    x, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
    # plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap="autumn")
    # plt.show()
    # print("main")
    split_line()
