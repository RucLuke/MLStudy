from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles
import numpy as np


def plot_svc_decision_function(inner_model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(x_lim[0], x_lim[1], 30)
    y = np.linspace(y_lim[0], y_lim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = inner_model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(inner_model.support_vectors_[:, 0],
                   inner_model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


def kernel_model_linear():
    """
    线性核函数
    :return:
    """
    x, y = make_circles(100, factor=.1, noise=.1)
    clf = SVC(kernel='linear').fit(x, y)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(clf, plot_support=False)
    plot_3d(x, y)
    plt.show()


def linear_model():
    x, y = make_blobs(n_samples=60, centers=2, random_state=0, cluster_std=0.8)
    model = SVC(kernel='linear')
    model.fit(x, y)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model)


def kernel_model_rbf():
    x, y = make_circles(100, factor=.1, noise=.1)
    clf = SVC(kernel='rbf', C=1E6).fit(x, y)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(clf, plot_support=False)
    plot_3d(x, y)
    plt.show()


def plot_3d(param_x, param_y):
    r = np.exp(-(param_x ** 2).sum(1))

    def plot_3D(elev=30, azim=30, X=param_x, y=param_y):
        ax = plt.subplot(projection='3d')
        ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('r')

    plot_3D(elev=45, azim=45, X=param_x, y=param_y)


def plot_soft_margin():
    x, y = make_blobs(n_samples=100, centers=2,
                      random_state=0, cluster_std=0.8)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    for axi, C in zip(ax, [10.0, 0.1]):
        model = SVC(kernel='linear', C=C).fit(x, y)
        axi.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
        plot_svc_decision_function(model, axi)
        axi.scatter(model.support_vectors_[:, 0],
                    model.support_vectors_[:, 1],
                    s=300, lw=1, facecolors='none')
        axi.set_title('C = {0:.1f}'.format(C), size=14)
    plt.show()


def plot_soft_margin_kernel():
    X, y = make_blobs(n_samples=100, centers=2,
                      random_state=0, cluster_std=1.1)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

    for axi, gamma in zip(ax, [10.0, 0.1]):
        print(gamma)
        model = SVC(kernel='rbf', gamma=gamma).fit(X, y)
        axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
        plot_svc_decision_function(model, axi)
        axi.scatter(model.support_vectors_[:, 0],
                    model.support_vectors_[:, 1],
                    s=300, lw=1, facecolors='none')
        axi.set_title('gamma = {0:.1f}'.format(gamma), size=14)
    plt.show()


if __name__ == '__main__':
    # linear_model()
    # kernel_model_linear()
    # kernel_model_rbf()
    # plot_soft_margin()
    plot_soft_margin_kernel()
