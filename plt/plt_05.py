import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    # sigmoid 函数
    return 1 / (1 + np.exp(-z))


x = np.arange(-10, 10)
y = sigmoid(x)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(x, y, 'r')
plt.show()
