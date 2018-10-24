import numpy as np
from numpy import pi

# numpy 常用函数
# a = np.arange(15).reshape(3, 5)
# print(a)
# print(a.shape)
# print(a.ndim)
# print(a.size)
# print(a.dtype.name)
#
# b = np.zeros((3, 4))
# print(b)
# c = np.ones((2, 3, 4), dtype=np.int32)
# print(c)

# d = np.arange(10, 30, 2)
# print(d)
#
# e = np.random.random((2, 3))
# print(e)

# f = np.linspace(0, 2 * pi, 100)
# f = f.reshape((10, 10))
# print(f)

# a = np.array([20, 30, 40, 50])
# b = np.arange(4)
# print(a - b)
# print(b ** 3)
# print(a < 31)

# 矩阵的相乘
A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 3], [4, 5]])

print(A * B)
print(A.dot(B))

