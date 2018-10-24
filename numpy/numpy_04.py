import numpy as np

# B = np.arange(3)
# print(B)
# print(np.exp(B))
# print(np.sqrt(B))

# a = np.floor(10 * np.random.random((3, 4)))
# print(a)
# print("---------")
# # 拉成向量
# print(a.ravel())
# a.shape = (6, 2)
# print(a)
# print("---------")
# print(a.T)
#
# a.reshape(3, -1)

# 矩阵的拼接
# a = np.floor(10 * np.random.random((2, 2)))
# b = np.floor(10 * np.random.random((2, 2)))
# print(a)
# print(b)
# print(np.hstack((a, b)))
# print(np.vstack((a, b)))

# 矩阵的分割
# a = np.floor(10 * np.random.random((2, 12)))
# print(a)
# print(np.hsplit(a, (3, 4)))

# ndarray 三种复制的方式
# a = np.arange(12)
# b = a
# print(b is a)
# b.shape = 3, 4
# print(id(b))
# print(id(a))

# a = np.arange(12)
# c = a.view()
# print(c is a)
# c.shape = 2, 6
# print(a.shape)
# print(a.shape)
# c[0, 4] = 1234
# print(a)
# print(c)
# print(id(a))
# print(id(c))

# a = np.arange(12)
# d = a.copy()
# d.shape = 3, 4
# print(d is a)
# d[0, 0] = 9999
# print(a)
# print(d)
# data = np.sin(np.arange(20)).reshape(5, 4)
# print(data)
# ind = data.argmax(axis=0)
# print(ind)
# data_max = data[ind, range(data.shape[1])]
# # print(data_max)
# a = np.arange(0, 40, 10)
# print(a)
# b = np.tile(a, (1, 0))
# print(b)
# a = np.array([[4, 3, 5], [2, 1, 6]])
# a.sort(axis=1)
# print(a)
a = np.array([4, 3, 1, 2])
j = np.argsort(a)
print(j)
print(a[j])
