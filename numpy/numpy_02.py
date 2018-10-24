import numpy

vector = numpy.array([5, 10, 15, 20])
# print(vector == 10)
# equal_to_ten = (vector == 10)
# print(vector[equal_to_ten])
# equal_to_ten_or_five = (vector == 10) | (vector == 5)
# print(equal_to_ten_or_five)
# vector[equal_to_ten_or_five] = 50
# print(vector)

vector = numpy.array(["1", "2", "3"])
print(vector.dtype)

vector = vector.astype(float)
print(vector.dtype)
print(vector)
print(vector.max())

# 按列求和
matrix = numpy.array([[1, 5, 10],
                      [15, 20, 25],
                      [30, 35, 40]])
# axis=0 按列相加 axis=1 按行相加
print(matrix.sum(axis=0))
