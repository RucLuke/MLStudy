import numpy

world_alcohol = numpy.genfromtxt("world_alcohol.txt", delimiter=",", dtype=str)
print(world_alcohol[2, 4])
# print(type(world_alcohol))
# print(world_alcohol)
# print(help(numpy.genfromtxt))
# vector = numpy.array([1, 10, 15, 20])
matrix = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(vector.shape)
print(matrix.shape)
print(matrix[:, 1])

# numpy.array 里面的数据必须是同一个类型的
numbers = numpy.array([1, 2, 3, 4])
print(numbers.dtype)
print(numbers)
print(numbers[1:2])
