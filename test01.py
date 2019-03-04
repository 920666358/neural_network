import matplotlib.pyplot
import numpy
import pylab

# a = numpy.zeros([2, 3])
# print(a)
# p = matplotlib.pyplot.imshow(a, interpolation='nearest')
# print(p)

with open('mnist_train_100.csv', 'r') as f:
    data_l = f.readlines()
# print(data_l)
# print(len(data_l))

all_values = data_l[0].split(',')
print(all_values)
# 通过缩放和位移，将输入值整体偏移至所需范围（0.0，1.0）
image_array = ((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01).reshape((28, 28))
print(image_array)

matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
pylab.show()
