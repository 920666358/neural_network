import numpy
import scipy.special, pylab
import matplotlib.pyplot


class NueralNetwork(object):
    # initialise the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate
        self.activation_func = lambda x: scipy.special.expit(x)

        # 初始化 链接权重 矩阵，并随机赋值权重初始值:均值为0方差为链接数量平方根的倒数(pow(nodes,-0.5))的正态分布中取样
        self.w_ih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_ho = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

    # train the nueral network
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)
        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.w_ho.T, output_errors)
        # 根据误差值，调整权重值（即训练过程）
        self.w_ho += self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)), numpy.transpose(hidden_outputs))
        self.w_ih += self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)), numpy.transpose(inputs))


    # query the outputs
    def query(self, inputs):
        hidden_inputs = numpy.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs


if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    # 初始化网络，学习率为0.3
    n = NueralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读取并准备输入数据
    with open('mnist_train_100.csv', 'r') as f:
        training_data = f.readlines()
    for data in training_data:
        all_values = data.split(',')
        inputs = (numpy.asfarray(all_values[1:])/255*0.99)+0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

    # print(n.w_ih)
    # print('*'*50)
    # print(n.w_ho)

    with open('mnist_test_10.csv', 'r') as f:
        test_data = f.readlines()
    # for data in test_data:
    all_values = test_data[0].split(',')
    test_output = n.query((numpy.asfarray(all_values[1:])/255*0.99)+0.01)
    print(test_output)
    matplotlib.pyplot.imshow(numpy.asfarray(all_values[1:]).reshape((28, 28)), cmap='Greys', interpolation='None')
    pylab.show()

    print(test_output.tolist().index(max(test_output.tolist())))
