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

    def test(self, inputs):
        pass


if __name__ == '__main__':
    input_nodes = 784
    # hidden_nodes = 100   # 调整节点数，改变网络形状
    hidden_nodes = 200
    output_nodes = 10
    # learning_rate = 0.5
    # learning_rate = 0.3
    learning_rate = 0.1
    # learning_rate = 0.2
    # 初始化网络，学习率为0.5|0.3|0.1
    n = NueralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读取并准备输入数据
    with open('..\data\mnist_train.csv', 'r') as f:
        training_data = f.readlines()

    # 设置循环次数，即训练世代数
    epochs = 5  # 通过多次试错，找出甜蜜点（最佳值）
    for i in range(epochs):
        for data in training_data:
            all_values = data.split(',')
            inputs = (numpy.asfarray(all_values[1:])/255*0.99)+0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            # training the network
            n.train(inputs, targets)

    # 使用测试集进行测试，并计算预测准确率
    with open('..\data\mnist_test.csv', 'r') as f:
        test_data = f.readlines()
    score = []  # 用于记录预测结果是否正确
    for data in test_data:
        all_values = data.split(',')
        # print(all_values[0])  # 正确的标签
        test_output = n.query((numpy.asfarray(all_values[1:])/255*0.99)+0.01)
        # print(test_output)   # 预测的标签
        predic_lable = test_output.tolist().index(max(test_output.tolist()))
        # print(predic_lable)
        # matplotlib.pyplot.imshow(numpy.asfarray(all_values[1:]).reshape((28, 28)), cmap='Greys', interpolation='None')
        # pylab.show()

        if int(predic_lable) == int(all_values[0]):
            score.append(1)
        else:
            score.append(0)

    accuracy = score.count(1)/len(score)
    print(accuracy)
