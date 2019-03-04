import numpy
import scipy.special


class NueralNetwork(object):
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learning_rate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learning_rate

        # 初始化 链接权重 矩阵，并随机赋值权重初始值:均值为0方差为链接数量平方根的倒数(pow(nodes,-0.5))的正态分布中取样
        self.w_ih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.w_ho = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        


    # train the nueral network
    def train(self):
        pass

    # query the outputs
    def query(self, inputs):
        self.activ_func = lambda x: scipy.special.exp(x)
        hidden_inputs = numpy.dot(self.w_ih, inputs)
        hidden_outputs = self.activ_func(hidden_inputs)

        final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activ_func(final_inputs)

        return final_outputs


if __name__ == '__main__':
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.5
    # 初始化 一个3*3的网络，学习率为0.5
    n = NueralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print(scipy.rand())
