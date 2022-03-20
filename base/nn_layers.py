# -*- coding: utf-8 -*-
import torch.nn as nn
import torch


class Affine(nn.Module):
    """
    实现全连接层, y=x * W_t + b
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b.expand_as(x)


def test_Affine():
    # 输入特征数量是3， 输出特征数量是4
    affine = Affine(3, 4)
    x = torch.randn(2, 3, requires_grad=True)
    output = affine(x)

    # 直接使用nn.Linear
    linear = nn.Linear(3, 4)
    linear_output = linear(x)
    print("output: {}\nlinear_output: {}".format(output, linear_output))


class Perceptron(nn.Module):
    """
    多层感知机
    """

    def __init__(self, input_features, hidden_features, output_features):
        nn.Module.__init__(self)
        self.layer1 = Affine(input_features, hidden_features)
        self.layer2 = Affine(hidden_features, output_features)

    def forward(self, input):
        x1 = self.layer1(input)
        output = self.layer2(x1)
        return output

    def print_parameters(self):
        for name, patameter in self.named_parameters():
            print("name:{}, type:{}".format(name, patameter.size()))


class NN_Perecptron(nn.Module):

    def __init__(self, input_features, hidden_features, output_features):
        nn.Module.__init__(self)
        self.hidden_layer = nn.Linear(in_features=input_features, out_features=hidden_features, bias=True)
        self.output_layer = nn.Linear(in_features=hidden_features, out_features=output_features, bias=True)

    def forward(self, input: torch.Tensor):
        hidden_output = self.hidden_layer(input)
        output = self.output_layer(hidden_output)
        return output

    def print_parameters(self):
        for name, patameter in self.named_parameters():
            print("name:{}, type:{}".format(name, patameter.shape))


def test_Perceptron():
    input = torch.randn(5, 3)
    perceptron = Perceptron(3, 4, 2)
    perceptron.print_parameters()
    output = perceptron(input)
    print("output:{}".format(output))

    print("==" * 30)

    nn_perceptron = NN_Perecptron(input_features=3, hidden_features=4, output_features=2)
    print("output: {}".format(nn_perceptron(input)))
    nn_perceptron.print_parameters()

    # name:hidden_layer.weight, type:torch.Size([4, 3]) # 这里的[4,3]而不是[3,4]是由于x乘的是W转置的原因
    # name:hidden_layer.bias, type:torch.Size([4])
    # name:output_layer.weight, type:torch.Size([2, 4])
    # name:output_layer.bias, type:torch.Size([2])


def test_batch_norm():
    # 4 Channel
    bn1d = nn.BatchNorm1d(4)
    bn1d.weight.data = torch.ones(4) * 4
    bn1d.bias.data = torch.zeros(4)

    input_x = torch.randn(2, 4)
    bn_out = bn1d(input_x)
    print("bn_out: {}".format(bn_out))
    print("bn_out mean(0): {}\nbn_out.var() :{}".format(bn_out.mean(0), bn_out.var(unbiased=False)))


def test_dropout():
    # 每个元素以0.5的概率舍弃
    dropout = nn.Dropout(0.5)
    input_x = torch.randn(3, 4)
    print("before dropout, input_x:{}".format(input_x))
    o = dropout(input_x)
    print("after dropout, input_x:{}".format(o))


def test_RNN():
    """
     h_t = \text{tanh}(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})
    :return:
    """
    rnn = nn.RNN(input_size=100, hidden_size=50, num_layers=1, bias=True, batch_first=True, dropout=0.5,
                 nonlinearity="relu")
    # (batch, seq_len input_size)
    input_x = torch.rand((10, 20, 100))

    # num_layers * num_directions, batch, hidden_size
    h_0 = torch.zeros((1 * 1, 10, 50))

    # output size: [seq_len, batch, num_directions * hidden_size]
    # h_n size: [num_layers * num_directions, batch, hidden_size]
    output, h_n = rnn(input_x, h_0)

    print("output size:{}\nh_n size:{}".format(output.size(), h_n.size()))

    output = output.transpose(0, 1)
    print("output shape:{}".format(output.shape))

    id(output[-1, :, :].storage()) == id(h_n.storage())

    print("====" * 4 + "RNN 参数 size" + "====" * 4)
    for name, parameter in rnn.named_parameters():
        print("name:{}, parameter size:{}".format(name, parameter.shape))


def test_lstm():
    torch.manual_seed(1000)
    # input shape: [batch, seq_len, input_size] batch_first=True
    # seq可以理解为时间序列的长度，比如在NLP里面可以代表一句话的长度
    input_x = torch.randn((2, 3, 4))

    lstm = nn.LSTM(input_size=4, batch_first=True, hidden_size=5, num_layers=1)

    # h_0 shape: [num_layers * num_directions, batch, hidden_size]
    h_0 = torch.randn(1, 2, 5)
    # c_0 shape: [num_layers * num_directions, batch, hidden_size]
    c_0 = torch.rand(1, 2, 5)

    # output shape: [batch, seq_len, num_directions * hidden_size]
    # h_n shape: [num_layers * num_directions, batch, hidden_size]
    # c_n shape: [num_layers * num_directions, batch, hidden_size]
    output, (h_n, c_n) = lstm(input_x, (h_0, c_0))

    last_out = output[-1, :, :]

    print("last_out: {}".format(last_out))
    print("tanh(c_n): {}".format(torch.tanh(c_n.transpose(0, 1))))
    print("out_last :{}".format(h_n / torch.tanh(c_n)))

    # print("h_last: {}".format(h_last))
    print("output:{}\nh_n shape:{}\nh_n: {}\nc_n:shape:{}\nc_n:{}".format(output.shape, h_n.shape, h_n, c_n.shape, c_n))


def test_embedding():
    embedding = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)
    # 可以用预训练好的词向量初始化embedding
    embedding.weight.data = torch.arange(0, 30).view(10, 3)

    input_x = torch.arange(7, 0, -1).long()
    output = embedding(input_x)
    print("output: {}".format(output))

    embedding = nn.Embedding(5, 3)
    input_ids = torch.LongTensor([[1, 2, 3, 4], [0, 1, 2, 4]])
    print("input_ids shape:{}".format(input_ids.shape))
    out = embedding(torch.LongTensor(input_ids))
    print("out:{}\nout shape:{}".format(out, out.shape))


def conv1_test():
    conv1 = nn.Conv1d(in_channels=128, out_channels=100, kernel_size=3, stride=1, padding=3)
    # N, C, L = batch_size, in_channel, sequence_len
    input_x = torch.randn(100, 128, 63)
    # 计算输出的特征大小: (L_in - kernel_size +  2 * padding)/ stride + 1
    #
    conv1_out = conv1(input=input_x)
    print("conv1_out shape: {}".format(conv1_out.shape))


if __name__ == "__main__":
    # test_Perceptron()
    # test_batch_norm()
    # test_dropout()
    # test_RNN()
    # test_lstm()
    # test_embedding()
    conv1_test()
