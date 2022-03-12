# -*- coding: utf-8 -*-
import torch as t
from matplotlib import pyplot as plt
from IPython import display

# 设置随机种子确保在不同的计算机上运行时下面的输出一致
t.manual_seed(1000)


def generate_fake_data(batch_size=8):
    '''
    产生随机数据: y = x*2 + 3, 加上了一些噪声
    :param batch_size:
    :return:
    '''
    x = t.rand(batch_size, 1) * 20
    y = x * 2 + (1 + t.randn(batch_size, 1)) * 3
    return x, y


def train():
    # 随机初始化参数
    w = t.rand(1, 1)
    b = t.zeros(1, 1)

    lr = 0.001

    for epoch in range(1, 1000 + 1):
        x, y = generate_fake_data()

        # forward
        y_pred = x.mm(w) + b.expand_as(y)

        loss = 0.5 * (y_pred - y) ** 2
        loss = loss.sum()

        # backward() # 手动计算梯度
        dloss = 1
        dy_pred = dloss * (y_pred - y)

        dw = x.t().mm(dy_pred)
        db = dy_pred.sum()

        # 更新参数
        w.sub_(lr * dw)
        b.sub_(lr * db)

        print("w:{}, b:{}, loss:{}".format(w, b, loss))

    # print
    print("final w:{}, b:{}".format(w.squeeze(), b.squeeze()))


def train_use_autograd():
    # 随机初始化参数
    w = t.rand(1, 1, requires_grad=True)
    b = t.zeros(1, 1, requires_grad=True)

    lr = 0.001

    for ii in range(8000):
        x, y = generate_fake_data()

        # 计算loss
        y_pred = x.mm(w) + b.expand_as(y)
        loss = 0.5 * (y_pred - y) ** 2
        loss = loss.sum()

        # backward
        loss.backward()

        # update parameters
        w.data.sub_(lr * w.grad.data)
        b.data.sub_(lr * b.grad.data)

        # 梯度清零
        w.grad.zero_()
        b.grad.zero_()

        print("w:{}, b:{}, loss:{}".format(w, b, loss))
    print("final w:{}, b:{}".format(w.squeeze(), b.squeeze()))

if __name__ == "__main__":
    x, y = generate_fake_data()
    print("x: {}".format(x))
    print("y: {}".format(y))
    plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
    plt.show()
    train()
    # train_use_autograd()