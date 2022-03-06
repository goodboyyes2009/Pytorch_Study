# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim

# PyTorch的nn模块
x = torch.ones(1)
linear_model = nn.Linear(1, 1)  # 参数: input size, output size, bias(默认True)
print("linear_model(x): {}".format(linear_model(x)))
# linear_model(x): tensor([1.2192], grad_fn=<AddBackward0>)
print("weight: {}".format(linear_model.weight))
# weight: Parameter containing:
# tensor([[0.9374]], requires_grad=True)

print("bias:{}".format(linear_model.bias))
# bias:Parameter containing:
# tensor([0.2818], requires_grad=True)

x = torch.ones(10, 1)
print("linear_model(x) : {}".format(linear_model(x)))
# linear_model(x) : tensor([[0.1775],
#         [0.1775],
#         [0.1775],
#         [0.1775],
#         [0.1775],
#         [0.1775],
#         [0.1775],
#         [0.1775],
#         [0.1775],
#         [0.1775]], grad_fn=<AddmmBackward>)

# 使用nn.Linear重写
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1)  # <1>
t_u = torch.tensor(t_u).unsqueeze(1)  # <1>

print("t_u.shape: {}".format(t_u.shape))
# t_u.shape: torch.Size([11, 1])

optimizer = optim.SGD(params=linear_model.parameters(), lr=1e-2)


def training_loop(n_epoch, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
    for epoch in range(1, n_epoch + 1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)

        t_p_val = model(t_u_val)
        loss_val = loss_fn(t_p_val, t_c_val)

        optimizer.zero_grad()
        loss_train.backward()

        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print('Epoch %d, Training loss %.4f, Validation loss %.4f' % (epoch, float(loss_train), float(loss_val)))


# 拆分数据集
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

# 划分结果是随机的
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

print("train_indices: {}\nval_indices: {}".format(train_indices, val_indices))

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u

training_loop(
    n_epoch=3000,
    optimizer=optimizer,
    model=linear_model,
    loss_fn=nn.MSELoss(),
    t_u_train=train_t_un,
    t_u_val=val_t_un,
    t_c_train=train_t_c,
    t_c_val=val_t_c,
)
print()
print(linear_model.weight)
print(linear_model.bias)

# Epoch 1, Training loss 163.7547, Validation loss 184.6521
# Epoch 1000, Training loss 3.7418, Validation loss 2.2593
# Epoch 2000, Training loss 3.1955, Validation loss 2.2075
# Epoch 3000, Training loss 3.1870, Validation loss 2.2013
#
# Parameter containing:
# tensor([[5.3075]], requires_grad=True)
# Parameter containing:
# tensor([-17.2587], requires_grad=True)

# nn提供了一种通过nn.Sequential容器串联模块的简单方法：
seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
print("seq_model: {}".format(seq_model))
# seq_model: Sequential(
#   (0): Linear(in_features=1, out_features=13, bias=True)
#   (1): Tanh()
#   (2): Linear(in_features=13, out_features=1, bias=True)
# )

# 调用model.parameters()函数可以得到第一线性和第二线性模块的权重和偏差
param_list = [param.shape for param in seq_model.parameters()]
print("parameters shape: {}".format(param_list))
# parameters shape: [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]

# 实际上，Sequential中每个模块的名称都是该模块在参数中出现的顺序。
# 有趣的是，Sequential还可以接受OrderedDict作为参数，这样就可以给Sequential的每个模块命名：

from collections import OrderedDict

seq_model = nn.Sequential(
    OrderedDict([
        ('hidden_linear', nn.Linear(1, 8)),
        ('hidden_activation', nn.Tanh()),
        ("output_linear", nn.Linear(8, 1))
    ])
)

print("seq_model:{}".format(seq_model))
# seq_model:Sequential(
#   (hidden_linear): Linear(in_features=1, out_features=8, bias=True)
#   (hidden_activation): Tanh()
#   (output_linear): Linear(in_features=8, out_features=1, bias=True)
# )

for name, param in seq_model.named_parameters():
    print(name, param.shape)

# hidden_linear.weight torch.Size([8, 1])
# hidden_linear.bias torch.Size([8])
# output_linear.weight torch.Size([1, 8])
# output_linear.bias torch.Size([1])

# 可以通过访问子模块来访问特定的参数，就像它们是属性一样：

print("seq_model.output_linear.bias:{}".format(seq_model.output_linear.bias))
# seq_model.output_linear.bias:Parameter containing:
# tensor([-0.0001], requires_grad=True)

optimizer = optim.SGD(seq_model.parameters(), lr=1e-3)  # 为了稳定性调小了梯度

training_loop(
    n_epoch=3000,
    optimizer=optimizer,
    model=seq_model,
    loss_fn=nn.MSELoss(),
    t_u_train=train_t_un,
    t_u_val=val_t_un,
    t_c_train=train_t_c,
    t_c_val=val_t_c,
)

print("output", seq_model(val_t_un))
print("answer", val_t_c)
print("hidden", seq_model.hidden_linear.weight.grad)


# Epoch 1, Training loss 185.0161, Validation loss 112.6343
# Epoch 1000, Training loss 5.8437, Validation loss 1.3130
# Epoch 2000, Training loss 3.6705, Validation loss 1.3510
# Epoch 3000, Training loss 2.2396, Validation loss 2.7712
# output tensor([[-1.9794],
#         [12.4468]], grad_fn=<AddmmBackward>)
# answer tensor([[-4.],
#         [15.]])
# hidden tensor([[ 1.1246e+00],
#         [ 2.0605e+01],
#         [ 1.0027e-03],
#         [-1.3967e+00],
#         [ 1.2101e+01],
#         [-2.0412e+01],
#         [ 8.3988e-05],
#         [-1.9416e+01]])


# from matplotlib import pyplot as plt
#
# t_range = torch.arange(20., 90.).unsqueeze(1)
#
# fig = plt.figure(dpi=100)
# plt.xlabel("Fahrenheit")
# plt.ylabel("Celsius")
# plt.plot(t_u.numpy(), t_c.numpy(), 'o')
# plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
# plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')
# plt.show()

# 自定义nn.Module的子类
class SubClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 23)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(23, 1)

    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t


subclass_model = SubClassModel()
print("subclass_model: {}".format(subclass_model))

# subclass_model: SubClassModel(
#   (hidden_linear): Linear(in_features=1, out_features=23, bias=True)
#   (hidden_activation): Tanh()
#   (output_linear): Linear(in_features=23, out_features=1, bias=True)
# )

for name, param in subclass_model.named_parameters():
    print(name, param.shape)
