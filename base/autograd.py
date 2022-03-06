# -*- coding: utf-8 -*-
import torch

# Pytorch自动求导模块

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


# init params
params = torch.tensor([1.0, 0.0], requires_grad=True)

# 一般来讲，所有PyTorch张量都有一个初始为空的名为grad的属性：
assert params.grad is None

loss = loss_fn(model(t_u, *params), t_c)
loss.backward()

print("param.grad: {}".format(params.grad))
# param.grad: tensor([4517.2969,   82.6000])

# 调用backward会导致导数值在叶节点处累积。所以将其用于参数更新后，需要将梯度显式清零。
if params.grad is not None:
    params.grad.zero_()
print("after clean,  param.grad: {}".format(params.grad))


def train_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            # 这个可以在调用backward()之前的循环中任何时候完成
            params.grad.zero_()
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()
        # 更新参数
        params = (params - learning_rate * params.grad).detach().requires_grad_()
        if epoch % 50 == 0:
            print("Epoch %d, Loss %f" % (epoch, loss))
    return params


t_un = 0.1 * t_u

train_loop(n_epochs=5000, learning_rate=1e-2, params=torch.tensor([1.0, 0.0], requires_grad=True), t_u=t_un, t_c=t_c)
# Epoch 50, Loss 25.710938
# Epoch 100, Loss 22.148710
# Epoch 150, Loss 19.143446
# Epoch 200, Loss 16.608067
# Epoch 250, Loss 14.469097
# Epoch 300, Loss 12.664559
# Epoch 350, Loss 11.142170
# Epoch 400, Loss 9.857802
# Epoch 450, Loss 8.774253
# Epoch 500, Loss 7.860115
# Epoch 550, Loss 7.088911
# Epoch 600, Loss 6.438284
# Epoch 650, Loss 5.889383
# Epoch 700, Loss 5.426309
# Epoch 750, Loss 5.035636
# Epoch 800, Loss 4.706046
# Epoch 850, Loss 4.427990
# Epoch 900, Loss 4.193405
# Epoch 950, Loss 3.995498
# Epoch 1000, Loss 3.828538
# Epoch 1050, Loss 3.687683
# Epoch 1100, Loss 3.568848
# Epoch 1150, Loss 3.468597
# Epoch 1200, Loss 3.384018
# Epoch 1250, Loss 3.312663
# Epoch 1300, Loss 3.252462
# Epoch 1350, Loss 3.201678
# Epoch 1400, Loss 3.158830
# Epoch 1450, Loss 3.122686
# Epoch 1500, Loss 3.092191
# Epoch 1550, Loss 3.066463
# Epoch 1600, Loss 3.044759
# Epoch 1650, Loss 3.026447
# Epoch 1700, Loss 3.011001
# Epoch 1750, Loss 2.997968
# Epoch 1800, Loss 2.986974
# Epoch 1850, Loss 2.977696
# Epoch 1900, Loss 2.969871
# Epoch 1950, Loss 2.963266
# Epoch 2000, Loss 2.957698
# Epoch 2050, Loss 2.953000
# Epoch 2100, Loss 2.949035
# Epoch 2150, Loss 2.945690
# Epoch 2200, Loss 2.942870
# Epoch 2250, Loss 2.940489
# Epoch 2300, Loss 2.938481
# Epoch 2350, Loss 2.936788
# Epoch 2400, Loss 2.935356
# Epoch 2450, Loss 2.934151
# Epoch 2500, Loss 2.933134
# Epoch 2550, Loss 2.932277
# Epoch 2600, Loss 2.931554
# Epoch 2650, Loss 2.930941
# Epoch 2700, Loss 2.930426
# Epoch 2750, Loss 2.929992
# Epoch 2800, Loss 2.929626
# Epoch 2850, Loss 2.929316
# Epoch 2900, Loss 2.929054
# Epoch 2950, Loss 2.928833
# Epoch 3000, Loss 2.928648
# Epoch 3050, Loss 2.928491
# Epoch 3100, Loss 2.928361
# Epoch 3150, Loss 2.928249
# Epoch 3200, Loss 2.928154
# Epoch 3250, Loss 2.928075
# Epoch 3300, Loss 2.928006
# Epoch 3350, Loss 2.927951
# Epoch 3400, Loss 2.927904
# Epoch 3450, Loss 2.927863
# Epoch 3500, Loss 2.927830
# Epoch 3550, Loss 2.927801
# Epoch 3600, Loss 2.927776
# Epoch 3650, Loss 2.927757
# Epoch 3700, Loss 2.927739
# Epoch 3750, Loss 2.927724
# Epoch 3800, Loss 2.927713
# Epoch 3850, Loss 2.927701
# Epoch 3900, Loss 2.927693
# Epoch 3950, Loss 2.927686
# Epoch 4000, Loss 2.927679
# Epoch 4050, Loss 2.927673
# Epoch 4100, Loss 2.927670
# Epoch 4150, Loss 2.927667
# Epoch 4200, Loss 2.927663
# Epoch 4250, Loss 2.927661
# Epoch 4300, Loss 2.927659
# Epoch 4350, Loss 2.927656
# Epoch 4400, Loss 2.927656
# Epoch 4450, Loss 2.927654
# Epoch 4500, Loss 2.927652
# Epoch 4550, Loss 2.927651
# Epoch 4600, Loss 2.927649
# Epoch 4650, Loss 2.927650
# Epoch 4700, Loss 2.927649
# Epoch 4750, Loss 2.927649
# Epoch 4800, Loss 2.927648
# Epoch 4850, Loss 2.927649
# Epoch 4900, Loss 2.927648
# Epoch 4950, Loss 2.927647
# Epoch 5000, Loss 2.927647


# 优化器
import torch.optim as optim

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)
t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)
optimizer.zero_grad()  # 此调用可以在循环中更早的位置
loss.backward()
optimizer.step()
print("params:{}".format(params))


# params:tensor([1.7761, 0.1064], requires_grad=True)


def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print("Epoch %d, Loss %f" % (epoch, loss))
    return params


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2

optimizer = optim.SGD([params], lr=learning_rate)
training_loop(n_epochs=5000, optimizer=optimizer, params=params, t_u=t_un, t_c=t_c)

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


# def training_loop(n_epochs, optimizer, params,
#                   train_t_u, val_t_u, train_t_c, val_t_c):
#     for epoch in range(1, n_epochs + 1):
#         train_t_p = model(train_t_u, *params)
#         train_loss = loss_fn(train_t_p, train_t_c)
#
#         val_t_p = model(val_t_u, *params)
#         val_loss = loss_fn(val_t_p, val_t_c)
#
#         optimizer.zero_grad()
#         train_loss.backward() # 注意没有val_loss.backward因为不能在验证集上训练模型
#         optimizer.step()
#
#         if epoch <= 3 or epoch % 500 == 0:
#             print('Epoch %d, Training loss %.2f, Validation loss %.2f' % (
#                     epoch, float(train_loss), float(val_loss)))
#     return params

# params = torch.tensor([1.0, 0.0], requires_grad=True)
# learning_rate = 1e-2
# optimizer = optim.SGD([params], lr=learning_rate)

# training_loop(
#     n_epochs = 3000,
#     optimizer = optimizer,
#     params = params,
#     train_t_u = train_t_un,
#     val_t_u = val_t_un,
#     train_t_c = train_t_c,
#     val_t_c = val_t_c)


# 不需要时关闭autograd

def training_loop(n_epochs, optimizer, params,
                  train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)

        with torch.no_grad():
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert val_loss.requires_grad == False

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print('Epoch %d, Training loss %.2f, Validation loss %.2f' % (
                epoch, float(train_loss), float(val_loss)))
    return params

print("=" * 100)
training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    params = params,
    train_t_u = train_t_un,
    val_t_u = val_t_un,
    train_t_c = train_t_c,
    val_t_c = val_t_c)

# 你可以定义一个calc_forward函数，该函数接受输入中的数据，
# 并根据布尔值is_train参数运行带或不带autograd的model和loss_fn

def calc_forward(t_u, t_c, is_train):
    with torch.set_grad_enabled(is_train):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
    return loss
