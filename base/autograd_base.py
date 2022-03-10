# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
from torch.autograd import variable

x = torch.ones(1)
b = torch.rand(1, requires_grad=True)
w = torch.rand(1, requires_grad=True)
print("x:{}, b:{}".format(x, b))
assert x.requires_grad == False, b.requires_grad == True

y = w.matmul(x)
z = y + b
assert y.requires_grad == True, z.requires_grad == True

# 叶子节点
assert x.is_leaf
assert w.is_leaf
assert b.is_leaf
assert not y.is_leaf, not z.is_leaf

print(z.grad_fn)
print(z.grad_fn.next_functions)
assert z.grad_fn.next_functions[0][0] == y.grad_fn

# 第一个是w，叶子节点，需要求导，梯度是累加的
# 第二个是x,叶子节点，不需要求导，梯度的事None
print("y.grad_fn.next_functions:{}".format(y.grad_fn.next_functions))

assert x.grad_fn is None, w.grad_fn is None

z.backward(retain_graph=True)
print("w.grad :{}".format(w.grad))
z.backward()
# 多次反向传播，梯度会累加
print("w.grad :{}".format(w.grad))


# PyTorch使用的是动态图，它的计算图在每次前向传播时都有从头开始构建的
def abs(x):
    if x.data[0] > 0:
        return x
    else:
        return -x


# x = torch.ones(1, requires_grad=True)
# y = abs(x)
#
# y.backward()
# print("x.grad: {}".format(x.grad))

# x.grad: tensor([1.])

x = -1 * torch.ones(1, requires_grad=True)
print("x: {}".format(x))
y = abs(x)
y.backward()
print("x.grad: {}".format(x.grad))


# x.grad: None

def f(x):
    result = 1
    for ii in x:
        if ii > 0:
            result = ii * result
    return result


x = torch.range(-2, 4, requires_grad=True)
print("x: {}".format(x))
y = f(x)  # y = x[3] * x[4] * x[5]
y.backward()
print("x.grad:{}".format(x.grad))
print("y： {}".format(y))


# 使用hook方法
x = torch.ones(3, requires_grad=True)
w = torch.rand(3, requires_grad=True)

y = x * w
z = y.sum()

z.backward()
# 非叶子节点grad计算完后自动清空, y.grad是None
print(x.grad, w.grad, y.grad, z.grad)

# 若想查看非叶子节点的梯度信息使用hook方法
def variable_hook(grad):
    print("y 的梯度: {}".format(grad))


y = w * x

hook_handle = y.register_hook(variable_hook)
z = y.sum()
z.backward()

# 除非你每次都要调用hook, 不然用完后记得移除hook
hook_handle.remove()

x = torch.arange(0, 3, dtype=torch.float, requires_grad=True)
y = x ** 2 + x* 2

z = y.sum()
z.backward(retain_graph = True)
print("x.grad: {}".format(x.grad))
# x.grad: tensor([2., 4., 6.])

y_grad_variables = torch.Tensor([1,1,1]) # dz/dy
y.backward(y_grad_variables) # 从y开始反向传播
print("x.grad: {}".format(x.grad))