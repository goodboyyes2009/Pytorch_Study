# -*- coding: utf-8 -*-
import torch
from torch.autograd import Function

class Mul(Function):

    @staticmethod
    def forward(ctx, w, x, b, x_requires_grad=True):
        ctx.x_requires_grad = x_requires_grad
        ctx.save_for_backward(w, x)
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_variables
        grad_w = grad_output * x
        if ctx.x_requires_grad:
            grad_x = grad_output * w
        else:
            grad_x = None
        grad_b = grad_output * 1
        return grad_w, grad_x, grad_b, None

# 需要保存前向传播中的中间结果，使用Function.apply
class MultiplyAdd(Function):

    @staticmethod
    def forward(ctx, w, x, b):
        print('x type in forward {}'.format(type(x)))
        ctx.save_for_backward(w, x)
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_variables
        print("x type in backward {}".format(type(x)))

        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1
        return grad_w, grad_x, grad_b


x  = torch.ones(1)
w = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)
print("开始前向传播")
z = MultiplyAdd.apply(w, x, b)
print("开始反向传播")
z.backward(retain_graph = True)
print("x.grad:{}, w.grad:{}, b.grad:{}".format(x.grad, w.grad, b.grad))
apply_result = z.grad_fn.apply(torch.ones(1))
print("apply_result: {}".format(apply_result))

# 在backward函数中对variable操作是为了计算梯度的梯度
x = torch.tensor([5], dtype=torch.float64, requires_grad=True)
y = x ** 2

grad_x = torch.autograd.grad(y, x, create_graph=True)
print("dy/dx = {}".format(grad_x))
grad_grad_x = torch.autograd.grad(grad_x[0], x)
print("二阶导数： {}".format(grad_grad_x))


# 使用Function实现Sigmoid
class Sigmoid(Function):

    @staticmethod
    def forward(ctx, x):
        output = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_variables

        grad_x = output * (1 - output) * grad_output
        return grad_x


test_input = torch.randn((3,4), requires_grad=True)

print(torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-3))

def f_sigmoid(x):
    y = Sigmoid.apply(x)
    y.backward(torch.ones(x.size()))

def f_naive(x):
    y = 1/(1 + torch.exp(-x))
    y.backward(torch.ones(x.size()))

def f_th(x):
    y = torch.sigmoid(x)
    y.backward(torch.ones(x.size()))

xx = torch.randn(100,100, requires_grad=True)
import timeit
print("f_sigmoid(x), take: {}".format(timeit.timeit(stmt='list(f_sigmoid(xx) for n in range(100))', setup='from __main__ import f_sigmoid; import torch; from __main__ import xx;',number=100)))
print("f_naive(x), take: {}".format(timeit.timeit(stmt='list(f_naive(xx) for n in range(100))', setup='from __main__ import f_naive; import torch; from __main__ import xx;',number=100)))
print("f_th(x), take: {}".format(timeit.timeit(stmt='list(f_th(xx) for n in range(100))', setup='from __main__ import f_th; import torch; from __main__ import xx;',number=100)))

# f_sigmoid(x), take: 1.3120622370006458
# f_naive(x), take: 1.30623501700029
# f_th(x), take: 1.038511530001415

