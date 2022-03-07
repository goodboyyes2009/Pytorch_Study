# -*- coding: utf-8 -*-
import torch
import numpy as np

# 使用storage属性访问张量
points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
print("points: {}".format(points))

# 转置操作
points_t = points.t()
print("points_t:{}".format(points_t))

# 共享同一块内存
id(points.storage()) == id(points_t.storage())

# points是连续的，但其转置不是
print("{},{}".format(points.is_contiguous(), points_t.is_contiguous()))
# >> True, False


# 使用contiguous方法从非连续张量获得新的连续张量。 张量的内容保持不变，但步长发生变化，存储也是如此
points_t_cont = points_t.contiguous()
print("points_t_cont is contiguous? {}".format(points_t_cont.is_contiguous()))
# >> points_t_cont is contiguous? True

storage = points.storage()
print("storage:{}".format(storage))

# storage是占用连续内存，可以使用下标实现随机访问
second_point = storage[1]
assert storage[0] == 1.0
assert second_point == 4.0
assert storage[2] == 2.0
assert storage[5] == 5.0

storage[0] = 2.0

print("storage:{}".format(storage))

# 以下操作在torch 1.7中没有这些属性
# print("second point offset:{}".format(second_point.storage_offset()))
# print("second point shape:{}".format(second_point.shape()))
# print("second point size:{}".format(second_point.size()))

# 步长是一个元组，表示当索引在每个维度上增加1时必须跳过的存储中元素的数量。
# 用下标i和j访问二维张量等价于访问存储中的storage_offset + stride[0] * i + stride[1] * j元素。

# 存储将points张量中的元素【逐行】保存着
print("point stride:{}".format(points.stride()))

some_tensor = torch.ones(2, 4, 5)

print("some_tensor.shape:{}, some_tensor.stride:{}".format(some_tensor.shape, some_tensor.stride()))
# >> some_tensor.shape:torch.Size([2, 4, 5]), some_tensor.stride:(20, 5, 1)

# 将第一个维度和第三个维度进行对调
some_tensor_t = some_tensor.transpose(0, 2)
print("some_tensor_t.shape:{}, some_tensor.stride:{}".format(some_tensor_t.shape, some_tensor_t.stride()))
# >> some_tensor_t.shape:torch.Size([5, 4, 2]), some_tensor.stride:(1, 5, 20)

# 索引张量
print("第一行以及之后所有的行,默认所有的列,points[1:]: {}".format(points[1:]))
# >> tensor([[2., 1.],[3., 5.]])
print("第一行以及之后所有的行,所有的列,points[1:, :]: {}".format(points[1:, :]))
# >> tensor([[2., 1.],[3., 5.]])
print("第一行以及之后所有的行,仅第0列,points[1:, 0]: {}".format(points[1:, 0]))
# >> tensor([2., 3.])


# 使用numpy调用tensor中数组, 返回的数组与张量存储共享一个基础缓冲区。
# 如果在GPU上分配了张量，（调用numpy方法时）PyTorch会将张量的内容复制到在CPU上分配的NumPy数组中。
print("use numpy to access tensor:{}.".format(points.numpy()))
# >> [[2. 4.][2. 1.][3. 5.]].

# 使用numpy数组创建tensor
np_array = np.empty((3, 4))
points_from_numpy = torch.from_numpy(np_array)
print("points_from_numpy:{}".format(points_from_numpy))
# >> tensor([[0., 0., 1., 1.],
#        [1., 1., 1., 1.],
#        [1., 0., 0., 1.]], dtype=torch.float64)

# 张量的序列化
torch.save(points, '../data/tensor/ourpoints.t')

# 等价于以下代码
with open('../data/tensor/ourpoints.t_1', 'wb') as f:
    torch.save(points, f)

# 加载
load_points = torch.load('../data/tensor/ourpoints.t')

# or
with open('../data/tensor/ourpoints.t', 'rb') as f:
    load_points = torch.load(f)
print("load_points:{}".format(load_points))

# 将张量转移到GPU上运行
points_gpu = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]], device='cuda')
print("points_gpu:{}".format(points_gpu))

points_gpu_1 = points.to(device='cuda:0')
print("points_gpu_1:{}".format(points_gpu_1))

points = 2 * points  # 在CPU上做乘法
points_gpu = 2 * points.to(device='cuda')  # 在GPU上做乘法

# 请注意，当计算结果产生后，points_gpu的张量并不会返回到CPU。这里发生的是以下三个过程：
# 1.将points张量复制到GPU
# 2.在GPU上分配了一个新的张量，并用于存储乘法的结果
# 3.返回该GPU张量的句柄

# 如果要将张量移回CPU，你需要为to方法提供一个cpu参数,还可以修改数据类型
points_cpu = points_gpu.to(device='cpu', dtype=torch.int32)
print("points_cpu:{}".format(points_cpu))


# 练习
nparray = np.empty((3,3))
a = torch.from_numpy(nparray)
print("a:{}".format(a))

b = a.view(3,3)
print("b[1,1]:{}".format(b[1,1]))

c = b[1:, 1:]
#storage_offset: 存储偏移是存储中与张量中的第一个元素相对应的索引
print("c:{}, stride:{}, offerset:{}".format(c, c.stride(), c.storage_offset()))

# 直接修改输入而不是创建新的输出并返回
print("a:{},\n b.sin_():{}".format(a, b.sin_()))