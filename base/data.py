# -*- coding: utf-8 -*-
import math

import torch
from torch.utils.data import WeightedRandomSampler, BatchSampler, SequentialSampler


# torch.utils.data.IterableDataset 使用, 参考官网文档: https://pytorch.org/docs/1.2.0/data.html#torch.utils.data.IterableDataset

# 方法1: 通过__iter__()方法控制不同的work取数的行为,防止取到重复的数据

class MyIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start
        print("=== MyIterableDataset init ====")
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # 说明只有一个work处理数据,返回全部的iterator
            iter_start = self.start
            iter_end = self.end
        else:
            # split workload, 每一个work取数的数量
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            print("每一个worker需要处理的数据量: {}".format(per_worker))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
            print("当前worker的id=[{}], 处理的数据区间[{},{})".format(worker_id, iter_start, iter_end))
        return iter(range(iter_start, iter_end))


ds = MyIterableDataset(start=3, end=7)
# 一个worker处理的情况
print("===== 一个worker处理的情况")
print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
# [tensor([3]), tensor([4]), tensor([5]), tensor([6])]

# 两个worker处理的情况
print("\n===== 两个worker处理的情况")
print(list(torch.utils.data.DataLoader(ds, num_workers=2)))

# 每一个worker需要处理的数据量: 2
# 当前worker的id=[0], 处理的数据区间[3,5)
# 每一个worker需要处理的数据量: 2
# 当前worker的id=[1], 处理的数据区间[5,7)
# [tensor([3]), tensor([5]), tensor([4]), tensor([6])]

# 更多worker处理的情况
print("\n===== 更多worker处理的情况")
print(list(torch.utils.data.DataLoader(ds, num_workers=20)))


# 每一个worker需要处理的数据量: 1
# 当前worker的id=[0], 处理的数据区间[3,4)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[1], 处理的数据区间[4,5)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[2], 处理的数据区间[5,6)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[3], 处理的数据区间[6,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[4], 处理的数据区间[7,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[5], 处理的数据区间[8,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[6], 处理的数据区间[9,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[7], 处理的数据区间[10,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[8], 处理的数据区间[11,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[9], 处理的数据区间[12,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[10], 处理的数据区间[13,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[11], 处理的数据区间[14,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[12], 处理的数据区间[15,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[13], 处理的数据区间[16,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[14], 处理的数据区间[17,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[15], 处理的数据区间[18,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[16], 处理的数据区间[19,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[17], 处理的数据区间[20,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[18], 处理的数据区间[21,7)
# 每一个worker需要处理的数据量: 1
# 当前worker的id=[19], 处理的数据区间[22,7)
# [tensor([3]), tensor([4]), tensor([5]), tensor([6])]


# 方法2： 通过worker_init_fn函数切分不同worker的取数
class AnotherIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, start, end):
        print("=== AnotherIterableDataset init ===")
        super(AnotherIterableDataset).__init__()
        assert end > start
        self.start = start
        self.end = end

    def __iter__(self):
        print("AnotherIterableDataset处理的数据范围:[{},{})".format(self.start, self.end))
        return iter(range(self.start, self.end))


print("\n============ AnotherIterableDataset")
ds = AnotherIterableDataset(start=3, end=7)
# 一个work处理的情况
print("===== 一个worker处理的情况")
print(list(torch.utils.data.DataLoader(ds, num_workers=0)))

print("===== 三个worker处理的情况")
# 出现了数据的重复
print(list(torch.utils.data.DataLoader(ds, num_workers=3)))


# AnotherIterableDataset处理的数据范围:[3,7)
# AnotherIterableDataset处理的数据范围:[3,7)
# AnotherIterableDataset处理的数据范围:[3,7)
# [tensor([3]), tensor([3]), tensor([3]), tensor([4]), tensor([4]), tensor([4]), tensor([5]), tensor([5]), tensor([5]), tensor([6]), tensor([6]), tensor([6])]


# 定义一个worker_init_fn,控制每一个worker的取数范围,以避免取到重复数据
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    # 当前worker获取的数据集,是整个data的副本
    dataset = worker_info.dataset
    overall_start = dataset.start
    overall_end = dataset.end

    # 计算每一个worker处理的数据量
    per_worker = int(math.ceil(overall_end - overall_start) / float(worker_info.num_workers))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


print("\n===== 使用了worker_init_fn后， 两个worker处理的情况")
print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))

print("\n===== 使用了worker_init_fn后， 20个worker处理的情况")
print(list(torch.utils.data.DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))


# AnotherIterableDataset处理的数据范围:[3,5)
# AnotherIterableDataset处理的数据范围:[5,7)
# [tensor([3]), tensor([5]), tensor([4]), tensor([6])]


## Memory Pinning

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)

from torch.utils.data import TensorDataset, DataLoader

# TensorDataset可以类比zip来理解它的功能
dataset = TensorDataset(inps, tgts)
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper, pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print("batch index:{}, sample.inp is_pinned: {}".format(batch_ndx, sample.inp.is_pinned()))
    print("batch index:{}, sample.tgt is_pinned: {}".format(batch_ndx, sample.tgt.is_pinned()))

# batch index:0, sample.inp is_pinned: True
# batch index:0, sample.tgt is_pinned: True
# batch index:1, sample.inp is_pinned: True
# batch index:1, sample.tgt is_pinned: True
# batch index:2, sample.inp is_pinned: True
# batch index:2, sample.tgt is_pinned: True
# batch index:3, sample.inp is_pinned: True
# batch index:3, sample.tgt is_pinned: True
# batch index:4, sample.inp is_pinned: True
# batch index:4, sample.tgt is_pinned: True

# num_samples <= len(weights)
# replacement=False是没有放回的抽样
not_replacement = list(WeightedRandomSampler(weights=[0.1, 0.9, 0.4, 0.7, 3.0, 0.6], num_samples=5, replacement=False))
print("not_replacement: {}".format(not_replacement))
# not_replacement: [4, 1, 3, 5, 2]


# replacement=True是有放回的抽样
replacement = list(WeightedRandomSampler(weights=[0.1, 0.9, 0.4, 0.7, 3.0, 0.6], num_samples=5, replacement=True))
print("replacement: {}".format(replacement))
# replacement: [5, 1, 5, 1, 1]

batch_sampler_drop_last = list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
print("batch_sampler: {}".format(batch_sampler_drop_last))
# batch_sampler: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

# 丢弃最后一批不足batch_size数量的数据
batch_sampler = list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
print("batch_sampler_drop_last: {}".format(batch_sampler))
# batch_sampler_drop_last: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
