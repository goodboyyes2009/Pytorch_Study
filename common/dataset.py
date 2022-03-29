# -*- coding: utf-8 -*-
import os

import numpy as np
import torch.utils


class ChineseNewsData(torch.utils.data.Dataset):
    """Chinese News dataset."""

    train_file_name = "train.tsv"
    test_file_name = "test.tsv"

    def __init__(self, has_header=False, split_char=",", data_root_path=None, train=True, transforms=None):
        super().__init__()
        self.has_header = has_header
        self.data_root_path = data_root_path
        self.train = train
        self.transforms = transforms
        self.split_char = split_char

        self.data = []
        self.targets = []

        file_name = self.train_file_name if train else self.test_file_name

        with open(os.path.join(self.data_root_path, file_name), 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    if self.has_header:
                        continue
                    splits = line.split(self.split_char)
                    if len(splits) < 2:
                        continue
                    if len(splits[0].strip()) == 0:
                        continue
                    self.data.append(splits[0].strip())
                    target = splits[1].strip().replace('\n', '')
                    self.targets.append(int(target))
                except Exception as e:
                    print("解析{}数据发生异常，异常信息为:{}".format(file_name, e))

    def __len__(self):
        """ 返回整个数据的长度 """
        return len(self.data)

    def __getitem__(self, index):
        """
        :param index: 数据在list中的下标
        :return: a tuple, (text, label)
        """
        text, label = self.data[index], self.targets[index]
        if self.transforms is not None:
            text = self.transforms(text)
        if self.transforms is not None:
            label = self.transforms(label)
        return text, label


# 定义一些transforms
# tokenize
class Tokenize(object):
    """
    对文本进行编码
    """

    def __init__(self, encode_fn):
        self.encode_fn = encode_fn


    def __call__(self, sample):
        # label 为int类型, 因此下面的条件不成立
        if isinstance(sample, str):
            # 只对text进行onehot编码变成input_id
            sample = self.encode_fn(sample)
        return sample


class ToTensor(object):
    """
    将input_id(List[int])或者label(int)转换成tensor
    """

    def __init__(self, device):
        # print("ToTensor device:{}".format(device))
        self._device = device

    def __call__(self, sample):
        if isinstance(sample, np.ndarray):
            # 将input_id转换成LongTensor
            sample = torch.from_numpy(sample).long()
            sample = torch.tensor(sample, device=self._device)
        elif isinstance(sample, int):
            # 将label转换成LongTensor
            sample = torch.tensor(sample, device=self._device).long()
        return sample
