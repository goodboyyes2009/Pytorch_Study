# -*- coding: utf-8 -*-
from typing import List
import numpy as np


# Generate batches
def batch_iter(data: List, batch_size: int, num_epochs: int, shuffle=True):
    """
    Generate a batch iterable for a dataset
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    batch_num_for_each_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(data_size)
            data = data[shuffle_indices]
        for batch_num in range(batch_num_for_each_epoch):
            start_index = epoch * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]
