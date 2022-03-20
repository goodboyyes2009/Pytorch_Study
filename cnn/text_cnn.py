# -*- coding: utf-8 -*-
from typing import Any

import torch
import torch.nn as nn


class TextCNN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        self.filter_sizes = [3, 4, 5]
        self.filter_nums = [100, 100, 100]
        self.embedding_dim = 63
        self.conv_list = nn.ModuleList(
            [nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.filter_nums[i],
                       kernel_size=self.filter_sizes[i]) for i in
             range(len(self.filter_nums))])
        self.fc = nn.Linear(in_features=sum(self.filter_nums), out_features=num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input_ids):
        """

        :param input_ids: shpae=[batch, sequence_len]
        :return:
        """



        pass
