# -*- coding: utf-8 -*-
import torch

import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    # num_heads: 多头注意力的个数
    # hid_dim: 每个词输出的向量维度

    def __init__(self, hid_dim, nums_heads, dropout):
        super(MultiHeadedAttention, self).__init__()
        self._hid_dim = hid_dim
        self._num_heads = nums_heads

        # 强制hid_dim必须整除h
        assert hid_dim % nums_heads == 0

        # 定义W_q矩阵
        self.w_q = nn.Linear(self._hid_dim, self._hid_dim)
        # 定义W_k矩阵
        self.w_k = nn.Linear(self._hid_dim, self._hid_dim)
        # 定义W_v矩阵
        self.w_v = nn.Linear(self._hid_dim, self._hid_dim)

        self.fc = nn.Linear(self._hid_dim, self._hid_dim)

        self.dropout = nn.Dropout(p=dropout)

        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([self._hid_dim // self._num_heads]))

    def forward(self, query, key, value, mask=None):
        # Q, K, V在句子长度的这一个维度的数值可以一样， 也可以不一样。
        # K: [64,10,300], 假设batch_size为64,有10个词, 每个词的Query向量是 300 维.
        # V: [64,10,300], 假设batch_size为64，有 10 个词，每个词的 Query 向量是  300 维.
        # Q: [64,12,300], 假设batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维.
        batch_size = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 这里把K,Q,V矩阵拆分成多组注意力

        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算

        Q = Q.view(batch_size, -1, self._num_heads, self._hid_dim // self._num_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self._num_heads, self._hid_dim // self._num_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self._num_heads, self._hid_dim // self._num_heads).permute(0, 2, 1, 3)

        # 第一步： Q 乘以 K的转置, 除以scale
        # [64, 6, 12, 50] * [64, 6, 10, 50] = [64, 6, 12, 10]

        # attention shape: [64, 6, 12, 10]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 如果mask不为空，把mask为0的位置的 attention 分数设置为 -1e10，这里用"0"来指示哪些位置的词向量不能被attention到，比如padding位置，
        # 当然也可以用"1"或者其他数字来指示，主要设计下面2行代码的改动。
        if mask is not None:
            attention = attention.masked_fill(mask==0, -1e10)

        # 第二步： 计算上一步结果的 softmax, 再经过dropout, 得到attention
        # 注意这里是对最后一维做softmax, 也就是输入序列的维度做softmax
        # attention: [64, 6, 12, 10]
        attention = self.dropout(torch.softmax(attention, dim=-1))

        # 第三步, attention结果与V相乘, 得到多头注意力的结果
        # [64,6,12,10] * [64, 6,10,50] = [64,6,12,50]

        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 50 和 6 放到后面，方便下面拼接多组的结果
        # x: [64, 6, 12, 50] 转置-> [64, 12, 6, 50]

        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(batch_size, -1, self._num_heads * (self._hid_dim // self._num_heads))
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
    query = torch.rand(64, 12, 300)
    # batch_size 为 64，有 12 个词，每个词的 Key 向量是 300 维
    key = torch.rand(64, 10, 300)
    # batch_size 为 64，有 10 个词，每个词的 Value 向量是 300 维
    value = torch.rand(64, 10, 300)
    attention = MultiHeadedAttention(hid_dim=300, nums_heads=6, dropout=0.1)
    output = attention(query, key, value)
    ## output: torch.Size([64, 12, 300])
    print(output.shape)
