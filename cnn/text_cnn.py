# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
from common.data_hepler import *
from common.tokenization import *
import numpy as np


class TextCNN(nn.Module):

    def __init__(self, vocab_size=None, embedding_dim=None, num_filters=[100, 100, 100], filter_sizes=[3, 4, 5],
                 num_classes=2, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.filter_sizes = filter_sizes
        self.filter_nums = num_filters
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.conv_list = nn.ModuleList(
            [nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.filter_nums[i],
                       kernel_size=self.filter_sizes[i], bias=True) for i in range(len(self.filter_nums))])
        self.fc = nn.Linear(in_features=sum(self.filter_nums), out_features=num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        """
        :param input_ids: shape=[batch_size(句子数量), sequence_len(句子长度)]
        :return:
        """
        # embedding_out shape=[batch_size, sequence_len, embedding_dim]
        embedding_out = self.embedding(input_ids)
        # 由于Conv1d的输入shape为[N(句子数量), C_in(Conv1d的in_channels参数), L_in(句子长度)]即[batch_size, embedding_dim, sequence_len]
        # embedding_out shape=[batch_size, embedding_dim, sequence_len]
        embedding_out = embedding_out.permute(0, 2, 1)
        conv1_result_list = [conv1(embedding_out) for conv1 in self.conv_list]
        # cov1d out shape: [batch_size, filter_nums[i], features]
        nonlinear_result_list = [F.relu(conv_result, inplace=True) for conv_result in conv1_result_list]
        # max_pool out shape :[batch_size, filter_nums[i], 1]
        max_pooled_list = [F.max_pool1d(nonlinear_result, kernel_size=nonlinear_result.shape[2]) for nonlinear_result in
                           nonlinear_result_list]
        # Concatenate x_pool_list to feed the fully connected layer.
        # out shape: [batch_size, sums[filter_nums[i]]]
        x_fc = torch.cat([pool_result.squeeze(dim=2) for pool_result in max_pooled_list], dim=1)

        # Compute logits. Output shape: (b, n_classes)
        # shape: [batch_size, num_classes]
        logits = self.fc(self.dropout(x_fc))
        return logits


# optimzer
import torch.optim as optim


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("There are {} GPU(s) available".format(torch.cuda.device_count()))
        print("Device name:{}".format(torch.cuda.get_device_name()))
    else:
        print("No GPU available, use CPU instead.")
        device = torch.device("cpu")
    return device


def initilize_model(vocab_size=None, embedding_dim=100, num_filters=[100, 100, 100], filter_sizes=[2, 3, 3],
                    num_classes=2, dropout=0.5, learning_rate=0.001):
    # 初始化模型
    text_cnn_model = TextCNN(vocab_size=vocab_size,
                             embedding_dim=embedding_dim,
                             num_filters=num_filters,
                             filter_sizes=filter_sizes,
                             num_classes=num_classes,
                             dropout=dropout)

    # send model to gpu
    device = get_device()
    text_cnn_model.to(device)

    # 初始化优化器
    optimizer = optim.Adadelta(text_cnn_model.parameters(), lr=learning_rate, rho=0.95)

    # 初始化loss函数
    loss_fn = nn.CrossEntropyLoss()

    return text_cnn_model, optimizer, loss_fn


def train_loop(model, optimizer, loss_fn, batch_size=2, num_epochs=100, x_train=None,
               y_train=None):
    batches = batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)
    # Training loop. For each batch...

    model.train()
    device = get_device()
    for batch in batches:
        # 变成tensor
        if len(batch) > 0:
            input_ids_batch, label_batch = tuple(torch.tensor(data) for data in zip(*batch))

            # load batch to GPU
            input_ids_batch, label_batch = tuple(t.to(device) for t in (input_ids_batch, label_batch))

            # 计算loss
            logist = model(input_ids_batch)
            optimizer.zero_grad()
            loss = loss_fn(logist, label_batch)
            loss.backward()
            optimizer.step()
            print("train loss:{}".format(loss))


def evaluate():
    pass


# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

tokenized_texts, word2index, index2word, max_sequence_len = tokenize(sentences, lambda x: x.split())

text_cnn_model, optimizer, loss_fn = initilize_model(vocab_size=len(word2index))

# train_loop(text_cnn_model, optimizer, loss_fn, x_train=encode(tokenized_texts, word2index, max_sequence_len),
#            y_train=labels)


inputs = encode(tokenized_texts, word2index, max_sequence_len)
device = get_device()

# 将array变成tensor
labels = torch.tensor(labels, device=device, dtype=torch.long)
inputs = torch.tensor(inputs, device=device, dtype=torch.long)

for epoch in range(5001):

    # 前向传播进行模型预测
    predict = text_cnn_model(inputs)

    # 在反向传播前将梯度清零
    optimizer.zero_grad()

    # 计算loss
    loss = loss_fn(predict, labels)

    # 通过loss进行反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    if epoch % 100 == 0:
        print("epoch {}, train loss: {}".format(epoch, loss))

# 模型预测
test_text = "sorry hate you"


def encode_text(text: str):
    encode_text_input_ids = []
    for x in text.strip().split():
        input_id = word2index[x] if x in word2index else word2index['<unk>']
        encode_text_input_ids.append(input_id)
    return encode_text_input_ids

encode_test_text = encode_text(test_text)
encode_test_text += [0] * (max_sequence_len - len(encode_test_text))

# 转换成tensor
encode_test_text = torch.tensor(encode_test_text, device=device, dtype=torch.long)

# predict
pred =  text_cnn_model(encode_test_text.unsqueeze_(dim=0)).data.max(1, keepdim=True)[1]

print("pred: {}".format(pred))
if pred[0][0] == 0:
    print("is Bad mean...")
else:
    print("is God mean...")