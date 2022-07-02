# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TextRNN(nn.Module):

    def __init__(self):
        super(TextRNN, self).__init__()

        # input_size: input_x的特征个数, 可以理解为embedding_dim
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=n_hidden, num_layers=1)

        self.W = nn.Linear(in_features=n_hidden, out_features=vocab_size, bias=False)

        self.b = nn.Parameter(torch.ones([vocab_size]))

    def forward(self, input_x, hidden):
        input_x = input_x.transpose(0, 1)  # X :[sequence_len, batch_size, input_size]

        outputs, hidden = self.rnn(input_x, hidden)
        # output: [sequence_len, batch_size, num_directions(=1) * n_hidden]
        # hidden: [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1]  # [sequence_len, batch_size, num_directions(=1) * n_hidden]

        model = self.W(outputs) + self.b  # model: [batch_size, n_classes]
        return model


def make_batch():
    input_batch = []
    target_batch = []

    for sentence in sentences:
        word = sentence.split()
        input = [word2index_dict[n] for n in word[:-1]]  # create (1, n-1) as input
        target = word2index_dict[word[-1]]  # create (n) as target, We usually call this 'casual language model'
        # 构造输入的one-hot, input_batch: [word_num ,vocab_size]

        input_batch.append(np.eye(vocab_size)[input])
        target_batch.append(target)

    return input_batch, target_batch


def training_loop(n_epochs, optimizer, model, loss_fn, input_batch, target_batch):
    for epoch in range(1, n_epochs + 1):

        h_0 = torch.zeros(1, batch_size, n_hidden)
        # h_0 : [num_layers * num_directions, batch, hidden_size],
        # tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided.
        # If the RNN is bidirectional, num_directions should be 2, else it should be 1.

        # input_batch : [batch_size, n_step, n_class]
        output = model(input_batch, h_0)
        # output : [batch_size, n_class],

        # target_batch : [batch_size] (LongTensor, not one-hot)
        loss = loss_fn(output, target_batch)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    # n_step = 2  # number of cells(= number of step)
    n_hidden = 5  # number of hidden units in one cell

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    # 使用set进行去重
    word_list = list(set(word_list))
    word2index_dict = {w: i for i, w in enumerate(word_list)}
    index2word_dict = {i: w for i, w in enumerate(word_list)}

    vocab_size = len(word_list)

    batch_size = len(sentences)

    model = TextRNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 5000

    input_batch, target_batch = make_batch()

    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    print("model:{}".format(model))

    for name, param in model.named_parameters():
        print("name:{}, param:{}".format(name, param.shape))

    training_loop(n_epochs, optimizer, model, loss_fn, input_batch, target_batch)
