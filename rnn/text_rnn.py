# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TextRNN(nn.Module):

    def __init__(self):
        super(TextRNN, self).__init__()

        self.rnn = nn.RNN(input_size=n_classes, hidden_size=n_hidden)

        self.W = nn.Linear(in_features=n_hidden, out_features=n_classes, bias=False)

        self.b = nn.Parameter(torch.ones([n_classes]))

    def forward(self, hidden, X):
        X = X.transpose(0, 1)  # X :[n_step, batch_size, n_classes]
        outputs, hidden = self.rnn(X, hidden)
        # output: [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden: [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1]  # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b  # model: [batch_size, n_classes]
        return model


def make_bacth():
    input_batch = []
    target_batch = []

    for sentence in sentences:
        word = sentence.split()
        input = [word_dict[n] for n in word[:-1]]  # create (1, n-1) as input
        target = word_dict[word[-1]]  # create (n) as target, We usually call this 'casual language model'

        # 构造输入的one-hot
        input_batch.append(np.eye(n_classes)[input])
        target_batch.append(target)

    return input_batch, target_batch


def training_loop(n_epochs, optimizer, model, loss_fn, input_batch, target_batch):
    for epoch in range(1, n_epochs + 1):

        hidden = torch.zeros(1, batch_size, n_hidden)
        output = model(hidden, input_batch)
        # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)

        loss = loss_fn(output, target_batch)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    n_step = 2  # number of cells(= number of step)
    n_hidden = 5  # number of hidden units in one cell

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}

    n_classes = len(word_list)
    batch_size = len(sentences)

    model = TextRNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 5000

    input_batch, target_batch = make_bacth()

    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)


    print("model:{}".format(model))

    for name, param in model.named_parameters():
        print("name:{}, param:{}".format(name, param.shape))


    # training_loop(n_epochs, optimizer, model, loss_fn, input_batch, target_batch)
