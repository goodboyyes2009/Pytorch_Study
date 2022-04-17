# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.autograd import Variable


class PoetryRNNModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = 2
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_lstm_layers,
                            batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=vocab_size)

    def forward(self, input, hidden=None):
        sequence_length, batch_size = input.size()
        if hidden is None:
            h_0 = input.data.new(self.num_lstm_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(self.num_lstm_layers, batch_size, self.hidden_dim).fill_(0).float()
            h_0, c_0 = Variable(h_0), Variable(c_0)
        else:
            h_0, c_0 = hidden

        # size: sequence_length, batch_size, embedding_dim
        embeds = self.embedding(input)
        # output size: (sequence_length, batch_size,hidde_dim
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # size: (sequence_length, vocab_size)
        output = self.linear(output.reshape(sequence_length * batch_size, -1))
        return output, hidden