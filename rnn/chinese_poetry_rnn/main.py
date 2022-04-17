# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
import sys
import tqdm
from torch.autograd import Variable

from torch.utils.data import DataLoader
from rnn.chinese_poetry_rnn.config import Config
from rnn.chinese_poetry_rnn.data import Tokenization
from rnn.chinese_poetry_rnn.model import PoetryRNNModel
from rnn.chinese_poetry_rnn.utils import *

conf = Config()
device = get_device()


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(conf, k, v)

    # get data
    tokenizer = Tokenization(conf)
    data = tokenizer.encode()
    word2index = tokenizer.word2index
    index2word = tokenizer.index2word

    data = torch.from_numpy(data)

    data_loader = DataLoader(data, batch_size=conf.batch_size, shuffle=True)

    embedding_dim = 128
    hidden_dim = 256

    # define model
    poetry_model = PoetryRNNModel(len(word2index), embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    optimizer = torch.optim.Adam(poetry_model.parameters(), lr=conf.lr)

    criterion = nn.CrossEntropyLoss()

    if conf.model_path and os.path.isfile(conf.model_path):
        poetry_model.load_state_dict(torch.load(conf.model_path))

    if conf.use_gpu:
        poetry_model.to(device)
        criterion.to(device)

    for epoch in range(conf.epoch):
        cnt = 0
        total_loss = 0
        for ii, data_ in tqdm.tqdm(enumerate(data_loader)):
            data_ = data_.long().transpose(1, 0).contiguous()
            if conf.use_gpu:
                data_ = data_.to(device)
            optimizer.zero_grad()

            input_, target = Variable(data_[:-1, :]), Variable(data_[1:, :])
            output, _ = poetry_model(input_)
            loss = criterion(output, target=target.view(-1))
            total_loss += loss
            loss.backward()
            optimizer.step()
            cnt += 1
        if (1 + ii) % 20 == 0:
            print("loss: {}".format(total_loss / cnt))



def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    给定几个词，根据这几个词接着生成一首完整的诗歌
    start_words：u'春江潮水连海平'
    比如start_words 为 春江潮水连海平，可以生成：
    """

    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    if conf.use_gpu:
        input = input.to(device)
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(conf.max_sequence_length):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results


def gen(**kwargs):
    """
    提供命令行接口，用以生成相应的诗
    """

    for k, v in kwargs.items():
        setattr(conf, k, v)

    tokenizer = Tokenization(conf)
    word2index = tokenizer.word2index
    index2word = tokenizer.index2word

    model = PoetryRNNModel(len(word2index), 128, 256)
    map_location = lambda s, l: s
    state_dict = torch.load(conf.model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    if conf.use_gpu:
        model.cuda()

    # python2和python3 字符串兼容
    if sys.version_info.major == 3:
        if conf.start_words.isprintable():
            start_words = conf.start_words
            prefix_words = conf.prefix_words if conf.prefix_words else None
        else:
            start_words = conf.start_words.encode('ascii', 'surrogateescape').decode('utf8')
            prefix_words = conf.prefix_words.encode('ascii', 'surrogateescape').decode(
                'utf8') if conf.prefix_words else None
    else:
        start_words = conf.start_words.decode('utf8')
        prefix_words = conf.prefix_words.decode('utf8') if conf.prefix_words else None

    start_words = start_words.replace(',', u'，') \
        .replace('.', u'。') \
        .replace('?', u'？')

    gen_poetry = generate
    result = gen_poetry(model, start_words, index2word, word2index, prefix_words)
    print(''.join(result))


train()
