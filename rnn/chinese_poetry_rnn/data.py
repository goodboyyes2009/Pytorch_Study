# -*- coding: utf-8 -*-
import json
import re
import os
import numpy as np


class Tokenization(object):

    def __init__(self, conf):
        # 判断poetry_vocab.pkl文件是否存在
        self.conf = conf
        vocab_path = conf.pickle_path
        self.padding_char = "</s>"
        self.start_of_poetry_char = "<START>"
        self.end_of_poetry_char = "<END>"
        self.data = None
        self.word2index = {}
        self.index2word = {}

        if os.path.exists(vocab_path):
            data = np.load(vocab_path, allow_pickle=True)
            data, word2index, index2word = data['data'], data['word2index'].item(), data['index2word'].item()
            self.data = data
            self.index2word = index2word
            self.word2index = word2index
            return

        self.data = parseRawData(conf.author, conf.constrain, conf.data_path)

        # init word2index
        self.vocab = {_word for _sentence in self.data for _word in _sentence}
        self.word2index = {w: i + 3 for i, w in enumerate(self.vocab)}
        self.word2index[self.padding_char] = 0
        self.word2index[self.start_of_poetry_char] = 1
        self.word2index[self.end_of_poetry_char] = 2

        # init index2word
        self.index2word = {i + 3: w for i, w in enumerate(self.vocab)}
        self.index2word[0] = self.padding_char
        self.index2word[1] = self.start_of_poetry_char
        self.index2word[2] = self.end_of_poetry_char

    def encode(self):
        # 加上起始符和终止符号
        if not os.path.exists(self.conf.pickle_path):
            for i in range(len(self.data)):
                self.data[i] = [self.start_of_poetry_char] + list(self.data[i]) + [self.end_of_poetry_char]
            self.data = [[self.word2index[_word] for _word in _sentence] for _sentence in self.data]

            # padding
            self.data = pad_sequences(self.data, maxlen=self.conf.max_sequence_length, truncating='post',
                                      value=self.word2index[self.padding_char])
        return self.data

    def save(self):
        # save
        if not os.path.exists(self.conf.pickle_path):
            np.savez_compressed(self.conf.pickle_path, data=self.data, word2index=self.word2index,
                                index2word=self.index2word)
        else:
            print("文件{}已存在!".format(self.conf.pickle_path))


def parseRawData(author=None, constrain=None, src='../../data/quan_tang_shi/json/'):
    def sentenceParse(para):
        # para = "-181-村橋路不端，數里就迴湍。積壤連涇脉，高林上笋竿。早嘗甘蔗淡，生摘琵琶酸。（「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）好是去塵俗，煙花長一欄。"
        result, number = re.subn("（.*）", "", para)
        result, number = re.subn("{.*}", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("[\]\[]", "", result)
        r = ""
        for s in result:
            if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
                r += s
        r, number = re.subn("。。", "。", r)
        return r

    def handleJson(file):
        # print file
        rst = []
        data = json.loads(open(file).read())
        for poetry in data:
            pdata = ""
            if (author != None and poetry.get("author") != author):
                continue
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split("[，！。]", s)
                for tr in sp:
                    if constrain != None and len(tr) != constrain and len(tr) != 0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue
            for sentence in poetry.get("paragraphs"):
                pdata += sentence
            pdata = sentenceParse(pdata)
            if pdata != "":
                rst.append(pdata)
        return rst

    # print sentenceParse("")
    data = []
    # src = '../../data/quan_tang_shi/json/'
    for filename in os.listdir(src):
        # if filename.startswith("poet.tang"):
        if filename.endswith(".json"):
            data.extend(handleJson(src + filename))
    return data


def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
    """
    code from keras
    Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)
    Raises:
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


if __name__ == "__main__":
    from rnn.chinese_poetry_rnn.config import Config

    conf = Config()
    tokenizer = Tokenization(conf)
    print(tokenizer.data[43102])
    data = tokenizer.encode()
    print(np.shape(data))
    print(data[43102])
    tokenizer.save()
