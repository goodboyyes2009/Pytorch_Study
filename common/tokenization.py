# -*- coding: utf-8 -*-
import os
import pickle
from typing import List, Dict
import torch

import jieba
import numpy as np

current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(__file__)
# 重复嵌套获取到工程目录
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Vocabulary(object):

    def __init__(self, token_fn, stop_words, corpus_texts=None):
        """

        :param corpus_texts: 训练词表的预料
        :param stop_words: 停用词列表
        :param token_fn: 分词函数
        """
        self._corpus_texts = corpus_texts
        self._token_fn = token_fn
        self._stop_words = stop_words
        self.max_sentence_length = 0
        self.word2index = {}
        self.index2word = {}
        self.save_path = os.path.join(project_path, 'data/vocab.pkl')

        # 加入词汇表中没有未登录词 Out-of-vocabulary(OOV)的代替符号<unk>和用于padding的符号<pad>
        self.word2index['<pad>'] = 0
        self.word2index['unk'] = 1
        self.index2word[0] = "<pad>"
        self.index2word[1] = "<unk>"

        self._create_vocab()
        self.vocab_size = len(self.word2index)

        if not os.path.exists(self.save_path):
            self._save_vocab()

    def _create_vocab(self):
        # 如果文件存则直接加载词表文件，格式为
        # {"vocab_size": 词表长度, "word2index":{单词到index的映射}, "index2word":{索引到单词的映射}}
        print("建立词表......")
        if os.path.isfile(self.save_path) and os.path.getsize(self.save_path) > 0:
            try:
                fr = open(self.save_path, 'rb')
                unpickler = pickle.Unpickler(fr)
                vocab_info = unpickler.load()
                self.max_sentence_length = vocab_info['max_sentence_length']
                self.word2index = vocab_info['word2index']
                self.index2word = vocab_info['index2word']
                self.vocab_size = len(self.word2index)
                fr.close()
            except Exception as e:
                print("加载词表发生异常, {}".format(e))
        else:
            self._process_data(self._corpus_texts, 2)

    def _save_vocab(self):
        try:
            fw = open(os.path.join(project_path, 'data/vocab.pkl', 'wb'))
            vocab_info = {"vocab_size": self.vocab_size, "word2index": self.word2index, "index2word": self.index2word,
                          "max_sentence_length": self.max_sentence_length}
            pickle.dump(vocab_info, fw)
            fw.close()
        except Exception as e:
            print("保存词表发生异常, {}".format(e))

    def _process_data(self, texts, init_idx):
        idx = init_idx
        for text in texts:
            tokens = [word for word in self._token_fn(text) if word not in self._stop_words]
            # 向word2index中添加单词
            for token in tokens:
                if token not in self.word2index:
                    self.word2index[token] = idx
                    self.index2word[idx] = token
                    idx += 1
            self.max_sentence_length = max(self.max_sentence_length, len(tokens))

    def encode(self, text: str) -> np.array:
        """ 将一个句子编码为onehot向量 """
        input_id = []
        tokens = [word for word in self._token_fn(text) if word not in self._stop_words]
        # padding
        while len(tokens) < self.max_sentence_length:
            tokens.append("<pad>")
        for token in tokens:
            # 处理oov的情况
            token_id = self.word2index[token] if token in self.word2index else self.word2index['<unk>']
            input_id.append(token_id)
        return np.array(input_id)

    def decode(self, input_id: np.array) -> np.array:
        """ 将onehot向量input_id解码为一个分词列表 """
        tokenized_text = []
        for idx in input_id:
            if idx == 0:
                break
            token = self.index2word[idx]
            tokenized_text.append(token)
        return np.array(tokenized_text)

    def encodes(self, texts: np.array) -> np.array:
        """ 将多个句子的编码为多个onehot向量 """
        return np.array([self.encode(text) for text in texts])

    def decodes(self, input_ids: np.array) -> np.array:
        """ 将多个onehot向量编码为多个句子的分词列表 """
        return np.array([self.decode(input_id) for input_id in input_ids])

    # 使用tencent的词向量进行编码
    def encode_by_tencent_word2vec(self, text):
        tencent_wv_embedding = load_tencent_word2vec_from_numpy()

        seg_words = [w for w in self._token_fn(text) if w not in self._stop_words]

        # padding
        seg_words += '<pad>' * (self.max_sentence_length - len(seg_words))
        input_id = []
        embedding_dim = tencent_wv_embedding.shape[1]
        for word in seg_words:
            if word in self.word2index:
                tencent_wv = np.array(tencent_wv_embedding[self.word2index[word]])
                input_id.append(tencent_wv)
            else:
                unknown_wv = np.random.randn(1, embedding_dim) / np.sqrt(embedding_dim)
                input_id.append(unknown_wv)
        return input_id


def tokenize(texts, token_fn):
    """

    :param texts (List[str]): 文本列表
    :param token_fn: 分词函数
    :return:
        tokenized_texts (List[List[str]]): 每一个句子的分词结果列表
        word2index (dict): 单词到索引的映射字典
        index2word (dict): 索引到单词的映射字典
        max_sequence_len (int): 句子的最大长度
    """
    max_sequence_len = 0
    tokenized_texts = []
    word2index = {}
    index2word = {}

    # 加入词汇表中没有未登录词 Out-of-vocabulary(OOV)的代替符号<unk>和用于padding的符号<pad>
    word2index['<pad>'] = 0
    word2index['unk'] = 1
    index2word[0] = "<pad>"
    index2word[1] = "<unk>"

    idx = 2
    # 由于上述两个特殊符合占据了两个index, 因此从下标2开始从预料库中构建Vocabulary
    for sentence in texts:
        tokenized_sentence = token_fn(sentence)

        tokenized_texts.append(tokenized_sentence)

        # 向word2index中添加单词
        for token in tokenized_sentence:
            if token not in word2index:
                word2index[token] = idx
                index2word[idx] = token
                idx += 1
        max_sequence_len = max(max_sequence_len, len(tokenized_sentence))
    return tokenized_texts, word2index, index2word, max_sequence_len


def encode(tokenized_texts: List[List[str]], word2index: Dict, max_sequence_len: int) -> np.array:
    """
    将每个句子padding到max_sequence_len的长度，将每一个句子中单词使用word2index映射关系转换成单词在Vocabulary的索引下标
    :param tokenized_texts: (List[List[str]]): 每一个句子的分词结果列表
    :param word2index: (dict): 单词到索引的映射字典
    :param max_sequence_len (int): 句子的最大长度
    :return:
        input_ids (np.array): 形状为[N(句子个数), max_sequence_len]
    """
    input_ids = []
    for tokenized_text in tokenized_texts:
        input_id = []
        while len(tokenized_text) < max_sequence_len:
            tokenized_text.append("<pad>")
        for token in tokenized_text:
            if token in word2index:
                input_id.append(word2index[token])
            else:
                # 处理oov的情况
                input_id.append(word2index['<unk>'])
        input_ids.append(input_id)
    return np.array(input_ids)


def decode_single(input_id: List[int], index2word: Dict) -> np.array:
    """
    将一个input_id转换成相应的句子的分词列表
    :param input_id (List): 句子中的单词在词汇表Vocabulary的索引列表
    :param index2word (Dict): 索引到单词的映射字典
    :return:
        tokenized_text: 对应的分词列表
    """
    tokenized_text = []
    for idx in input_id:
        if idx == 0:
            break
        token = index2word[idx]
        tokenized_text.append(token)
    return np.array(tokenized_text)


def decode(input_ids: List[List[int]], index2word: Dict) -> np.array:
    """
    将多个input_id转换成相应的句子分词列表
    :param input_ids:
    :param index2word:
    :return:
    """
    return np.array([decode_single(input_id, index2word) for input_id in input_ids])


def token_function(text):
    if not text and len(text.strip()) == 0:
        return []
    try:
        tokens = jieba.cut(text, cut_all=False)
        return tokens
    except Exception as e:
        print("处理句子[{}]分词失败, 异常信息{}".format(text, e))
        return []


def get_stop_words():
    try:
        stop_words = [word.strip() for word in
                      open(os.path.join(project_path, 'data/stop_words'), 'r', encoding='utf-8')]
        return stop_words
    except Exception as e:
        print("获取停用词失败,{}".format(e))
        return []


def load_pretrained_tencent_word_embedding(word2vec_path, embedding_dim=100):
    from gensim.models import KeyedVectors
    from collections import OrderedDict

    print("start load tencent word2vec ....")
    tencent_wv_text = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    print("load tencent word2vec end ....")

    # numpy保存顺序字典

    vocab = Vocabulary(token_fn=token_function, stop_words=get_stop_words())
    vocab_size = vocab.vocab_size
    sorted_word2index = sorted(vocab.word2index.items(), key=lambda kv: (kv[1], kv[0]))

    order_index2word = OrderedDict()
    ovv_cnt = 0
    for w, index in sorted_word2index:
        if w in tencent_wv_text:
            tencent_wv_embedding = tencent_wv_text[w]
            order_index2word[index] = tencent_wv_embedding
        else:
            ovv_cnt += 1
            # Out of Vocabulary(OOV)的情况， 给一个随机的embedding

            # 使用torch的做法
            # oov_embedding = np.ones(1, embedding_dim)
            # torch.nn.init.xavier_uniform_(oov_embedding)

            # 使用numpy
            oov_embedding = np.random.randn(1, embedding_dim) / np.sqrt(embedding_dim)
            print("{%d: %s}" % (index, w))
            order_index2word[index] = oov_embedding
    print("{}/{}".format(ovv_cnt, vocab_size))
    np.save('news_tencent_wv_embedding_d100-v0.2.0.npy', order_index2word)


def load_tencent_word2vec_from_numpy():
    npy_path = os.path.join(project_path, 'common/news_tencent_wv_embedding_d100-v0.2.0.npy')
    target_index2word_dict = np.load(npy_path, allow_pickle=True)
    # target_embedding = list(map(lambda x: np.shape(x), [target_index2word_dict.take(0)[i] for i in
    #                                                     range(len(target_index2word_dict.take(0)))]))
    target_embedding = [
        np.squeeze(target_index2word_dict.take(0)[i], axis=0) if np.ndim(target_index2word_dict.take(0)[i]) > 1 else
        target_index2word_dict.take(0)[i] for i in range(len(target_index2word_dict.take(0)))]
    target_embedding = torch.from_numpy(np.array(target_embedding, dtype=np.float))
    return target_embedding


if __name__ == "__main__":
    # texts = ["i love you", "i like cat", "i hate dog", "you just a boy"]
    # tokenized_texts, word2index, index2word, max_sequence_length = tokenize(texts, lambda x: x.split())
    # assert tokenized_texts == [['i', 'love', 'you'], ['i', 'like', 'cat'], ['i', 'hate', 'dog'],
    #                            ['you', 'just', 'a', 'boy']]
    # assert index2word[word2index["i"]] == "i"
    # encode_texts = encode(tokenized_texts, word2index, max_sequence_length)
    # print("encode_texts: {}".format(encode_texts))
    # decode_result = decode_single([1, 2, 3, 2, 0], index2word)
    # print(decode_result)
    #
    # data_root_path = '/home/hj/dataset/news_data/news_zh'
    #
    # data_file_list = [
    #     os.path.join(data_root_path, 'dev.tsv'),
    #     os.path.join(data_root_path, 'train.tsv'),
    #     os.path.join(data_root_path, 'test.tsv'),
    # ]
    #
    # texts = [line.split('\t')[0].strip() for f in data_file_list for line in open(f, 'r', encoding='utf-8')]
    #
    # stop_words = get_stop_words()
    #
    # vocabulary = Vocabulary(corpus_texts=texts, stop_words=stop_words, token_fn=token_function)
    # print("词表长度:{}".format(vocabulary.vocab_size))
    # vocabulary = Vocabulary(stop_words=stop_words, token_fn=token_function)
    # print("max_sentence_length:{}".format(vocabulary.max_sentence_length))

    # word2vec_path = "/home/hj/data/pretrain-word2vec/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt"
    # word2vec_path = "/home/hj/dataset/word2vec/tencent-ailab-embedding-zh-d100-v0.2.0/tencent-ailab-embedding-zh-d100-v0.2.0.txt"
    # load_pretrained_tencent_word_embedding(word2vec_path)

    npy_path = 'news_tencent_wv_embedding_d100-v0.2.0.npy'
    index2word_embedding = load_tencent_word2vec_from_numpy()
    print(index2word_embedding)