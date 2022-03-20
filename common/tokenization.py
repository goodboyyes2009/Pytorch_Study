# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Dict


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


def load_pretrained_tencent_word_embedding():
    pass




if __name__ == "__main__":
    texts = ["i love you", "i like cat", "i hate dog", "you just a boy"]
    tokenized_texts, word2index, index2word, max_sequence_length = tokenize(texts, lambda x: x.split())
    assert tokenized_texts == [['i', 'love', 'you'], ['i', 'like', 'cat'], ['i', 'hate', 'dog'],
                               ['you', 'just', 'a', 'boy']]
    assert index2word[word2index["i"]] == "i"
    encode_texts = encode(tokenized_texts, word2index, max_sequence_length)
    print("encode_texts: {}".format(encode_texts))
    decode_result = decode_single([1, 2, 3, 2, 0], index2word)
    print(decode_result)
