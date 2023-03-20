# -*- coding: utf-8 -*-
import os
import pickle

import torch
from tqdm import tqdm


class Dictionary(object):

    def __init__(self):
        self.word2id = {}
        self.id2word = []

    def __len__(self):
        return len(self.id2word)

    def add_word(self, word):
        if word not in set(self.id2word):
            self.id2word.append(word)
            self.word2id[word] = len(self.id2word) - 1
        return self.word2id[word]


class Corpus(object):
    """构建语料"""

    def __init__(self, base_dir):

        self._dictionary_path = os.path.join(base_dir, 'dict.pickle')

        self._train_path = os.path.join(base_dir, 'train_data.pt')
        self._valid_path = os.path.join(base_dir, 'valid_data.pt')
        self._test_path = os.path.join(base_dir, 'test_data.pt')

        # 如果词典文件dict.pickle存在，则直接加载，如果不存在在重新创建
        if os.path.exists(self._dictionary_path):
            with open(self._dictionary_path, 'rb') as f:
                self.dictionary = pickle.load(f)
        else:
            self.dictionary = Dictionary()

        # 标记句子结束的符号
        self.end_of_sentence_char = "<eos>"
        # 标记词典中没有出现的词
        self.unknown_char = "<unk>"

        if os.path.exists(self._train_path):
            self.train_data = torch.load(self._train_path)
        else:
            self.train_data = self.tokenizer(os.path.join(base_dir, 'wiki.train.tokens'))
            torch.save(self.train_data, self._train_path)

        print(f"dict len: {len(self.dictionary)}")

        if os.path.exists(self._valid_path):
            self.valid_data = torch.load(self._valid_path)
        else:
            self.valid_data = self.tokenizer(os.path.join(base_dir, 'wiki.valid.tokens'))
            torch.save(self.valid_data, self._valid_path)

        print(f"dict len: {len(self.dictionary)}")

        if os.path.exists(self._test_path):
            self.test_data = torch.load(self._test_path)
        else:
            self.test_data = self.tokenizer(os.path.join(base_dir, 'wiki.test.tokens'))
            torch.save(self.test_data, self._test_path)

        if not os.path.exists(self._dictionary_path):
            with open(self._dictionary_path, 'wb') as f:
                print(f"dict len: {len(self.dictionary)}")
                pickle.dump(self.dictionary, f)

    def tokenizer(self, filename):
        ff = open(filename, 'r')
        total = len(ff.readlines())
        ff.close()

        self._build_dictionary(filename, total)
        with open(filename, 'r', encoding='utf-8') as f:
            sentence_token_ids_list = []
            with tqdm(f, desc='read' + filename) as pbar:
                for line in f:
                    words = line.split() + [self.end_of_sentence_char]
                    sentence_token_ids = list(map(lambda x: self.dictionary.word2id[x], words))
                    sentence_token_ids_list.append(torch.tensor(sentence_token_ids, dtype=torch.int64))
                    pbar.update(1)
            return torch.cat(sentence_token_ids_list)

    def _build_dictionary(self, filepath, total):
        with open(filepath, 'r', encoding='utf-8') as f:
            with tqdm(total=total, desc="构建dict" + filepath) as pbar:
                for line in f:
                    # 加上句子结束符<eos>
                    words = line.split() + [self.end_of_sentence_char]
                    for word in words:
                        self.dictionary.add_word(word)
                    pbar.update(1)


if __name__ == "__main__":
    base_dir = '../../data/wikitext-2'
    corpus = Corpus(base_dir=base_dir)
    print(f"dict len: {len(corpus.dictionary)}")
    print(f"train len: {len(corpus.train_data)}")
    print(f"test len: {len(corpus.test_data)}")
    print(f"valid len: {len(corpus.valid_data)}")
    train_data = corpus.train_data.data
    print(train_data[:3])
    with open('../../data/wikitext-2/dict.pickle', 'rb') as f:
        dd = pickle.load(f)
        print(len(dd))
