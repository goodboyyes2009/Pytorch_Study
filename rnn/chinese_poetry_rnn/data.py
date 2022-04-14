# -*- coding: utf-8 -*-
import json
from json import JSONEncoder
import os
from common.file_util import *


class PoetryInfo(object):
    def __init__(self, title, author, paragraphs, *args, **kwargs):
        self.title = title
        self.author = author
        self.paragraphs = paragraphs


class PoetryEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class Tokenization(object):

    def __init__(self):

        # 判断poetry_vocab.pkl文件是否存在
        vocab_path = 'poetry_vocab.pkl'
        self.padding_char = "</s>"
        self.start_of_poetry_char = "<START>"
        self.end_of_poetry_char = "<END>"
        if os.path.exists(vocab_path):
            pickle_obj = load_pickle(vocab_path)
            self.vocab = pickle_obj['vocab']
            self.word2index = pickle_obj['word2index']
            self.index2word = pickle_obj['index2word']

        else:
            self.vocab = self._load_data()
            self.word2index = {}
            self.index2word = {}

            # init word2index
            self.word2index = {w: i + 3 for i, w in enumerate(self.vocab)}
            self.word2index[self.padding_char] = 0
            self.word2index[self.start_of_poetry_char] = 1
            self.word2index[self.end_of_poetry_char] = 2

            # init index2word
            self.index2word = {i + 3: w for i, w in enumerate(self.vocab)}
            self.index2word[0] = self.padding_char
            self.index2word[1] = self.start_of_poetry_char
            self.index2word[2] = self.end_of_poetry_char

            self.vocab.append(self.padding_char)
            self.vocab.append(self.start_of_poetry_char)
            self.vocab.append(self.end_of_poetry_char)

    def _load_data(self):
        data_dir = "../../data/quan_tang_shi/json"
        char_list = []
        for file_name in os.listdir(data_dir):
            f = None
            try:
                f = open(os.path.join(data_dir, file_name), 'r')
                tangshi_list = json.load(f)
                for tangshi in tangshi_list:
                    poetry_list = tangshi['paragraphs']
                    poetry_list = list(map(lambda text: [token for token in text.replace("——(.?*)", '')], poetry_list))
                    for poetry in poetry_list:
                        char_list.extend(poetry)
            except Exception as e:
                print("解析{}失败,{}".format(file_name, e))
            finally:
                f.close()
        # 由于set是无序的，加一个list确保word有序
        return list(set(char_list))

    def encode(self, text, max_poetry_length=125):
        text = text[:max_poetry_length]
        text = [self.word2index[w] if w in self.word2index else self.word2index[self.padding_char] for w in text]


    def decode(self, input_id):
        pass


if __name__ == "__main__":
    s = """{
        "title": "兩儀殿賦柏梁體",
        "author": "李世民",
        "biography": "",
        "paragraphs": [
            "絕域降附天下平，——李世民",
            "八表無事悅聖情。——淮安王",
            "雲披霧斂天地明，——長孫無忌",
            "登封日觀禪雲亭，——房玄齡",
            "太常具禮方告成。——蕭瑀"
        ],
        "notes": [
            ""
        ],
        "volume": "卷一",
        "no#": 87
    }
    """
    # https://pynative.com/python-convert-json-data-into-custom-python-object/
    # student = Student(1, "Emma")
    #
    # # encode Object it
    # studentJson = json.dumps(student, cls=StudentEncoder, indent=4)
    p = json.loads(s)
    pp = PoetryInfo(**p)
    print(pp.paragraphs)

    s = list(map(lambda text: text.replace(r".*?(--*.?)", ''), pp.paragraphs))
    print(s)
    tt = Tokenization()
