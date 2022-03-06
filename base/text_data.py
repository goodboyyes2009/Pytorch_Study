# -*- coding: utf-8 -*-
import torch

# 加载《傲慢与偏见》英文小说， 下载地址: http://www.gutenberg.org/files/1342/1342-0.txt
with open("../data/representation/1342-0.txt", 'r', encoding='utf-8') as f:
    text = f.read()

    lines = text.split("\n")
    line = lines[200]
    print("lines[200]: {}".format(line))

    letter_tensor = torch.zeros(len(line), 128)  # 128是由于ASCII的限制
    print("letter_tensor shape:{}".format(letter_tensor.shape))

    for i, letter in enumerate(line.lower().strip()):
        # 文本里含有双引号，不是有效的ASCII，因此在此处将其屏蔽
        # ord()-->返回值是对应的十进制整数
        letter_index = ord(letter) if ord(letter) < 128 else 0
        letter_tensor[i][letter_index] = 1


    def clean_words(input_str):
        punctuation = '.,;:"!?”“_-'
        word_list = input_str.lower().replace('\n', ' ').split()
        word_list = [word.strip(punctuation) for word in word_list]
        return word_list


    word_list = sorted(set(clean_words(text)))
    word2index_dict = {word: i for (i, word) in enumerate(word_list)}
    print("len(word2index_dict):{}\nword2index_dict['impossible']:{}".format(len(word2index_dict), word2index_dict['impossible']))

