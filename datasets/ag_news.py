# -*- coding: utf-8 -*-
import os
import csv
import sys
import re
import torch

from torchtext.data import Field, TabularDataset

csv.field_size_limit(sys.maxsize)


def clean_string(string):
    """
    Performs tokenization and string cleaning for the AG_NEWS dataset
    """
    # " #39;" is apostrophe
    string = re.sub(r" #39;", "'", string)
    # " #145;" and " #146;" are left and right single quotes
    string = re.sub(r" #14[56];", "'", string)
    # " #147;" and " #148;" are left and right double quotes
    string = re.sub(r" #14[78];", "\"\"", string)
    # " &lt;" and " &gt;" are < and >
    string = re.sub(r" &lt;", "<", string)
    string = re.sub(r" &gt;", ">", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip().split()


def process_labels(label_str, num_classes):
    """

    :param label_str: 标签字符串
    :param num_classes: 类别个数
    :return:
    """
    label_num = int(label_str)  # label is one of "1", "2", "3", "4"
    label = [0.0] * num_classes
    label[label_num - 1] = 1.0
    return label


class AGNews(TabularDataset):
    # AG_News数据集中包含的分类任务标签的个数
    NUM_CLASSES = 4
    # 是否是多标签预测任务
    IS_MULTILABEL = False

    TEXT_FIELD = Field(batch_first=True, tokenize=clean_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=lambda s: process_labels(s, AGNews.NUM_CLASSES))

