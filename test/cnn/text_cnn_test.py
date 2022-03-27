# -*- coding: utf-8 -*-
from cnn.text_cnn import TextCNN


def test_summary():
    vocab_size = 100
    embedding_dim = 128
    text_cnn = TextCNN(vocab_size=vocab_size, embedding_dim=embedding_dim)
    print(text_cnn.summary())