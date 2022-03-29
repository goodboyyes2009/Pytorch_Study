# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from common.dataset import ChineseNewsData


def test_chinese_news_dataset():
    data_root_path = '/home/hj/dataset/news_data/news_zh'

    news_data = ChineseNewsData(split_char='\t', data_root_path=data_root_path,
                                transforms=None)
    for i in range(len(news_data)):
        sample = news_data[i]
        text, label = sample[0], sample[1]
        # 打印前4条
        if i > 3:
            break
        print("text:{}, label:{}".format(text, label))


def test_torchvision_transforms():
    from torchvision import transforms, utils
    from common.dataset import ToTensor, Tokenize
    from common.tokenization import get_stop_words, Vocabulary, token_function
    data_root_path = '/home/hj/dataset/news_data/news_zh'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    stop_words = get_stop_words()
    vocabulary = Vocabulary(stop_words=stop_words, token_fn=token_function)

    news_data = ChineseNewsData(split_char='\t', data_root_path=data_root_path,
                                transforms=transforms.Compose([Tokenize(encode_fn=vocabulary.encode), ToTensor(device=device)]))

    print(len(news_data))

    for i in range(len(news_data)):
        sample = news_data[i]
        print("iter sample: {}".format(sample))
        text, label = sample[0], sample[1]
        # 打印前2条
        if i > 1:
            break
        print("text:{}, label:{}".format(text, label))


def test_dataloader_batch_sampler():
    from common.dataset import Tokenize, ToTensor
    from common.tokenization import Vocabulary, get_stop_words, token_function
    # 需要添加下面这一行代码，否则报下面的错误
    # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    torch.multiprocessing.set_start_method('spawn')
    from torchvision import transforms, utils
    data_root_path = '/home/hj/dataset/news_data/news_zh'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    stop_words = get_stop_words()
    vocabulary = Vocabulary(stop_words=stop_words, token_fn=token_function)

    news_data = ChineseNewsData(split_char='\t', data_root_path=data_root_path,
                                transforms=transforms.Compose([Tokenize(encode_fn=vocabulary.encode), ToTensor(device=device)]))

    # Batch sample
    dataloader = DataLoader(news_data, batch_size=4,
                            shuffle=True, num_workers=0)

    for index_batch, sample_batched in enumerate(dataloader):
        print("index: {}, len of sample_batched: {}".format(index_batch, len(sample_batched)))
