# -*- coding: utf-8 -*-

class Config(object):
    data_path = '../../data/quan_tang_shi/json/' # 诗歌的文件存放的目录
    pickle_path = 'tang.npz'
    author = None # 作者信息
    constrain = None
    lr = 1e-3
    epoch = 10
    batch_size = 124
    max_sequence_length = 125
    max_gen_len = 200
    model_path = None
    use_gpu=True


    # gen tang shi config
    prefix_words = '床前明月光，疑似地上霜。'# 不是生成诗歌的部分，用来控制生成诗歌的语境
    start_words='落霞与孤鹜齐飞'
    model_checkpoint_path='checkpoints/tang'