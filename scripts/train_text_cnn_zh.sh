#!/usr/bin/env bash
python  ../cnn/train_text_cnn_zh.py \
        --num_filters 100 100 100 \
        --filter_sizes 3 4 5 \
        --max_epochs 100 \
        --word_embedding tencent \
        --data_root_path=/home/hj/dataset/news_data/news_zh