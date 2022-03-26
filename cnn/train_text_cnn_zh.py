# -*- coding: utf-8 -*-
import sys

sys.path.append('..')
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from common.dataset import ChineseNewsData, Vocabulary, ToTensor, Tokenize
from torchvision import transforms
from cnn.text_cnn import TextCNN
from common.trainer import Trainer
from common.tokenization import get_stop_words, token_function

parser = argparse.ArgumentParser(description="Train Text CNN Model")
parser.add_argument('--data_root_path', help="数据根目录", required=True, type=str)
parser.add_argument('--device', help="cpu or gpu", type=str, default='gpu')
parser.add_argument('--num_filters', help='CNN滤波器的个数', required=True, type=int, nargs='+')
parser.add_argument('--filter_sizes', help='滤波器的kernel size', required=True, type=int, nargs='+')
parser.add_argument('--num_classes', help="分类任务的标签数量", type=int, default=5)
parser.add_argument('--dropout', help='dropout', type=float, default=0.5)
parser.add_argument('--learning_rate', help='learning_rate', type=float, default=0.01)
parser.add_argument('--max_epochs', help='max_epochs', type=int, default=10)
parser.add_argument('--batch_size', help='batch_size', type=int, default=1000)
parser.add_argument('--eval_interval', help='eval_interval', type=int, default=100)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    torch.multiprocessing.set_start_method('spawn')

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu' if args.device == 'gpu' else 'cpu'
    device = torch.device(device_str)
    print("device: {}".format(device))

    news_train_data = ChineseNewsData(split_char='\t', data_root_path=args.data_root_path,
                                      transforms=transforms.Compose([Tokenize(), ToTensor(device=device)]))

    news_eval_data = ChineseNewsData(train=False, split_char='\t', data_root_path=args.data_root_path,
                                     transforms=transforms.Compose([Tokenize(), ToTensor(device=device)]))

    train_data_loader = DataLoader(news_train_data, batch_size=args.batch_size, shuffle=True)
    eval_data_loader = DataLoader(news_eval_data, batch_size=args.batch_size, shuffle=False)

    vocab = Vocabulary(token_fn=token_function, stop_words=get_stop_words())

    embedding_dim = vocab.max_sentence_length
    print("vocab size: {}".format(vocab.vocab_size))
    # num_filters = [100, 100, 100]
    # filter_sizes = [2, 3, 3]
    # num_classes = 5
    # dropout = 0.5
    # learning_rate = 0.001

    # 初始化模型
    text_cnn_model = TextCNN(vocab_size=vocab.vocab_size,
                             embedding_dim=embedding_dim,
                             num_filters=args.num_filters,
                             filter_sizes=args.filter_sizes,
                             num_classes=args.num_classes,
                             dropout=args.dropout)

    # send model to gpu
    text_cnn_model.to(device)

    # 初始化优化器
    optimizer = optim.Adadelta(text_cnn_model.parameters(), lr=args.learning_rate, rho=0.95)

    # 初始化loss函数, CrossEntropyLoss<==>LogSoftmax+NLLLoss
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(text_cnn_model, optimizer, loss_fn)
    trainer.fit(train_data_loader, max_epochs=args.max_epochs, eval_interval=args.eval_interval,
                val_loader=eval_data_loader)
    # trainer.plot()
    trainer.evaluate(text_cnn_model, eval_data_loader)
