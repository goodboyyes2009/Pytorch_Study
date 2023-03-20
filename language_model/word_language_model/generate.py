# -*- coding: utf-8 -*-
import argparse
import torch

import data_helper

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')
# Model parameters.
parser.add_argument('--data', type=str, default='../../data/wikitext-2', help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt', help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt', help='output file for generated text')
parser.add_argument('--words', type=int, default='1000', help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100, help='reporting interval')
args = parser.parse_args()

# set rand seed
torch.manual_seed(args.seed)

if args.cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device)

# 开启评估模式
model.eval()

