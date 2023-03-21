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

# 词汇表
vocab_dict = data_helper.Corpus(args.data).dictionary
num_of_tokens = len(vocab_dict)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
print("is_transformer_model: {}".format(is_transformer_model))

if not is_transformer_model:
    hidden = model.init_hidden(1)

# 从词表中随机选取一个字作为生成文本的输入字符
input_text = torch.randint(num_of_tokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, 'w', encoding='utf-8') as f:
    with torch.no_grad():
        for i in range(args.words):
            if is_transformer_model:
                output = model(input_text, has_mask=False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                # torch.multinominal方法可以根据给定权重对数组进行多次采样，返回采样后的元素下标
                word_idx = torch.multinomial(input=word_weights, num_samples=1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input_text = torch.cat(tensors=[input_text, word_tensor], dim=0)
            else:
                output, hidden = model(input_text, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(input=word_weights, num_samples=1)[0]
                input_text.fill_(word_idx)

            word = vocab_dict.id2word[word_idx]

            f.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print("| Generate {}/{} words".format(i, args.words))