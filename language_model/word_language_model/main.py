# -*- coding: utf-8 -*-
import os
import logging
import argparse
import torch
import torch.nn as nn
import time
import math
from data_helper import Corpus, Dictionary
import model

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Pytorch Wikitext-2 RNN/LSTM/GRU/Transformers Lanuage Model.")
parser.add_argument("--data", type=str, default='../../data/wikitext-2', help='location of the data corpus.')
parser.add_argument('--model', type=str, default='Transformer',
                    help='type of network(RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200, help="size of word embedding")
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of rnn layers')
parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=10, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=35, help='sentence length')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers( 0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='', help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true', help='verify the code and the model')

args = parser.parse_args()

# set random seed

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logger.warning("WARNING: You have a CUDA device. so you should probably run with --cuda")

if args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

###############################################################################
# Load data
###############################################################################

corpus = Corpus(args.data)


def batchfiy(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


eval_batch_size = 10
train_data = batchfiy(corpus.train_data, args.batch_size)
val_data = batchfiy(corpus.valid_data, args.batch_size)
test_data = batchfiy(corpus.test_data, args.batch_size)

###############################################################################
# Build the model
###############################################################################
ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(
        device)
criterion = nn.NLLLoss()


###############################################################################
# Traning code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensor, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i: i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training model which enables dropout
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset/
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            # data shape=(sequence_length, batch_size]
            # h_0 shape=(nlayers, batch_size, nhid)
            # 如果是rnn_type=LSTM, 需要c_0 shape=(nlayers, batch_size, nhid)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # 这里手动进行参数梯度裁剪和权重更新
        # clip_gard_norm help prevent the exploding gradient problem in RNNs / LSTM.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f}| '
                  'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, len(train_data) // args.bptt, lr,
                                                      elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


# loop over epochs
lr = args.lr
best_var_loss = None

# Ar any point you can hit Ctrl+C to break out of training early
try:
    for epoch in range(1, args.epochs):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        if not best_var_loss or val_loss < best_var_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_var_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print("Exiting from training early")

# Load the best saved model
if os.path.exists(args.save):
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
