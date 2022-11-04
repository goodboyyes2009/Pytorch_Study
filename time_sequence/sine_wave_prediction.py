# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse


# See https://github.com/pytorch/examples/blob/main/time_sequence_prediction/train.py

def fake_sine_wave_date():
    np.random.seed(2)
    T = 20
    L = 1000
    N = 100

    x = np.empty((N, L), 'int64')  # shape: [100 x 1000]
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)  # shape: [100 x 1000]
    data = np.sin(x / 1.0 / T).astype('float64')  # shape: [100 x 1000]
    print(f"data shape: {np.shape(data)}")
    torch.save(data, open('traindata.pt', 'wb'))


class Sequence(nn.Module):

    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []

        # init
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double).to(device)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double).to(device)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double).to(device)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double).to(device)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="time sequence example")
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # load data and make training set.
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1]).to(device)
    target = torch.from_numpy(data[3:, 1:]).to(device)
    test_input = torch.from_numpy(data[:3, :-1]).to(device)
    test_target = torch.from_numpy(data[:3, 1:]).to(device)


    # build model
    seq = Sequence()
    seq.double().to(device)
    mse_loss = nn.MSELoss().to(device)

    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    # begin to train
    for i in range(opt.steps):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = mse_loss(out, target)
            print("loss:", loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = mse_loss(pred[:, :-future], test_target)
            print("test loss:", loss.item())
