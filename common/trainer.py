# -*- coding: utf-8 -*-
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        # 记录loss列表
        self.loss_list = []
        # 评估间隔
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, train_loader, max_epochs, eval_interval=100, val_loader=None):
        self.eval_interval = eval_interval
        model = self.model
        optimizer = self.optimizer
        loss_fn = self.loss_fn

        total_loss = 0
        loss_count = 0
        model.train()
        start_time = time.time()
        for epoch in range(max_epochs):

            for index_batch, batch in enumerate(train_loader):
                x_batched = batch[0]
                y_batched = batch[1]

                # 前向计算
                logist = model(x_batched)
                optimizer.zero_grad()
                loss = loss_fn(logist, y_batched)
                loss.backward()
                optimizer.step()
                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) and (index_batch % eval_interval == 0):
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    # int(len(data_loader)/batch_size + 1 是dataloader在一个线程的情况下的计算方式，多个线程的情况下还需要处以线程数
                    print('| epoch %d |  iter %d / %d | time %d[s] | train loss %.2f'
                          % (self.current_epoch + 1, index_batch + 1,
                             int(len(train_loader) / train_loader.batch_size) + 1,
                             elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss = 0
                    loss_count = 0

            # 每一个打印当前的训练误差和评估误差
            if val_loader:
                train_loss = total_loss / loss_count
                evaluate_loss = self._compute_model_evaluate_loss(val_loader)
                print(
                    'epoch %d | train loss %.2f | evaluate loss %.2f' % (self.current_epoch, train_loss, evaluate_loss))

            self.current_epoch += 1

    def _compute_model_evaluate_loss(self, val_loader):
        """
        计算模型当前的在评估数据集上的损失
        :return:
        """
        total_loss = 0
        loss_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x_val_batch = batch[0]
                y_val_batch = batch[1]

                logist = self.model(x_val_batch)
                loss = self.loss_fn(logist, y_val_batch)
                total_loss += loss
                loss_count += 1
            avg_loss = total_loss / loss_count
            return avg_loss

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()

    def evaluate(self, model, val_data_loader):
        """
        注意这里的val_data_loader shuffle参数应设置为False
        :param val_data_loader(torch.utils.data.DataLoader): 评估数据集的data loader
        :param model: 待评估的模型
        :return:
        """
        correct = 0
        total = 0
        with torch.no_grad():  # 评估不需要求梯度
            for val_sample in val_data_loader:
                x_val = val_sample[0]
                y_val = val_sample[1]
                outputs = model(x_val)
                # 这里out的和不为1
                # out = list(map(lambda x:sum(x), outputs))
                # print("sum out: {}".format(out))
                _, predicted = torch.max(outputs, dim=1)
                total += y_val.shape[0]
                correct += (predicted == y_val).sum().item()
            print("Accuracy: %f" % (correct / total))