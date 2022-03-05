# -*- coding: utf-8 -*-
# 使用张量表示真实的数据

# 1.表格数据,葡萄酒数据: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
import csv
import numpy as np

wine_path = "../data/representation/winequality-white.csv"
wine_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
print("wine_numpy:{}, \n shape:{}".format(wine_numpy, np.shape(wine_numpy)))

col_list = next(csv.reader(open(wine_path), delimiter=';'))
print("shape:{}, \ncol_list:{}".format(wine_numpy.shape, col_list))

# 将numpy转换成tensor
import torch

wineq = torch.from_numpy(wine_numpy)

# 最后一列为质量分数，不进入模型，排除在外
data = wineq[:, :-1]
target = wineq[:, -1].long()  # 最后一列数据
target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
print("target_onehot:{}".format(target_onehot))

# 计算data每一列的平均值
data_mean = data.mean(dim=0)
# 标准差
data_var = data.var(dim=0)

data_normalized = (data - data_mean) / torch.sqrt(data_var)
print("data_normalized: {}".format(data_normalized))

# 找出质量小于或等于3的样本
bad_indexes = torch.le(target, 3)
print("bad_indexes.shape: {}\nbad_indexes.dtype:{}\nbad_indexes.sum()：{}".format(bad_indexes.shape, bad_indexes.dtype,
                                                                                 bad_indexes.sum()))

bad_data = data[bad_indexes]
print("bad_data shape:{}".format(bad_data.shape))

# 对于numpy数组和PyTorch张量，＆运算符执行逻辑和运算
mid_data = data[torch.gt(target, 3) & torch.lt(target, 7)]
good_data = data[torch.ge(target, 7)]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

# >>0 fixed acidity          7.60   6.89   6.73
#  1 volatile acidity       0.33   0.28   0.27
#  2 citric acid            0.34   0.34   0.33
#  3 residual sugar         6.39   6.71   5.26
#  4 chlorides              0.05   0.05   0.04
#  5 free sulfur dioxide   53.33  35.42  34.55
#  6 total sulfur dioxide 170.60 141.83 125.25
#  7 density                0.99   0.99   0.99
#  8 pH                     3.19   3.18   3.22
#  9 sulphates              0.47   0.49   0.50
# 10 alcohol               10.34  10.26  11.42

# 劣质葡萄酒似乎具有更高的二氧化硫总含量（total sulfur dioxide），另外还有其他差异。
# 可以使用二氧化硫总含量的阈值作为区分好酒差酒的粗略标准

total_sulfur_threshold = 141.83
# 取二氧化硫的全部数据
total_sulfur_data = data[:, 6]

predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)

print(
    "predicted_indexes.shape:{}, predicted_indexes.dtype:{}, predicted_indexes.sum():{}".format(predicted_indexes.shape,
                                                                                                predicted_indexes.dtype,
                                                                                                predicted_indexes.sum()))
# predicted_indexes.shape:torch.Size([4898]), predicted_indexes.dtype:torch.bool, predicted_indexes.sum():2727

# 获取优质葡萄酒的索引
actual_indexes = torch.gt(target, 5)
print("actual_indexes.shape:{}, actual_indexes.dtype:{}, actual_indexes.sum():{}".format(actual_indexes.shape,
                                                                                         actual_indexes.dtype,
                                                                                         actual_indexes.sum()))

n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()

print("n_matches:{}, n_matches / n_predicted:{}, n_matches / n_actual:{}".format(n_matches, n_matches / n_predicted,
                                                                                 n_matches / n_actual))
# n_matches:2018, n_matches / n_predicted:0.74000733406674, n_matches / n_actual:0.6193984039287906
