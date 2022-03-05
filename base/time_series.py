# -*- coding: utf-8 -*-
import numpy as np
import torch

bikes_numpy = np.loadtxt("../data/bike_sharing_data/hour-fixed.csv",
                         dtype=np.float32,
                         delimiter=",",
                         skiprows=1,
                         converters={1: lambda x: float(x[8:10])})

bikes = torch.from_numpy(bikes_numpy)
print("bikes:{}\nshape:{}\nbikes.stride:{}".format(bikes, bikes.shape, bikes.stride()))

# shape:torch.Size([17520, 17])
# bikes.stride:(17, 1)

# 将数据重新排列为三个轴（日期,小时,然后是17列）
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
print("daily_bikes.shape: {}, daily_bikes.stride: {}".format(daily_bikes.shape, daily_bikes.stride()))
# daily_bikes.shape: torch.Size([730, 24, 17]), daily_bikes.stride: (408, 17, 1)

# 最右边的维度是原始数据集中的列数。在中间维度中，你将时间分为24个连续小时的块。
# 换句话说，你现在每天有C个通道的N个L小时的序列。为了获得你所需的NxCxL顺序，你需要转置张量：

daily_bikes = daily_bikes.transpose(1, 2)
print("daily_bikes.shape:{}, daily_bikes.stride:{}".format(daily_bikes.shape, daily_bikes.stride()))
# daily_bikes.shape:torch.Size([730, 17, 24]), daily_bikes.stride:(408, 1, 17)

first_day = bikes[:24].long()
print("first_day.shape:{}".format(first_day.shape))
# first_day.shape:torch.Size([24, 17])
weather_onehot = torch.zeros(first_day.shape[0], 4)

# 取第一天天气状况那一列的所有数据
print("取第一天天气状况那一列的所有数据: {}\nunsqueeze:{}".format(first_day[:, 9], first_day[:, 9].unsqueeze(1)))

# unsqueeze:tensor([[1],
#         [1],
#         [1],
#         [1],
#         [1],
#         [2],
#         [1],
#         [1],
#         [1],
#         [1],
#         [1],
#         [1],
#         [1],
#         [2],
#         [2],
#         [2],
#         [2],
#         [2],
#         [3],
#         [3],
#         [2],
#         [2],
#         [2],
#         [2]])

weather_onehot.scatter_(
    dim=1,
    # one-hot是从1开始编码, 因此这里减去1 ？？
    index=first_day[:, 9].unsqueeze(1) - 1,
    value=1.0
)

print("weather_onehot:{}".format(weather_onehot))

# 使用cat函数将矩阵连接到原始数据集。看第一个结果：
first_day_cat = torch.cat((bikes[:24], weather_onehot), dim=1)[:1]
print("first_day_cat:{}".format(first_day_cat))
# first_day_cat:tensor([[ 1.0000,  1.0000,  1.0000,  0.0000,  1.0000,  0.0000,  0.0000,  6.0000,
#           0.0000,  1.0000,  0.2400,  0.2879,  0.8100,  0.0000,  3.0000, 13.0000,
#          16.0000,  1.0000,  0.0000,  0.0000,  0.0000]])

# 你也可以使用重新排列的daily_bikes张量完成相同的操作。请记住，它的形状为（B,C,L），其中L=24

daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])
print("daily_weather_onehot.shape:{}".format(daily_weather_onehot.shape))
# daily_weather_onehot.shape:torch.Size([730, 4, 24])
daily_weather_onehot.scatter_(dim=1, index=daily_bikes[:, 9, :].long().unsqueeze(1) - 1, value=1.0)
print("daily_weather_onehot shape:{}".format(daily_weather_onehot.shape))
# daily_weather_onehot shape:torch.Size([730, 4, 24])

# 沿C维度连接
daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)
print("daily_bikes: {}\nshape:{}".format(daily_bikes, daily_bikes.shape))
# shape:torch.Size([730, 21, 24])

# 将第10列中的温度调整到[0.0, 1.0]之间
# 方法1：
temp = daily_bikes[:, 10,:]
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:,10,:] = (daily_bikes[:,10,:] - temp_min)/ (temp_max - temp_min)

# 方法2： 减去平均值并除以标准差
temp = daily_bikes[:, 10,:]
daily_bikes[:, 10,:] = (daily_bikes[:, 10,:] - torch.mean(temp))/ torch.std(temp)

