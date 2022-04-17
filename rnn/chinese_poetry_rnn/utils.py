# -*- coding: utf-8 -*-
import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("There are {} GPU(s) available".format(torch.cuda.device_count()))
        print("Device name:{}".format(torch.cuda.get_device_name()))
    else:
        print("No GPU available, use CPU instead.")
        device = torch.device("cpu")
    return device