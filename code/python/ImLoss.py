from __future__ import division
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Function
from torch.autograd import Variable

class ImLoss(nn.Module):
    def __init__(self):
        super(ImLoss,self).__init__()

    def forward(self, predictions, targets):
        predictions=predictions.detach().numpy()
        targets=targets.detach().numpy()
        expect = 1 / (1 + np.exp(-targets))
        input = 1 / (1 + np.exp(-predictions))
        loss = -np.multiply(expect, np.log(input)) - np.multiply((1 - expect), np.log((1 - input)))
        loss=loss.sum()
        loss=torch.Tensor(loss)
        return loss

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()  # 没有需要保存的参数和状态信息

    def forward(self, x, y):  # 定义前向的函数运算即可
        x=torch.exp(-x)
        x=1/(1+x)
        y=torch.exp(-y)
        y=1/(1+y)
        d = -(y * torch.log(x) + (1 - y) * torch.log(1 - x))
        return torch.sum(d)
        # return torch.sum(100*torch.pow((x - y)*y, 2)+torch.pow((x - y), 2))
        # return( nn.CrossEntropyLoss(x,y) )

    def myNomal(a):
        b = torch.exp(-a)
        c = 1 / (1 + b)
        # d = -(expect * torch.log(c) + (1 - expect) * torch.log(1 - c))
        return c


