# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/14 11:09
# @Organization: YQN

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout_prob=0.1, max_len=5000):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:,::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        self.dropout=nn.Dropout(dropout_prob)#[max_len,d_model]
        pe=pe.unsqueeze(0)#[batch_size,max_len,d_model]
        self.register_buffer("pe",pe)

    def forward(self,x):
        """
        :param x: [batch_size,src_len,d_model]
        :return:
        """
        x=x+self.pe[:,:x.size(1)]
        return self.dropout(x)




if __name__ == "__main__":
    plt.figure(figsize=(15, 5))
    pe = PositionEncoding(20, 0)
    x = torch.zeros(1, 100, 20)
    y=pe(x)
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d"%p for p in [4,5,6,7]])
    plt.show()
