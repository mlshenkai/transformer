# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/11 11:48
# @Organization: YQN
import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def get_sinusoid_encoding_position(n_position: int, d_model: int) -> torch.FloatTensor:
    """
    依据正弦曲线计算位置编码
    :param n_position: int 字符个数
    :param d_model: int embedding size
    :return: FloatTensor 该位置的正选曲线编码
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_position_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_position_angle_vec(position_i) for position_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        # pe=pe.unsqueeze(0).transpose(0,1)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        # x=x+self.pe[:x.size(0),:]
        x=x+self.pe[:,:x.size(1)]
        return self.dropout(x)


if __name__ == "__main__":
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    x = torch.zeros(1, 100, 20)
    y=pe(x)
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d"%p for p in [4,5,6,7]])
    plt.show()

