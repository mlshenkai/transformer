# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/11 11:48
# @Organization: YQN
import torch
import numpy as np


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


if __name__ == "__main__":
    position_embedding = get_sinusoid_encoding_position(10, 512)
    print(position_embedding)
