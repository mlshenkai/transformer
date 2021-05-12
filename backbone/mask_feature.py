# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/12 15:47
# @Organization: YQN
import torch
import numpy as np


# Pad Mask
def get_attn_pad_mask(seq_q: torch.Tensor, seq_k: torch.Tensor):
    """
    compute q and k pad
    :param seq_q: list [batch_size,seq_len]
    :param seq_k: list [batch_size,seq_len]
    :return:
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size,1,len_k],
    return pad_attn_mask.expend(batch_size, len_q, len_k)


# subsequence mask 同于在decode中屏蔽未来时刻的单词
def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


if __name__ == "__main__":
    shape = [2, 2]
    bb = np.triu(np.ones(shape), k=1)
    subsequence_mask = torch.from_numpy(bb).byte()
    print(subsequence_mask)
