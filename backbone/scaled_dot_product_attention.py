# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/12 16:16
# @Organization: YQN
import numpy as np
import torch.nn as nn
import torch


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size,n_heads,len_q,d_k]
        :param K: [batch_size,n_heads,len_k,d_k]
        :param V: [batch_size,n_heads,len_v(=len_k),d_v]
        :param attn_mask: [batch_size,n_heads,len_q,len_k]
        :return:
        """
        batch_size, n_heads, len_q, d_k = Q.size()
        scores: torch.FloatTensor = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            d_k)  # [batch_size,n_heads,len_q,len_k]
        scores.masked_fill_(attn_mask, -1e9)  # [batch_size,n_heads,len_q,len_k]
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size,n_heads,len_q,len_k]
        context = torch.matmul(attn, V)  # [batch,n_heads,len_q,d_v]
        return context, attn
