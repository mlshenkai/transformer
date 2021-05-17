# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/17 20:29
# @Organization: YQN
import torch
import torch.nn as nn
import numpy as np

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self,Q,K,V,attn_mask):
        """
        :param Q: [batch_size,n_heads,len_q,d_k]
        :param K: [batch_size,n_heads,len_k,d_k]
        :param V: [batch_size,n_heads,len_v(len_v=len_k),d_v]
        :param attn_mask: [batch_size,n_heads,len_q,len_q]
        :return: [batch_size,n_heads,]
        """
        batch_size,n_heads,len_q,d_k=Q.size()
        score=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k) #[batch_size,n_head,len_q,len_k]
        score.masked_fill_(attn_mask,-1e9)
        attn=nn.Softmax(dim=-1)(score)#[batch_size,n_heads,len_q,len_k]
        context=torch.matmul(attn,V)#[batch_size,n_heads,len_q,d_v]
        return context,attn



