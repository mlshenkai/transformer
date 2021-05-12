# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/12 16:47
# @Organization: YQN
import torch.nn as nn
from backbone.scaled_dot_product_attention import ScaleDotProductAttention
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_k, d_v, d_model):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.scale_dot_product_attention = ScaleDotProductAttention()

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        :param input_Q: [batch_size,len_q,d_model]
        :param input_K: [batch_size,len_k,d_model]
        :param input_V: [batch_size,len_v(=len_k),d_model]
        :param attn_mask: [batch_size,len_q,len_k]
        :return:
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B,S,D)-proj->(B,S,D_new)-split->(B,S,H,W)-trans->(B,H,S,W)

        # [batch_size,len_q,d_model]-linear->[batch_size,len_q,d_k*n_heads]-split->[batch_size,len_q,n_heads,d_k]-trans->[batch_size,n_heads,len_q,d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # [batch_size,n_heads,d_v,d_k]

        # context [batch,n_heads,len_q,d_v] ,attn [batch_size,n_heads,len_q,len_k]
        context, attn = self.scale_dot_product_attention(Q, K, V, attn_mask)

        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # [batch_size,len_q,n_heads*d_v]
        output = self.fc(context)  # [batch_size,len_q,d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn
