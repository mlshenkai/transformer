# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/18 10:10
# @Organization: YQN
import torch
import torch.nn as nn
from models.backbone.feed_forward import PoswiseFeedForWardNet
from models.backbone.multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.feed_forward = PoswiseFeedForWardNet(d_model, d_ff)
        self.self_attn = MultiHeadAttention(n_heads, d_k, d_v, d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [batch_size,src_len,d_model]
        :param enc_self_attn_mask:[batch_size,src_len,src_len]
        :return:
        """
        enc_outputs, attn = self.self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.feed_forward(enc_outputs)
        return enc_outputs, attn
