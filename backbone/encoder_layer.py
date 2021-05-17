# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/12 18:21
# @Organization: YQN

import torch
import torch.nn as nn
from backbone.multi_head_attention import MultiHeadAttention
from backbone.feed_forward import PoswiseFeedForWardNet


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_k, d_v, d_model, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_k, d_v, d_model)
        self.pos_ff = PoswiseFeedForWardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_input: [batch_size,src_len,d_mdoel]
        :param enc_self_attn_mask: [batch_size,src_len,src_len]
        :return:
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ff(enc_outputs)
        return enc_outputs, attn
