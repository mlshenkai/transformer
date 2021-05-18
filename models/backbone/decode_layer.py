# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/18 10:40
# @Organization: YQN
import torch
import torch.nn as nn
from models.backbone.feed_forward import PoswiseFeedForWardNet
from models.backbone.multi_head_attention import MultiHeadAttention
class DecoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_k,d_v,d_ff):
        super(DecoderLayer, self).__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.d_k=d_k
        self.d_v=d_v
        self.d_ff=d_ff
        self.feed_ward=PoswiseFeedForWardNet(d_model,d_ff)
        self.self_dec_attn=MultiHeadAttention(n_heads,d_k,d_v,d_model)
        self.dec_enc_attn=MultiHeadAttention(n_heads,d_k,d_v,d_model)

    def forward(self,dec_inputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
        dec_outputs,attn=self.self_dec_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs=self.feed_ward(dec_outputs)
        return dec_outputs,attn,dec_enc_attn
