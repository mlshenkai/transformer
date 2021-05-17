# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/12 18:33
# @Organization: YQN
import torch
import torch.nn as nn
from backbone.multi_head_attention import MultiHeadAttention
from backbone.feed_forward import PoswiseFeedForWardNet

class DecoderLayer(nn.Module):
    def __init__(self,n_heads,d_k,d_v,d_model,d_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn=MultiHeadAttention(n_heads,d_k,d_v,d_model)
        self.dec_enc_attn=MultiHeadAttention(n_heads,d_k,d_v,d_model)
        self.pos_ffn=PoswiseFeedForWardNet(d_model,d_ff)

    def forward(self,dec_inputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

