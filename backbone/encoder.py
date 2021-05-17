# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/12 19:21
# @Organization: YQN
import torch
import torch.nn as nn
from backbone.position_encoding import PositionalEncoding
from backbone.encoder_layer import EncoderLayer
from backbone.mask_feature import get_attn_pad_mask


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, n_layers, n_heads, d_k, d_v, d_model, d_ff):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(n_heads, d_k, d_v, d_model, d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        :param enc_inputs: [batch_size,src_len]
        :return:
        """
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size,src_len,d_model]
        enc_outputs = self.pos_emb(enc_outputs)  # [batch_size,src_len,d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size,src_len,src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


if __name__=="__main__":
    d_model=512
    d_ff=2048
    d_k=d_v=64
    n_layers=6
    n_heads=8
    encoder=Encoder(10,n_layers,n_heads,d_k,d_v,d_model,d_ff)
    print(encoder)
    src_input=torch.randint(10,(1,10))
    enc_outputs, enc_self_attns=encoder(src_input)
    print(enc_outputs)