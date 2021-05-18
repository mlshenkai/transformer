# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/18 10:55
# @Organization: YQN
import torch
import torch.nn as nn
from models.backbone.position_encoding import PositionEncoding
from models.backbone.encode_layer import EncoderLayer
from models.backbone.mask_feature import get_attn_pad_mask


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_heads, d_k, d_v, d_ff):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_heads)
        ])

    def forward(self, enc_inputs):
        """
        :param enc_inputs: [batch_size,src_len]
        :return:
        """
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


if __name__=="__main__":
    encode=Encoder(12,512,1,512,512,1024)
    input=torch.randint(12,[1,12])
    output=encode(input)
    print(output)