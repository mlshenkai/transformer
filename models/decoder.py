# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/18 11:34
# @Organization: YQN
import torch
import torch.nn as nn
from models.backbone.position_encoding import PositionEncoding
from models.backbone.decode_layer import DecoderLayer
from models.backbone.mask_feature import get_attn_pad_mask,get_attn_subsequence_mask

class Decoder(nn.Module):
    def __init__(self,tgt_vocab_size,d_model,n_heads,d_k,d_v,d_ff):
        super(Decoder, self).__init__()
        self.tgt_emb=nn.Embedding(tgt_vocab_size,d_model)
        self.pos_emb=PositionEncoding(d_model)
        self.layers=nn.ModuleList([DecoderLayer(d_model,n_heads,d_k,d_v,d_ff) for _ in range(n_heads)])

    def forward(self,dec_inputs,enc_inputs,enc_outputs):
        """
        :param dec_inputs: [batch_size,tgt_len]
        :param enc_inputs: [batch_size,src_len]
        :param enc_outputs: [batch_size,src_len,d_model]
        :return:
        """
        dec_outputs=self.tgt_emb(dec_inputs)#[batch_size,tgt_len,d_model]
        dec_outputs=self.pos_emb(dec_outputs)#[batch_size,tgt_len,d_model]
        dec_self_attn_mask=get_attn_pad_mask(dec_inputs,dec_inputs)#[batch_size,tgt_len,tgt_len]
        dec_self_attn_subsequence_mask=get_attn_subsequence_mask(dec_inputs)#[batch_size,tgt_len,tgt_len]

        dec_self_attn_mask=torch.gt((dec_self_attn_mask+dec_self_attn_subsequence_mask),0)#[batch_size,tgt_len,tgt_len]

        dec_enc_attn_mask=get_attn_pad_mask(dec_inputs,enc_inputs)#[batch_size,tgt_len,src_len]

        dec_self_attns,dec_enc_attns=[],[]
        for layer in self.layers:
            dec_outputs,dec_self_attn,dec_enc_attn=layer(dec_outputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs,dec_self_attns,dec_enc_attns



