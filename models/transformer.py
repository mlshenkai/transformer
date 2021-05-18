# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/18 11:54
# @Organization: YQN
import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model, n_heads, d_k, d_v, d_ff):
        super(Transformer, self).__init__()
        self.encode=Encoder(src_vocab_size,d_model,n_heads,d_k,d_v,d_ff)
        self.decode=Decoder(tgt_vocab_size,d_model,n_heads,d_k,d_v,d_ff)
        self.projection=nn.Linear(d_model,tgt_vocab_size,bias=False)

    def forward(self,enc_inputs,dec_inputs):
        """
        :param enc_inputs: [batch_size,src_len]
        :param dec_inputs: [batch_size,tgt_len]
        :return:
        """
        enc_outputs,enc_self_attns=self.encode(enc_inputs) #[batch_size,src_len,d_model]
        dec_outputs,dec_self_attns,dec_enc_attns=self.decode(dec_inputs,enc_inputs,enc_outputs)#[batch_size,tgt_len,d_model]
        dec_logits=self.projection(dec_outputs)#[batch_size,tgt_len,tgt_vocab_size]
        dec_logits=dec_logits.view(-1,dec_logits.size(-1))#[batch_size*tgt_len,tgt_vocab_size]
        return dec_logits,enc_self_attns,dec_self_attns,dec_enc_attns


if __name__=="__main__":
    model=Transformer(5,6,512,8,64,64,2048)
    print(model)
    enc_inputs=torch.randint(5,(1,5))
    dec_inputs=torch.randint(6,(1,6))
    dec_logits,enc_self_attns,dec_self_attns,dec_enc_attns=model(enc_inputs,dec_inputs)
    print(dec_logits)