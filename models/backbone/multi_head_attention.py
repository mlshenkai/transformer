# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/17 21:29
# @Organization: YQN
import torch
import torch.nn as nn
from models.backbone.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads, d_k, d_v, d_model):
        super(MultiHeadAttention, self).__init__()
        self.n_heads=n_heads
        self.d_k=d_k
        self.d_v=d_v
        self.d_model=d_model
        self.W_Q=nn.Linear(d_model,d_k*n_heads,bias=False)
        self.W_K=nn.Linear(d_model,d_k*n_heads,bias=False)
        self.W_V=nn.Linear(d_model,d_v*n_heads,bias=False)
        self.fc=nn.Linear(d_v*n_heads,d_model,bias=False)
        self.scale_dot_product_attn=ScaleDotProductAttention()

    def forward(self,input_q,input_k,input_v,attn_mask):
        """
        :param input_q:[batch_size,len_q,d_model]
        :param input_k:[batch_size,len_k,d_model]
        :param input_v:[batch_size,len_v(len_v=len_k),d_model]
        :param attn_mask:[batch_size,len_q,len_k]
        :return:
        """
        residual,batch_size=input_q,input_q.size(0)
        Q=self.W_Q(input_q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)#[batch_size,n_heads,len_q,d_k]
        K=self.W_K(input_k).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)#[batch_size,n_heads,len_q,d_k]
        V=self.W_V(input_v).view(batch_size,-1,self.n_heads,self.d_v).transpose(1,2)#[batch_size,n_heads,len_q,d_v]
        attn_mask=attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)#[batch_size,n_heads,len_q,len_k]
        context,attn=self.scale_dot_product_attn(Q,K,V,attn_mask)#context [batch_size,n_heads,len_q,d_v], attn [batch_size,n_heads,len_q,len_k]
        context=context.transpose(1,2).reshape(batch_size,-1,self.n_heads*self.d_v)
        output=self.fc(context)#[batch_size,len_q,d_model]
        return nn.LayerNorm(self.d_model)(output+residual),attn






