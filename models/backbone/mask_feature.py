# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/17 21:04
# @Organization: YQN

import torch
import numpy as np

def get_attn_pad_mask(seq_q:torch.FloatTensor,seq_k:torch.FloatTensor):
    """

    :param seq_q: [batch_size,seq_q]
    :param seq_k: [batch_size,seq_k]
    :return:
    """
    batch_size,len_q=seq_q.size()
    batch_size,len_k=seq_k.size()

    pad_attn_mask=seq_k.data.eq(0).unsqueeze(1)#[batch_size,1,len_k]
    return pad_attn_mask.expand(batch_size,len_q,len_k)


def get_attn_subsequence_mask(seq):
    attn_shape=[seq.size(0),seq.size(1),seq.size(1)]
    subsequence_mask=np.triu(np.ones(shape=attn_shape),k=1)
    subsequence_mask=torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

if __name__=='__main__':
    a=torch.from_numpy(np.array([[1,0,1],[0,1,0]]))
    subsequence_mask=get_attn_subsequence_mask(a)
    print(subsequence_mask)
    print(subsequence_mask.shape)