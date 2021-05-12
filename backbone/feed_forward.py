# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2021/5/12 17:50
# @Organization: YQN
import torch.nn as nn


class PoswiseFeedForWardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForWardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        :param inputs: [batch_size,seq_len,d_model]
        :return:
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size,seq_len,d_model]
