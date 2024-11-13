"""_LSTM class based on torch_
:param param1: a dict content the parameters of MLP architecture and the training parameter

Param={
    'inputsize':M,
    'outputsize':N,
    'LSTM_hidden':L,
    'num_layer':n,
    'learning_rate':0.001,
    'opt':'Adam',
}

:returns: MLP model
:raises keyError: raises an exception
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.param = params
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=self.param['embedding_dim'],
                               hidden_size=self.param['num_hiddens'],
                               num_layers=self.param['num_layers'],
                               batch_first=True,
                               bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(
            self.param['num_hiddens'] * 2, self.param['num_hiddens'] * 2))
        self.u_omega = nn.Parameter(torch.Tensor(self.param['num_hiddens'] * 2, 1))
        self.decoder = nn.Linear(2*self.param['num_hiddens'], self.param['output_dim'])

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        outputs, _ = self.encoder(inputs)  # output, (h, c)
        x = outputs
        # x形状是(batch_size, seq_len, 2 * num_hiddens)
        
        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score
        
        feat = torch.sum(scored_x, dim=1) #加权求和
        outs = self.decoder(feat)
        return outs
        