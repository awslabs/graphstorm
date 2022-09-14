"""Node prediction decoders
"""
import torch as th
import torch.nn as nn

class EntityClassifier(nn.Module):
    def __init__(self,
                 h_dim,
                 out_dim,
                 num_hidden_layers=1,
                 dropout=0):
        super(EntityClassifier, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.decoder = nn.Parameter(th.Tensor(h_dim, out_dim))
        nn.init.xavier_uniform_(self.decoder,
                                gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        return th.matmul(h, self.decoder)