"""Node prediction decoders
"""
import torch as th
from torch import nn

class EntityClassifier(nn.Module):
    ''' Classifier for node entities.

    Parameters
    ----------
    h_dim : int
        The hidden dimension
    out_dim : int
        The output dimension
    dropout : float
        The dropout
    '''
    def __init__(self,
                 h_dim,
                 out_dim,
                 dropout=0):
        super(EntityClassifier, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.decoder = nn.Parameter(th.Tensor(h_dim, out_dim))
        nn.init.xavier_uniform_(self.decoder,
                                gain=nn.init.calculate_gain('relu'))
        # TODO(zhengda): The dropout is not used here.
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        ''' The forward function.
        '''
        return th.matmul(inputs, self.decoder)

class EntityRegression(nn.Module):
    ''' Regression on entity nodes

    Parameters
    ----------
    h_dim : int
        The hidden dimensions
    dropout : float
        The dropout
    '''
    def __init__(self,
                 h_dim,
                 dropout=0):
        super(Regression, self).__init__()
        self.h_dim = h_dim
        self.decoder = nn.Parameter(th.Tensor(h_dim, 1))
        nn.init.xavier_uniform_(self.decoder)
        # TODO(zhengda): The dropout is not used.
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        ''' The forward function.
        '''
        return th.matmul(inputs, self.decoder)
