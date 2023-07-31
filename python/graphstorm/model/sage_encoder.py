import warnings

import torch as th
from torch import nn
import torch.nn.functional as F
import dgl.nn as dglnn

from .ngnn_mlp import NGNNMLP
from .gnn_encoder_base import GraphConvEncoder


class SAGEConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 aggregator_type='mean',
                 bias=True,
                 dropout=0.0,
                 activation=F.relu,
                 num_ffn_layers_in_gnn=0):
        self.in_feat, self.out_feat = in_feat, out_feat
        self.aggregator_type = aggregator_type

        self.conv = dglnn.SAGEConv(self.in_feat, self.out_feat, self.aggregator_type, bias=bias)

        self.activation = activation
        # ngnn
        self.num_ffn_layers_in_gnn = num_ffn_layers_in_gnn
        self.ngnn_mlp = NGNNMLP(out_feat, out_feat,
                                     num_ffn_layers_in_gnn, ffn_activation, dropout)

    def forward(self, g, inputs):
        g = g.local_var()

        print("information inside: ")
        print(inputs)
        h_conv = self.conv(g, inputs)
        if self.activation:
            h_conv = self.activation(h_conv)

        return h_conv


class SAGEEncoder(GraphConvEncoder):
    r""" Relational graph conv encoder.

    Parameters
    ----------
    h_dim : int
        Hidden dimension
    out_dim : int
        Output dimension
    num_bases: int
        Number of bases.
    num_hidden_layers : int
        Number of hidden layers. Total GNN layers is equal to num_hidden_layers + 1. Default 1
    dropout : float
        Dropout. Default 0.
    use_self_loop : bool
        Whether to add selfloop. Default True
    last_layer_act : torch.function
        Activation for the last layer. Default None
    num_ffn_layers_in_gnn: int
        Number of ngnn gnn layers between GNN layers
    """
    def __init__(self,
                 g,
                 h_dim, out_dim,
                 num_hidden_layers=1,
                 dropout=0,
                 aggregator_type='mean',
                 activation='F.relu',
                 num_ffn_layers_in_gnn=0):
        super(SAGEEncoder, self).__init__(h_dim, out_dim, num_hidden_layers)
        self.aggregator_type = aggregator_type
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(SAGEConv(h_dim, h_dim, self.aggregator_type,
                                        bias=False, activation=activation,
                                        num_ffn_layers_in_gnn=num_ffn_layers_in_gnn))
            # self.layers.append(dglnn.SAGEConv(h_dim, h_dim, self.aggregator_type, bias=False))

        self.layers.append(SAGEConv(h_dim, out_dim, self.aggregator_type,
                                    bias=False, activation=activation,
                                    num_ffn_layers_in_gnn=num_ffn_layers_in_gnn))
        # self.layers.append(dglnn.SAGEConv(h_dim, out_dim, self.aggregator_type, bias=False))

    # TODO(zhengda) refactor this to support edge features.
    def forward(self, blocks, h):
        """Forward computation

        Parameters
        ----------
        blocks: DGL MFGs
            Sampled subgraph in DGL MFG
        h: dict[str, torch.Tensor]
            Input node feature for each node type.
        """
        for layer, block in zip(self.layers, blocks):
            print('information outside: ')
            print(block)
            print(h)
            h = layer(block, h)
        return h