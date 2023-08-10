"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Sage layer implementation.
"""
from torch import nn
import torch.nn.functional as F
import dgl.nn as dglnn

from .ngnn_mlp import NGNNMLP
from .gnn_encoder_base import GraphConvEncoder


class SAGEConv(nn.Module):
    r"""Sage Convolutional layer.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    aggregator_type : str
        One of mean, gcn, pool, lstm
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    dropout : float, optional
        Dropout rate. Default: 0.0
    num_ffn_layers_in_gnn: int, optional
        Number of layers of ngnn between gnn layers
    ffn_actication: torch.nn.functional
        Activation Method for ngnn
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 aggregator_type='mean',
                 bias=True,
                 dropout=0.0,
                 activation=F.relu,
                 num_ffn_layers_in_gnn=0,
                 ffn_activation=F.relu):
        super(SAGEConv, self).__init__()
        self.in_feat, self.out_feat = in_feat, out_feat
        self.aggregator_type = aggregator_type

        self.conv = dglnn.SAGEConv(self.in_feat, self.out_feat, self.aggregator_type,
                                   feat_drop=dropout, bias=bias)

        self.activation = activation
        # ngnn
        self.num_ffn_layers_in_gnn = num_ffn_layers_in_gnn
        self.ngnn_mlp = NGNNMLP(out_feat, out_feat,
                                 num_ffn_layers_in_gnn, ffn_activation, dropout)

    def forward(self, g, inputs):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict["_N", torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict{"_N", torch.Tensor}
            New node features for each node type.
        """
        g = g.local_var()

        inputs = inputs['_N']
        h_conv = self.conv(g, inputs)
        if self.activation:
            h_conv = self.activation(h_conv)
        if self.num_ffn_layers_in_gnn > 0:
            h_conv = self.ngnn_mlp(h_conv)

        return {'_N': h_conv}


class SAGEEncoder(GraphConvEncoder):
    r""" Sage Conv Layer

    Parameters
    ----------
    h_dim : int
        Hidden dimension
    out_dim : int
        Output dimension
    num_hidden_layers : int
        Number of hidden layers. Total GNN layers is equal to num_hidden_layers + 1. Default 1
    dropout : float
        Dropout. Default 0.
    num_ffn_layers_in_gnn: int
        Number of ngnn gnn layers between GNN layers
    """
    def __init__(self,
                 h_dim, out_dim,
                 num_hidden_layers=1,
                 dropout=0,
                 aggregator_type='mean',
                 activation=F.relu,
                 num_ffn_layers_in_gnn=0):
        super(SAGEEncoder, self).__init__(h_dim, out_dim, num_hidden_layers)

        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(SAGEConv(h_dim, h_dim, aggregator_type,
                                        bias=False, activation=activation,
                                        dropout=dropout,
                                        num_ffn_layers_in_gnn=num_ffn_layers_in_gnn))

        self.layers.append(SAGEConv(h_dim, out_dim, aggregator_type,
                                    bias=False, activation=activation,
                                    dropout=dropout,
                                    num_ffn_layers_in_gnn=num_ffn_layers_in_gnn))

    def forward(self, blocks, h):
        """Forward computation

        Parameters
        ----------
        blocks: DGL MFGs
            Sampled subgraph in DGL MFG
        h: dict["_N", torch.Tensor]
            Input node feature for each node type.
        """
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h
