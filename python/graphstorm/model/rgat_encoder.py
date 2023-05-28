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

    RGAT layer implementation
"""
import warnings

import torch as th
from torch import nn
import torch.nn.functional as F
import dgl.nn as dglnn

from .gnn_encoder_base import GraphConvEncoder

class RelationalAttLayer(nn.Module):
    r"""Relational graph attention layer.

    For inner relation message aggregation we use multi-head attention network.
    For cross relation message we just use average

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_heads : int
        Number of attention heads
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_heads,
                 *,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelationalAttLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
                rel : dglnn.GATConv(in_feat, out_feat // num_heads, num_heads, bias=False)
                for rel in rel_names
            })

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=invalid-name
    def forward(self, g, inputs):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs_src)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        for k, _ in inputs.items():
            if g.number_of_dst_nodes(k) > 0:
                if k not in hs:
                    warnings.warn("Warning. Graph convolution returned empty "
                          f"dictionary, for node with type: {str(k)}")
                    for _, in_v in inputs_src.items():
                        device = in_v.device
                    hs[k] = th.zeros((g.number_of_dst_nodes(k), self.out_feat), device=device)
                    # TODO the above might fail if the device is a different GPU
                else:
                    hs[k] = hs[k].view(hs[k].shape[0], hs[k].shape[1] * hs[k].shape[2])

        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}

class RelationalGATEncoder(GraphConvEncoder):
    r"""Relational graph attention encoder

    Parameters
    g : DGLHeteroGraph
        Input graph.
    h_dim: int
        Hidden dimension size
    out_dim: int
        Output dimension size
    num_heads: int
        Number of heads
    num_hidden_layers: int
        Num hidden layers
    dropout: float
        Dropout
    use_self_loop: bool
        Self loop
    last_layer_act: bool
        Whether add activation at the last layer
    """
    def __init__(self,
                 g,
                 h_dim, out_dim, num_heads,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=True,
                 last_layer_act=False):
        super(RelationalGATEncoder, self).__init__(h_dim, out_dim, num_hidden_layers)
        self.num_heads = num_heads
        # h2h
        for _ in range(num_hidden_layers):
            self.layers.append(RelationalAttLayer(
                h_dim, h_dim, g.canonical_etypes,
                self.num_heads, activation=F.relu, self_loop=use_self_loop,
                dropout=dropout))
        # h2o
        self.layers.append(RelationalAttLayer(
            h_dim, out_dim, g.canonical_etypes,
            self.num_heads, activation=F.relu if last_layer_act else None,
            self_loop=use_self_loop))

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
            h = layer(block, h)
        return h
