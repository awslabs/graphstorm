"""RGCN layer implementation"""

import torch as th
from torch import nn
import torch.nn.functional as F
import dgl.nn as dglnn

from .rgnn_base import RelGraphConvEncoder

class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
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
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
                for rel in rel_names
            })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

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
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i] : {'weight' : w.squeeze(0)} \
                for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs_src, mod_kwargs=wdict)

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
                    print("Warning. Graph convolution returned empty dictionary, "
                          f"for node with type: {str(k)}")
                    for _, in_v in inputs_src.items():
                        device = in_v.device
                    hs[k] = th.zeros((g.number_of_dst_nodes(k), self.out_feat), device=device)
                    # TODO the above might fail if the device is a different GPU
        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}

class RelationalGCNEncoder(RelGraphConvEncoder):
    r""" Relational graph conv encoder.

    Parameters
    ----------
    g : DGLHeteroGraph
        Input graph.
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
    self_loop_init: bool
        Make the input as identical to it self.
    """
    def __init__(self,
                 g,
                 h_dim, out_dim,
                 num_bases=0,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=True,
                 last_layer_act=False,
                 self_loop_init=False):
        if num_bases < 0 or num_bases > len(g.etypes):
            self.num_bases = len(g.etypes)
        else:
            self.num_bases = num_bases
        self.self_loop_init = self_loop_init
        super(RelationalGCNEncoder, self).__init__(g, h_dim, out_dim,
            num_hidden_layers, dropout, use_self_loop, last_layer_act)

    def init_encoder(self):
        """ Initialize RelationalGCNEncoder encoder
        """
        self.layers = nn.ModuleList()
        # h2h
        for _ in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.rel_names,
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=F.relu if self.last_layer_act else None,
            self_loop=self.use_self_loop))

        if self.self_loop_init:
            for param in self.parameters():
                nn.init.zeros_(param)
            for layer in self.layers:
                nn.init.eye_(layer.loop_weight)

    def forward(self, h=None, blocks=None):
        """Forward computation

        Parameters
        ----------
        h: dict[str, torch.Tensor]
            Input node feature for each node type.
        blocks: DGL MFGs
            Sampled subgraph in DGL MFG
        """
        if blocks is None:
            # full graph training
            for layer in self.layers:
                h = layer(self.g, h)
        else:
            # minibatch training
            for layer, block in zip(self.layers, blocks):
                h = layer(block, h)
        return h
