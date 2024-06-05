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
import logging

import torch as th
from torch import nn
import torch.nn.functional as F
import dgl.nn as dglnn

from .ngnn_mlp import NGNNMLP
from .gnn_encoder_base import (GraphConvEncoder,
                               GSgnnGNNEncoderInterface)


class RelationalAttLayer(nn.Module):
    r"""Relational graph attention layer from `
    Relational Graph Attention Networks <https://arxiv.org/abs/1904.05811>`__.

    For the GATConv on each relation type:
    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score between node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Note:
    -----
    * For inner relation message aggregation we use multi-head attention network.
    * For cross relation message we just use average.

    Examples:
    ----------

    .. code:: python

        # suppose graph and input_feature are ready
        from graphstorm.model.rgat_encoder import RelationalAttLayer

        layer = RelationalAttLayer(
                h_dim, h_dim, g.canonical_etypes,
                num_heads, activation, self_loop,
                dropout, num_ffn_layers_in_gnn,
                fnn_activation, norm)
        h = layer(g, input_feature)

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
    num_ffn_layers_in_gnn: int, optional
        Number of layers of ngnn between gnn layers
    ffn_actication: torch.nn.functional
        Activation Method for ngnn
    norm : str, optional
        Normalization Method. Default: None
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
                 dropout=0.0,
                 num_ffn_layers_in_gnn=0,
                 fnn_activation=F.relu,
                 norm=None):
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

        # get the node types
        ntypes = set()
        for rel in rel_names:
            ntypes.add(rel[0])
            ntypes.add(rel[2])

        # normalization
        self.norm = None
        if activation is None and norm is not None:
            raise ValueError("Cannot set gnn norm layer when activation layer is None")
        if norm == "batch":
            self.norm = nn.ParameterDict({ntype:nn.BatchNorm1d(out_feat) for ntype in ntypes})
        elif norm == "layer":
            self.norm = nn.ParameterDict({ntype:nn.LayerNorm(out_feat) for ntype in ntypes})
        else:
            # by default we don't apply any normalization
            self.norm = None

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        # ngnn
        self.num_ffn_layers_in_gnn = num_ffn_layers_in_gnn
        self.ngnn_mlp = NGNNMLP(out_feat, out_feat,
                                     num_ffn_layers_in_gnn, fnn_activation, dropout)

        # dropout
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
            # handle the case when len(h) is 0
            if h.shape[0] == 0:
                return h.reshape((0, self.out_feat))
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.norm:
                h = self.norm[ntype](h)
            if self.activation:
                h = self.activation(h)
            if self.num_ffn_layers_in_gnn > 0:
                h = self.ngnn_mlp(h)
            return self.dropout(h)

        for k, _ in inputs.items():
            if g.number_of_dst_nodes(k) > 0:
                if k not in hs:
                    logging.warning("Warning. Graph convolution returned empty " + \
                          f"dictionary for nodes in type: {str(k)}. Please check your data" + \
                          f" for no in-degree nodes in type: {str(k)}.")
                    hs[k] = th.zeros((g.number_of_dst_nodes(k),
                                      self.out_feat),
                                     device=inputs[k].device)
                    # TODO the above might fail if the device is a different GPU
                else:
                    hs[k] = hs[k].view(hs[k].shape[0], hs[k].shape[1] * hs[k].shape[2])

        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}

class RelationalGATEncoder(GraphConvEncoder, GSgnnGNNEncoderInterface):
    r"""Relational graph attention encoder

    The RelationalGATEncoder employs several RelationalAttLayers as its encoding mechanism.
    The RelationalGATEncoder should be designated as the model's encoder within Graphstorm.

    Parameters
    -----------
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
    num_ffn_layers_in_gnn: int
        Number of ngnn gnn layers between GNN layers
    norm : str, optional
        Normalization Method. Default: None

    Examples:
    ----------

    .. code:: python

        # Build model and do full-graph inference on RelationalGATEncoder
        from graphstorm import get_node_feat_size
        from graphstorm.model.rgat_encoder import RelationalGATEncoder
        from graphstorm.model.node_decoder import EntityClassifier
        from graphstorm.model import GSgnnNodeModel, GSNodeEncoderInputLayer
        from graphstorm.dataloading import GSgnnData
        from graphstorm.model import do_full_graph_inference

        np_data = GSgnnData(...)

        model = GSgnnNodeModel(alpha_l2norm=0)
        feat_size = get_node_feat_size(np_data.g, 'feat')
        encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                          dropout=0,
                                          use_node_embeddings=True)
        model.set_node_input_encoder(encoder)

        gnn_encoder = RelationalGATEncoder(g, 4, 4,
                                           num_heads=2,
                                           num_hidden_layers=1,
                                           dropout=0,
                                           use_self_loop=True,
                                           norm=norm)
        model.set_gnn_encoder(gnn_encoder)
        model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))

        h = do_full_graph_inference(model, np_data)
    """
    def __init__(self,
                 g,
                 h_dim, out_dim, num_heads,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=True,
                 last_layer_act=False,
                 num_ffn_layers_in_gnn=0,
                 norm=None):
        super(RelationalGATEncoder, self).__init__(h_dim, out_dim, num_hidden_layers)
        self.num_heads = num_heads
        # h2h
        for _ in range(num_hidden_layers):
            self.layers.append(RelationalAttLayer(
                h_dim, h_dim, g.canonical_etypes,
                self.num_heads, activation=F.relu, self_loop=use_self_loop,
                dropout=dropout, num_ffn_layers_in_gnn=num_ffn_layers_in_gnn,
                fnn_activation=F.relu, norm=norm))
        # h2o
        self.layers.append(RelationalAttLayer(
            h_dim, out_dim, g.canonical_etypes,
            self.num_heads, activation=F.relu if last_layer_act else None,
            self_loop=use_self_loop, norm=norm if last_layer_act else None))

    def skip_last_selfloop(self):
        self.last_selfloop = self.layers[-1].self_loop
        self.layers[-1].self_loop = False

    def reset_last_selfloop(self):
        self.layers[-1].self_loop = self.last_selfloop

    def forward(self, blocks, h):
        """Forward computation

        Parameters
        ----------
        blocks: DGL MFGs
            Sampled subgraph in DGL MFG
        h: dict[str, torch.Tensor]
            Input node feature for each node type.

        Returns
        ----------
        h: dict[str, torch.Tensor]
            Output node feature for each node type.

        """
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h
