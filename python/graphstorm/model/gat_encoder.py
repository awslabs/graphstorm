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
import torch as th
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.distributed.constants import DEFAULT_NTYPE

from .ngnn_mlp import NGNNMLP
from .gnn_encoder_base import GraphConvEncoder


class GATConv(nn.Module):
    r""" Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__.

    The message passing formulas of ``GATConv`` are:

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Note:
    -----
    * ``GATEConv`` is only effective on homogeneous graphs.

    Examples:
    ----------

    .. code:: python

        # suppose graph and input_feature are ready
        from graphstorm.model import GATConv

        layer = GATConv(h_dim, h_dim, num_heads, num_ffn_layers_in_gnn)
        h = layer(g, input_feature)

    Parameters
    ----------
    in_feat: int
        Input feature size.
    out_feat: int
        Output feature size.
    num_heads: int
        Number of heads in Multi-head attention.
    activation: callable, optional
        Activation function. Default: relu
    dropout: float, optional
        Dropout rate. Default: 0.0
    bias: bool, optional
        True if bias is added. Default: True
    num_ffn_layers_in_gnn: int, optional
        Number of layers of ngnn between gnn layers. Default: 0
    ffn_actication: torch.nn.functional, optional
        Activation Method for ngnn. Default: relu
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_heads,
                 activation=F.relu,
                 dropout=0.0,
                 bias=True,
                 num_ffn_layers_in_gnn=0,
                 ffn_activation=F.relu):
        super(GATConv, self).__init__()
        self.conv = dglnn.GATConv(in_feat, out_feat // num_heads, num_heads, dropout,
                                  activation=activation, allow_zero_in_degree=True,
                                  bias=bias)

        # ngnn
        self.num_ffn_layers_in_gnn = num_ffn_layers_in_gnn
        self.ngnn_mlp = NGNNMLP(out_feat, out_feat,
                                 num_ffn_layers_in_gnn, ffn_activation, dropout)

    def forward(self, g, inputs):
        """ GAT layer forward computation.

        Parameters
        ----------
        g: DGLHeteroGraph
            Input DGL heterogenous graph.
        inputs: dict of Tensor
            Node features for the default node type in the format of
            {``dgl.DEFAULT_NTYPE``: tensor}. The definition of ``dgl.DEFAULT_NTYPE`` can
            be found at `DGL official Github site <https://github.com/dmlc/dgl/blob/
            cb4604aca2e9a79eb61827a71f1f781b70ceac83/python/dgl/distributed/constants.py#L8>`_.

        Returns
        -------
        dict of Tensor: New node embeddings for the default node type in the format of
        {``dgl.DEFAULT_NTYPE``: tensor}. The definition of ``dgl.DEFAULT_NTYPE`` can
        be found at `DGL official Github site <https://github.com/dmlc/dgl/blob/
        cb4604aca2e9a79eb61827a71f1f781b70ceac83/python/dgl/distributed/constants.py#L8>`_.
        """
        # add self-loop during computation.
        src, dst = g.edges()
        src = th.cat([src, th.arange(g.num_dst_nodes(), device=g.device)], dim=0)
        dst = th.cat([dst, th.arange(g.num_dst_nodes(), device=g.device)], dim=0)
        new_g= dgl.create_block(
            (src, dst),
            num_src_nodes=g.num_src_nodes(),
            num_dst_nodes=g.num_dst_nodes(),
            device=g.device
        )

        new_g.nodes[DEFAULT_NTYPE].data[dgl.NID] = g.nodes[DEFAULT_NTYPE].data[dgl.NID]
        g = g.local_var()

        assert DEFAULT_NTYPE in inputs, "GAT encoder only support homogeneous graph."
        inputs = inputs[DEFAULT_NTYPE]

        h_conv = self.conv(g, inputs)
        h_conv = h_conv.view(h_conv.shape[0], h_conv.shape[1] * h_conv.shape[2])

        if self.num_ffn_layers_in_gnn > 0:
            h_conv = self.ngnn_mlp(h_conv)

        return {DEFAULT_NTYPE: h_conv}

class GATEncoder(GraphConvEncoder):
    r""" GAT Conv Encoder.

    The ``GATEncoder`` employs several ``GATConv`` Layers as its encoding mechanism.
    The ``GATEncoder`` should be designated as the model's encoder within Graphstorm.

    Examples:
    ----------

    .. code:: python

        # Build model and do full-graph inference on GATEncoder
        from graphstorm import get_node_feat_size
        from graphstorm.model import GATEncoder
        from graphstorm.model import EntityClassifier
        from graphstorm.model import GSgnnNodeModel, GSNodeEncoderInputLayer
        from graphstorm.dataloading import GSgnnData
        from graphstorm.model import do_full_graph_inference

        np_data = GSgnnData(...)

        model = GSgnnNodeModel(alpha_l2norm=0)
        feat_size = get_node_feat_size(np_data.g, "feat")
        encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                          dropout=0,
                                          use_node_embeddings=True)
        model.set_node_input_encoder(encoder)

        gnn_encoder = GATEncoder(4, 4, num_heads=2
                                 num_hidden_layers=1)
        model.set_gnn_encoder(gnn_encoder)
        model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))

        h = do_full_graph_inference(model, np_data)

    Parameters
    ----------
    h_dim: int
        Hidden dimension size.
    out_dim: int
        Output dimension size.
    num_heads: int
        Number of multi-heads attention heads.
    num_hidden_layers: int
        Number of hidden layers. Total GNN layers is equal to ``num_hidden_layers + 1``.
    dropout: float
        Dropout rate. Default: 0.
    activation: torch.nn.functional
        Activation function. Default: relu.
    last_layer_act: bool
        Whether to call activation function in the last GNN layer. Default: False.
    num_ffn_layers_in_gnn: int
        Number of fnn layers between GNN layers. Default: 0.
    """
    def __init__(self,
                 h_dim, out_dim,
                 num_heads,
                 num_hidden_layers=1,
                 dropout=0,
                 activation=F.relu,
                 last_layer_act=False,
                 num_ffn_layers_in_gnn=0):
        super(GATEncoder, self).__init__(h_dim, out_dim, num_hidden_layers)

        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(GATConv(h_dim, h_dim, num_heads,
                                        activation=activation,
                                        dropout=dropout, bias=True,
                                        num_ffn_layers_in_gnn=num_ffn_layers_in_gnn))

        self.layers.append(GATConv(h_dim, out_dim, num_heads,
                                    activation=activation if last_layer_act else None,
                                    dropout=dropout, bias=True))

    def forward(self, blocks, h):
        """ GAT encoder forward computation.

        Parameters
        ----------
        blocks: list of DGL MFGs
            Sampled subgraph in the list of DGL message flow graphs (MFGs) format. More
            detailed information about DGL MFG can be found in `DGL Neighbor Sampling
            Overview
            <https://docs.dgl.ai/stochastic_training/neighbor_sampling_overview.html>`_.
        h: dict of Tensor
            Node features for the default node type in the format of
            {``dgl.DEFAULT_NTYPE``: tensor}. The definition of ``dgl.DEFAULT_NTYPE`` can
            be found at `DGL official Github site <https://github.com/dmlc/dgl/blob/
            cb4604aca2e9a79eb61827a71f1f781b70ceac83/python/dgl/distributed/constants.py#L8>`_.

        Returns
        -------
        h: dict of Tensor
            New node embeddings for the default node type in the format of
            {``dgl.DEFAULT_NTYPE``: tensor}. The definition of ``dgl.DEFAULT_NTYPE`` can
            be found at `DGL official Github site <https://github.com/dmlc/dgl/blob/
            cb4604aca2e9a79eb61827a71f1f781b70ceac83/python/dgl/distributed/constants.py#L8>`_.
        """
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h
