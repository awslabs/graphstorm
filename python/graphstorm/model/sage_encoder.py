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
from dgl.distributed.constants import DEFAULT_NTYPE

from .ngnn_mlp import NGNNMLP
from .gnn_encoder_base import GraphConvEncoder


class SAGEConv(nn.Module):
    r"""GraphSage Convolutional layer from `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__.

    The message passing formulas of ``SAGEConv`` are:

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} &= \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} &= \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1}) \right)

        h_{i}^{(l+1)} &= \mathrm{norm}(h_{i}^{(l+1)})

    Note:
    -----
    * ``SAGEConv`` is only effective on homogeneous graphs.

    Examples:
    ----------

    .. code:: python

        # suppose graph and input_feature are ready
        from graphstorm.model import SAGEConv

        layer = SAGEConv(h_dim, h_dim, aggregator_type,
                         bias, activation, dropout,
                         num_ffn_layers_in_gnn, norm)
        h = layer(g, input_feature)

    Parameters
    ----------
    in_feat: int
        Input feature size.
    out_feat: int
        Output feature size.
    aggregator_type: str
        Message aggregation type. Options: ``mean``, ``gcn``, ``pool``, ``lstm``.
        Default: ``mean``.
    bias: bool
        Whether to add bias. Default: True.
    dropout: float
        Dropout rate. Default: 0.
    activation: torch.nn.functional
        Activation function. Default: relu.
    num_ffn_layers_in_gnn: int
        Number of fnn layers between gnn layers. Default: 0.
    ffn_actication: torch.nn.functional
        Activation for ffn. Default: relu.
    norm: str
        Normalization methods. Options:``batch``, ``layer``, and ``None``. Default: None,
        meaning no normalization.
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 aggregator_type='mean',
                 bias=True,
                 dropout=0.0,
                 activation=F.relu,
                 num_ffn_layers_in_gnn=0,
                 ffn_activation=F.relu,
                 norm=None):
        super(SAGEConv, self).__init__()
        self.in_feat, self.out_feat = in_feat, out_feat
        self.aggregator_type = aggregator_type

        self.conv = dglnn.SAGEConv(self.in_feat, self.out_feat, self.aggregator_type,
                                   feat_drop=dropout, bias=bias)

        self.activation = activation
        # normalization
        self.norm = None
        if activation is None and norm is not None:
            raise ValueError("Cannot set gnn norm layer when activation layer is None")
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_feat)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_feat)
        else:
            # by default we don't apply any normalization
            self.norm = None
        # ngnn
        self.num_ffn_layers_in_gnn = num_ffn_layers_in_gnn
        self.ngnn_mlp = NGNNMLP(out_feat, out_feat,
                                 num_ffn_layers_in_gnn, ffn_activation, dropout)

    def forward(self, g, inputs):
        """ GraphSage layer forward computation.

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
        g = g.local_var()

        inputs = inputs[DEFAULT_NTYPE]
        h_conv = self.conv(g, inputs)
        if self.norm:
            h_conv = self.norm(h_conv)
        if self.activation:
            h_conv = self.activation(h_conv)
        if self.num_ffn_layers_in_gnn > 0:
            h_conv = self.ngnn_mlp(h_conv)

        return {DEFAULT_NTYPE: h_conv}


class SAGEEncoder(GraphConvEncoder):
    r""" GraphSage Conv Encoder.

    The ``SAGEEncoder`` employs several ``SAGEConv`` Layers as its encoding mechanism.
    The ``SAGEEncoder`` should be designated as the model's encoder within Graphstorm.

    Parameters
    ----------
    h_dim: int
        Hidden dimension size.
    out_dim: int
        Output dimension size.
    num_hidden_layers: int
        Number of hidden layers. Total GNN layers is equal to ``num_hidden_layers + 1``.
    dropout: float
        Dropout rate. Default: 0.
    aggregator_type: str
        Message aggregation type. Options: ``mean``, ``gcn``, ``pool``, ``lstm``.
    activation: torch.nn.functional
        Activation function. Default: relu.
    num_ffn_layers_in_gnn: int
        Number of fnn layers between GNN layers. Default: 0.
    norm: str
        Normalization methods. Options:``batch``, ``layer``, and ``None``. Default: None,
        meaning no normalization.

    Examples:
    ----------

    .. code:: python

        # Build model and do full-graph inference on SAGEEncoder
        from graphstorm import get_node_feat_size
        from graphstorm.model import SAGEEncoder
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

        gnn_encoder = SAGEEncoder(4, 4,
                                  num_hidden_layers=1,
                                  dropout=0,
                                  aggregator_type="mean",
                                  norm="batch")
        model.set_gnn_encoder(gnn_encoder)
        model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))

        h = do_full_graph_inference(model, np_data)
    """
    def __init__(self,
                 h_dim, out_dim,
                 num_hidden_layers=1,
                 dropout=0,
                 aggregator_type='mean',
                 activation=F.relu,
                 num_ffn_layers_in_gnn=0,
                 norm=None):
        super(SAGEEncoder, self).__init__(h_dim, out_dim, num_hidden_layers)

        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(SAGEConv(h_dim, h_dim, aggregator_type,
                                        bias=False, activation=activation,
                                        dropout=dropout,
                                        num_ffn_layers_in_gnn=num_ffn_layers_in_gnn,
                                        norm=norm))

        self.layers.append(SAGEConv(h_dim, out_dim, aggregator_type,
                                    bias=False, activation=activation,
                                    dropout=dropout,
                                    num_ffn_layers_in_gnn=num_ffn_layers_in_gnn))

    def forward(self, blocks, h):
        """ GraphSage encoder forward computation.

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
