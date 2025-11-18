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

import dgl.nn as dglnn
import dgl.function as fn
from dgl.distributed.constants import DEFAULT_NTYPE, DEFAULT_ETYPE
from dgl.utils import expand_as_pair

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


class SAGEConvWithEdgeFeat(nn.Module):
    r"""
    The message passing formulas of ``SAGEConvWithEdgeFeat`` are:

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} &= \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} &= \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1}) \right)

        h_{i}^{(l+1)} &= \mathrm{norm}(h_{i}^{(l+1)})

    Note:
    -----
    * ``SAGEConvWithEdgeFeat`` is only effective on homogeneous graphs.

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
                 feat_drop=0.0,
                 activation=F.relu,
                 num_ffn_layers_in_gnn=0,
                 ffn_activation=F.relu,
                 norm=None):
        super(SAGEConvWithEdgeFeat, self).__init__()
        self.in_feat, self.out_feat = in_feat, out_feat
        self.aggregator_type = aggregator_type

        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feat)
        self._in_edge_feats = self._in_src_feats
        self._out_feats = out_feat
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        self.fc_edge  = nn.Linear(self._in_edge_feats, self._in_edge_feats)
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(
                self._in_src_feats,
                self._in_src_feats
            )
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats + self._in_edge_feats,
                self._in_src_feats + self._in_edge_feats,
                batch_first=True
            )
        self.fc_neigh = nn.Linear(
            self._in_src_feats + self._in_edge_feats,
            self._out_feats, bias=False
            )

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, self._out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(th.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
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

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, g, inputs, edge_weight=None):
        """ GraphSage layer forward computation.

        Parameters
        ----------
        g: DGLHeteroGraph
            Input DGL heterogenous graph.
        inputs: dict of Tensor
            Features for the default node type and edge type in the format of
            {``dgl.DEFAULT_NTYPE``: tensor, ``dgl.DEFAULT_ETYPE``: tensor}. 
            The definition of ``dgl.DEFAULT_NTYPE`` and ``dgl.DEFAULT_ETYPE`` can
            be found at DGL official Github site <https://github.com/dmlc/dgl/blob/
            cb4604aca2e9a79eb61827a71f1f781b70ceac83/python/dgl/distributed/constants.py#L8>`_.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. Not implemented. Reserved for future use.

        Returns
        -------
        dict of Tensor: New node embeddings for the default node type in the format of
        {``dgl.DEFAULT_NTYPE``: tensor}. The definition of ``dgl.DEFAULT_NTYPE`` and
        ``dgl.DEFAULT_ETYPE``can be found at DGL official Github site 
        <https://github.com/dmlc/dgl/blob/cb4604aca2e9a79eb61827a71f1f781b70ceac83/
        python/dgl/distributed/constants.py#L8>`_.
        """
        g = g.local_var()

        node_inputs = inputs[DEFAULT_NTYPE]
        edge_inputs = inputs[DEFAULT_ETYPE]

        g.edata["ft_edge"] = self.fc_edge(edge_inputs)
        with g.local_scope():
            if isinstance(node_inputs, tuple):
                feat_src = self.feat_drop(node_inputs[0])
                feat_dst = self.feat_drop(node_inputs[1])
            else:
                feat_src = feat_dst = self.feat_drop(node_inputs)
                if g.is_block:
                    feat_dst = feat_src[:g.number_of_dst_nodes()]

            h_self = feat_dst

            # Handle the case of graphs without edges
            if g.num_edges() == 0:
                g.dstdata["neigh"] = th.zeros(
                    feat_dst.shape[0], self._in_src_feats + self._in_edge_feats
                ).to(feat_dst)

            # Message Passing
            if self._aggre_type == "mean":
                g.srcdata["h"] = feat_src
                g.apply_edges(lambda edges: {'m': edges.src['h']})
                g.edata["m"] = th.cat([g.edata['m'], edge_inputs], dim=-1)
                # pylint: disable=no-member
                g.update_all(fn.copy_e("m", "m"), fn.mean("m", "neigh"))
                h_neigh = g.dstdata["neigh"]
                h_neigh = self.fc_neigh(h_neigh)

            elif self._aggre_type == "gcn":
                g.srcdata["h"] = feat_src
                if isinstance(node_inputs, tuple):
                    g.dstdata["h"] = feat_dst
                else:
                    if g.is_block:
                        g.dstdata["h"] = g.srcdata["h"][:g.num_dst_nodes()]
                    else:
                        g.dstdata["h"] = g.srcdata["h"]
                g.apply_edges(lambda edges: {'m': edges.src['h']})
                g.edata["m"] = th.cat([g.edata['m'], edge_inputs], dim=-1)
                # pylint: disable=no-member
                g.update_all(fn.copy_e("m", "m"), fn.sum("m", "neigh"))
                # divide in_degrees
                degs = g.in_degrees().to(feat_dst)
                h_neigh = (g.dstdata["neigh"] + th.cat(
                    [g.dstdata["h"],
                    th.zeros((g.dstdata["h"].shape[0], edge_inputs.shape[-1])
                        ).to(g.dstdata["neigh"].device)], dim=-1)) \
                    / degs.unsqueeze(-1) + 1
                h_neigh = self.fc_neigh(h_neigh)

            elif self._aggre_type == "pool":
                g.srcdata["h"] = F.relu(self.fc_pool(feat_src))
                g.apply_edges(lambda edges: {'m': edges.src['h']})
                g.edata["m"] = th.cat([g.edata['m'], edge_inputs], dim=-1)
                # pylint: disable=no-member
                g.update_all(fn.copy_e("m", "m"), fn.max("m", "neigh"))
                h_neigh = self.fc_neigh(g.dstdata["neigh"])

            elif self._aggre_type == "lstm":
                g.srcdata["h"] = feat_src
                g.apply_edges(lambda edges: {'m': edges.src['h']})
                g.edata['m']  = th.cat([g.edata['m'], edge_inputs], dim=-1)
                g.update_all(fn.copy_e("m", "m"), self._lstm_reducer)
                h_neigh = self.fc_neigh(g.dstdata["neigh"])

            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                h_conv = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    h_conv = h_conv + self.bias
            else:
                h_conv = self.fc_self(h_self) + h_neigh

            # activation
            if self.activation is not None:
                h_conv = self.activation(h_conv)

            # normalization
            if self.norm is not None:
                h_conv = self.norm(h_conv)

        if self.norm:
            h_conv = self.norm(h_conv)
        if self.activation:
            h_conv = self.activation(h_conv)
        if self.num_ffn_layers_in_gnn > 0:
            h_conv = self.ngnn_mlp(h_conv)

        return {DEFAULT_NTYPE: h_conv}

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats + self._in_edge_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats + self._in_edge_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}


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
    edge_feat_name: str
        Name of the edge features used.
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
                 edge_feat_name=None,
                 dropout=0,
                 aggregator_type='mean',
                 activation=F.relu,
                 num_ffn_layers_in_gnn=0,
                 norm=None):
        super(SAGEEncoder, self).__init__(h_dim, out_dim, num_hidden_layers)
        # check edge type string format
        if edge_feat_name:
            assert len(edge_feat_name) == 1, 'Single edge type for homogenous graph.'
            etype = list(edge_feat_name.keys())[0]
            assert etype == DEFAULT_ETYPE, \
                f'The edge type should be {DEFAULT_ETYPE} for homogeneous graphs, ' + \
                f'but got \"{etype}\".'
        self.edge_feat_name = edge_feat_name

        self.layers = nn.ModuleList()
        if edge_feat_name is not None:
            for _ in range(num_hidden_layers):
                self.layers.append(SAGEConvWithEdgeFeat(h_dim, h_dim, aggregator_type,
                                            bias=False, activation=activation,
                                            dropout=dropout,
                                            num_ffn_layers_in_gnn=num_ffn_layers_in_gnn,
                                            norm=norm))

            self.layers.append(SAGEConvWithEdgeFeat(h_dim, out_dim, aggregator_type,
                                        bias=False, activation=activation,
                                        dropout=dropout,
                                        num_ffn_layers_in_gnn=num_ffn_layers_in_gnn))
        else:
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

    def forward(self, blocks, h, edge_feats=None):
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
        edge_feats: list of dict of Tensor
            Input edge features for each edge type in the format of [{etype: tensor}, ...],
            or [{}, {}. ...] for zero number of edges in input blocks. The length of edge_feats
            should be equal to the number of gnn layers. Default is None.

        Returns
        -------
        h: dict of Tensor
            New node embeddings for the default node type in the format of
            {``dgl.DEFAULT_NTYPE``: tensor}. The definition of ``dgl.DEFAULT_NTYPE`` can
            be found at `DGL official Github site <https://github.com/dmlc/dgl/blob/
            cb4604aca2e9a79eb61827a71f1f781b70ceac83/python/dgl/distributed/constants.py#L8>`_.
        """
        if self.edge_feat_name is not None:
            assert edge_feats is not None,\
             f"edge features for {DEFAULT_ETYPE} should not be None"
        if edge_feats is not None:
            assert len(edge_feats) == len(blocks), \
                'The layer of edge features should be equal to ' + \
                f'the number of blocks, but got {len(edge_feats)} layers of edge features ' + \
                f'and {len(blocks)} blocks.'
            for layer, block, e_h in zip(self.layers, blocks, edge_feats):
                # Prepare input features for the layer
                layer_input = {DEFAULT_NTYPE: h[DEFAULT_NTYPE]}

                # Add edge features if available and encoder supports them
                if self.edge_feat_name is not None:
                    if isinstance(e_h, dict) and DEFAULT_ETYPE in e_h:
                        layer_input[DEFAULT_ETYPE] = e_h[DEFAULT_ETYPE]
                    else:
                        raise NotImplementedError(
                            f"Edge features must be a dict containing "
                            f"'{DEFAULT_ETYPE}' key, but got {type(e_h)}"
                        )
                # Call the layer with the prepared input
                h = layer(block, layer_input)

        else:
            for layer, block in zip(self.layers, blocks):
                # Prepare input features for the layer (no edge features)
                layer_input = {DEFAULT_NTYPE: h[DEFAULT_NTYPE]}
                # Call the layer with the prepared input
                h = layer(block, layer_input)
        return h

    def is_support_edge_feat(self):
        """ Overwrite ``GraphConvEncoder`` class' method, indicating SAGEEncoder
        supports edge features which is previously obtained by edge_feat_name.
        
        Returns
        -------
        bool
            True indicating that SAGEEncoder supports edge features.
        """
        return self.edge_feat_name is not None
