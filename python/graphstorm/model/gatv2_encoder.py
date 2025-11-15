"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GAT V2 layer implementation.
"""
import torch as th
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from dgl.nn.pytorch.conv.gatv2conv import DGLError
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from dgl.distributed.constants import DEFAULT_NTYPE, DEFAULT_ETYPE

from .ngnn_mlp import NGNNMLP
from .gnn_encoder_base import GraphConvEncoder


class GATv2Conv(nn.Module):
    r""" GATv2 Convolutional layer from `How Attentive are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`__.

    The message passing formulas of ``GATv2Conv`` are:

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{ij}^{(l)} W^{(l)}_{right} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{(l)} &= \mathrm{softmax_i} (e_{ij}^{(l)})

        e_{ij}^{(l)} &= {\vec{a}^T}^{(l)}\mathrm{LeakyReLU}\left(
            W^{(l)}_{left} h_{i} + W^{(l)}_{right} h_{j}\right)

    Note:
    -----
    * ``GATv2Conv`` is only effective on homogeneous graphs.

    Examples:
    ----------

    .. code:: python

        # suppose graph and input_feature are ready
        from graphstorm.model import GATv2Conv

        layer = GATv2Conv(h_dim, h_dim, num_heads, num_ffn_layers_in_gnn)
        h = layer(g, input_feature)

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-head attention.
    activation : callable, optional
        Activation function. Default: relu
    dropout : float, optional
        Dropout rate. Default: 0.0
    bias : bool, optional
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
        super(GATv2Conv, self).__init__()
        self.conv = dglnn.conv.GATv2Conv(in_feat, out_feat // num_heads, num_heads, dropout,
                                  activation=activation, allow_zero_in_degree=True,
                                  bias=bias)

        # ngnn
        self.num_ffn_layers_in_gnn = num_ffn_layers_in_gnn
        self.ngnn_mlp = NGNNMLP(out_feat, out_feat,
                                 num_ffn_layers_in_gnn, ffn_activation, dropout)

    def forward(self, g, inputs):
        """ GATv2 layer Forward computation.

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


class GATv2ConvWithEdgeFeat(nn.Module):
    r""" 
    The formulation of ``GATv2ConvWithEdgeFeat`` are:
    .. math::
        h_i^{(l+1)} = W_s^{(l)} h_i^{(l)} + 
            \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}^{(l)} 
            ( W^{(l)} h_j^{(l)} + W_e^{(l)} e_{i,j} )

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{i,j}^{(l)} = \mathrm{softmax}_j ( \vec{a}^{(l)\,T} \mathrm{LeakyReLU}
            \left( W^{(l)} h_i^{(l)} \| W^{(l)} h_j^{(l)} \| W_e^{(l)} e_{i,j} \right) )
    Note:
    -----
    * ``GATv2ConvWithEdgeFeat`` is only effective on homogeneous graphs with edge features.

    Examples:
    ----------

    .. code:: python

        # suppose graph and input_feature are ready
        from graphstorm.model import GATv2ConvWithEdgeFeat

        layer = GATv2ConvWithEdgeFeat(h_dim, h_dim, num_heads, num_ffn_layers_in_gnn)
        h = layer(g, input_feature)

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-head attention.
    activation : callable, optional
        Activation function. Default: relu
    dropout : float, optional
        Dropout rate. Default: 0.0
    bias : bool, optional
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
                 feat_drop=0.0,
                 attn_drop=0.0,
                 bias=True,
                 num_ffn_layers_in_gnn=0,
                 ffn_activation=F.relu,
                 negative_slope=0.2):
        super(GATv2ConvWithEdgeFeat, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feat)
        self._edge_feats = self._in_src_feats
        self._out_feats = out_feat // num_heads
        self._allow_zero_in_degree = True
        if isinstance(in_feat, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, self._out_feats * self._num_heads, bias=bias
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, self._out_feats * self._num_heads, bias=bias
            )
        else:
            self.fc_src = nn.Linear(
                self._in_src_feats, self._out_feats * self._num_heads, bias=bias
            )
            self.fc_dst = nn.Linear(
                self._in_src_feats, self._out_feats * self._num_heads, bias=bias
            )

        self.attn = nn.Parameter(th.FloatTensor(size=(1, self._num_heads, self._out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if self._in_dst_feats != self._out_feats * self._num_heads:
            self.res_fc = nn.Linear(
                self._in_dst_feats, self._num_heads * self._out_feats, bias=bias
            )
        else:
            self.res_fc = None
        self.activation = activation
        self.bias = bias
        self.fc_edge = nn.Linear(self._edge_feats, self._out_feats * self._num_heads, bias=False)

        self.reset_parameters()
        # ngnn
        self.num_ffn_layers_in_gnn = num_ffn_layers_in_gnn
        self.ngnn_mlp = NGNNMLP(
            self._out_feats * self._num_heads, out_feat,
            num_ffn_layers_in_gnn, ffn_activation, dropout
        )

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_src.bias, 0)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_dst.bias, 0)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.res_fc.bias, 0)

    def forward(self, g, inputs):
        """ GATv2WithEdgeFeat layer Forward computation.

        Parameters
        ----------
        g: DGLHeteroGraph
            Input DGL heterogenous graph.
        inputs: dict of Tensor
            Node features for the default node type in the format of
            {``dgl.DEFAULT_NTYPE``: tensor, ``dgl.DEFAULT_ETYPE``: tensor}. 
            The definition of ``dgl.DEFAULT_NTYPE`` and ``dgl.DEFAULT_ETYPE`` 
            can be found at DGL official Github site <https://github.com/dmlc/dgl/blob/
            cb4604aca2e9a79eb61827a71f1f781b70ceac83/python/dgl/distributed/constants.py#L8>`_.

        Returns
        -------
        dict of Tensor: New node embeddings for the default node type in the format of
            {``dgl.DEFAULT_NTYPE``: tensor}. The definition of ``dgl.DEFAULT_NTYPE``   
            can be found at DGL official Github site <https://github.com/dmlc/dgl/blob/
            cb4604aca2e9a79eb61827a71f1f781b70ceac83/python/dgl/distributed/constants.py#L8>`_.
        """
        # add self-loop during computation.
        assert DEFAULT_NTYPE in inputs and DEFAULT_ETYPE in inputs, \
            "Both node and edge features are needed to for GATConvwithEdgeFeat."

        node_inputs = inputs[DEFAULT_NTYPE]
        edge_inputs = inputs[DEFAULT_ETYPE]

        feat = node_inputs
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                feat_dst = self.fc_dst(h_dst).view(
                    -1, self._num_heads, self._out_feats
                )
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                feat_dst = self.fc_dst(h_dst).view(
                    -1, self._num_heads, self._out_feats
                )
                if g.is_block:
                    feat_dst = feat_dst[: g.number_of_dst_nodes()]
                    h_dst = h_dst[: g.number_of_dst_nodes()]

            # (num_edges, num_heads, out_dim)
            feat_edge = self.fc_edge(edge_inputs).view(
                -1, self._num_heads, self._out_feats
            )
            g.srcdata.update({"ft_src": feat_src})
            g.dstdata.update({"ft_dst": feat_dst})
            g.edata.update({"ft_edge": feat_edge})

            # Compute attention scores using message function
            g.apply_edges(
                lambda edges: {
                    'ft_tmp': edges.src["ft_src"] + \
                    edges.dst["ft_dst"] + edges.data["ft_edge"]
                }
            )

            # (num_edges, num_heads, out_dim)
            e = self.leaky_relu(g.edata["ft_tmp"])
            # (num_edges, num_heads)
            e = (e * self.attn).sum(dim=-1).unsqueeze(-1)
            # compute softmax
            g.edata["a"] = self.attn_drop(edge_softmax(g, e))

            # Create new edges features that combine the
            # features of the source node and the edge features.
            g.srcdata.update({"ft": feat_src})
            g.apply_edges(
                lambda edges: {'ft_combined': edges.src["ft"] + edges.data["ft_edge"]}
            )
            # the attention coefficient.
            g.edata["m_combined"] = g.edata["ft_combined"] * g.edata["a"]
            # pylint: disable=no-member
            g.update_all(fn.copy_e("m_combined", "m"), fn.sum("m", "ft"))
            h_conv = g.dstdata["ft"]

            # residual
            if self.res_fc is not None:
                if h_dst.numel() != 0:
                    resval = self.res_fc(h_dst).view(
                        h_dst.shape[0], -1, self._out_feats
                    )
                    h_conv = h_conv + resval

            # activation
            if self.activation:
                h_conv = self.activation(h_conv)

        h_conv = h_conv.view(h_conv.shape[0], h_conv.shape[1] * h_conv.shape[2])
        if self.num_ffn_layers_in_gnn > 0:
            h_conv = self.ngnn_mlp(h_conv)
        return {DEFAULT_NTYPE: h_conv}


class GATv2Encoder(GraphConvEncoder):
    r""" GATv2 Conv Encoder.

    The ``GATv2Encoder`` employs several ``GATv2Conv`` Layers as its encoding mechanism.
    The ``GATv2Encoder`` should be designated as the model's encoder within Graphstorm.

    Examples:
    ----------

    .. code:: python

        # Build model and do full-graph inference on GATv2Encoder
        from graphstorm import get_node_feat_size
        from graphstorm.model import GATv2Encoder
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

        gnn_encoder = GATv2Encoder(4, 4, num_heads=2
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
    edge_feat_name: str
        Name of the edge features used.
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
                 h_dim,
                 out_dim,
                 num_heads,
                 num_hidden_layers=1,
                 edge_feat_name=None,
                 dropout=0,
                 activation=F.relu,
                 last_layer_act=False,
                 num_ffn_layers_in_gnn=0):
        super(GATv2Encoder, self).__init__(h_dim, out_dim, num_hidden_layers)
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
                self.layers.append(GATv2ConvWithEdgeFeat(
                    h_dim, h_dim, num_heads,
                    activation=activation,
                    dropout=dropout, bias=True,
                    num_ffn_layers_in_gnn=num_ffn_layers_in_gnn)
                )

            self.layers.append(GATv2ConvWithEdgeFeat(
                h_dim, out_dim, num_heads,
                activation=activation if last_layer_act else None,
                dropout=dropout, bias=True)
            )
        else:
            for _ in range(num_hidden_layers):
                self.layers.append(GATv2Conv(
                    h_dim, h_dim, num_heads,
                    activation=activation,
                    dropout=dropout, bias=True,
                    num_ffn_layers_in_gnn=num_ffn_layers_in_gnn)
                )

            self.layers.append(GATv2Conv(
                h_dim, out_dim, num_heads,
                activation=activation if last_layer_act else None,
                dropout=dropout, bias=True)
            )

    def forward(self, blocks, h, edge_feats=None):
        """ GATv2 encoder forward computation.

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
                h = layer(block, h)
        return h

    def is_support_edge_feat(self):
        """ Overwrite ``GraphConvEncoder`` class' method, indicating GATv2Encoder
        supports edge features which is previously obtained by edge_feat_name.
        
        Returns
        -------
        bool
            True indicating that GATv2Encoder supports edge features.
        """
        return self.edge_feat_name is not None
