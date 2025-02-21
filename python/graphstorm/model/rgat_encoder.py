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

import dgl.function as fn
from dgl.nn.pytorch.hetero import get_aggregate_fn
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from .ngnn_mlp import NGNNMLP
from .gnn_encoder_base import (GraphConvEncoder,
                               GSgnnGNNEncoderInterface)
from ..config import BUILTIN_EDGE_FEAT_MP_OPS


class RelationalAttLayer(nn.Module):
    r"""Relational graph attention layer from `Relational Graph
    Attention Networks <https://arxiv.org/abs/1904.05811>`__.

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
        from graphstorm.model import RelationalAttLayer

        layer = RelationalAttLayer(
                in_feat=h_dim, out_feat=h_dim, rel_names=g.canonical_etypes,
                num_heads=4, self_loop,
                dropout, num_ffn_layers_in_gnn,
                fnn_activation, norm)
        h = layer(g, input_feature)

    Parameters
    ----------
    in_feat: int
        Input feature size.
    out_feat: int
        Output feature size.
    rel_names: list of tuple
        Relation type list in the format of [('src_ntyp1', 'etype1', 'dst_ntype1'), ...].
    num_heads: int
        Number of attention heads.
    edge_feat_name: dict of list of str
        User provided edge feature names in the format of {etype1:[feat1, feat2, ...],
        etype2:[...], ...}, or None if not provided.
    edge_feat_mp_op: str
        The opration method to combine source node embeddings with edge embeddings in message
        passing. Options include ``concat``, ``add``, ``sub``, ``mul``, and ``div``.
        ``concat`` operation will concatenate the source node features with edge features;
        ``add`` operation will add the source node features with edge features together;
        ``sub`` operation will subtract the source node features by edge features;
        ``mul`` operation will multiply the source node features with edge features; and
        ``div`` operation will divide the source node features by edge features.
    bias: bool
        Whether to add bias. Default: True.
    activation: callable
        Activation function. Default: None.
    self_loop: bool
        Whether to include self loop message. Default: False.
    dropout: float
        Dropout rate. Default: 0.
    num_ffn_layers_in_gnn: int
        Number of fnn layers between gnn layers. Default: 0.
    ffn_activation: torch.nn.functional
        Activation for ffn. Default: relu.
    norm: str
        Normalization methods. Options:``batch``, ``layer``, and ``None``. Default: None,
        meaning no normalization.
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_heads,
                 *,
                 edge_feat_name=None,
                 edge_feat_mp_op='concat',
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
        self.num_heads = num_heads
        self.rel_names = rel_names
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if edge_feat_name:
            assert len(set(edge_feat_name.keys()).intersection(
                set(rel_names))) > 0, (f'To use GATConvwithEdgeFeat, must provide valid '
                                       f'edge feature name, but got {edge_feat_name}.')
            # warning
            if len(set(edge_feat_name.keys()) - set(rel_names)) > 0:
                warn_msg = (f"Warning. Not using edge features for the invalid edge_feat_name: "
                            f"{set(edge_feat_name.keys()) - set(rel_names)}.")
                self.warning_once(warn_msg)
        self.edge_feat_name = edge_feat_name
        self.edge_feat_mp_op = edge_feat_mp_op

        rel_convs = {}
        for rel in rel_names:
            if edge_feat_name and rel in edge_feat_name:
                rel_convs[rel] = GATConvwithEdgeFeat(in_feat, out_feat // num_heads, num_heads,
                                                       edge_feat_mp_op=edge_feat_mp_op,
                                                       bias=False)
            else:
                rel_convs[rel] = dglnn.GATConv(in_feat, out_feat // num_heads, num_heads,
                                               bias=False)
        self.conv = HeteroGraphConv(rel_convs)

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
        self.warn_msg = set()

    def warning_once(self, warn_msg):
        """ Print same warning msg only once

        Parameters
        ----------
        warn_msg: str
            Warning message
        """
        if warn_msg in self.warn_msg:
            # Skip printing warning
            return
        self.warn_msg.add(warn_msg)
        logging.warning(warn_msg)

    # pylint: disable=invalid-name
    def forward(self, g, n_h, e_h=None):
        """ RGAT layer forward computation.

        Parameters
        ----------
        g: DGLHeteroGraph
            Input DGL heterogenous graph.
        n_h: dict of Tensor
            Node features for each node type in the format of {ntype: tensor}.
        e_h: dict of Tensor
            edge features for each edge type in the format of {etype: tensor}. Default is None.

        Returns
        -------
        dict of Tensor: New node embeddings for each node type in the format of {ntype: tensor}.

        .. versionchanged:: 0.4.1
            Change inputs into n_h and e_h in v0.4.1 to support edge feature
            in RGAT layer.
        """
        g = g.local_var()

        if g.is_block:
            inputs_src = n_h
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in n_h.items()}
        else:
            inputs_src = inputs_dst = n_h

        if e_h is None:
            hs = self.conv(g, (inputs_src, inputs_dst, {}))
        else:
            assert len(e_h) == 0 or self.edge_feat_name is not None, \
                "Since you want to use edge features {list(e_h.keys())} in " + \
                 "message passing computation, please initialize the RelGraphConvLayer " + \
                 "by setting the \"edge_feat_name\" argument."
            hs = self.conv(g, (inputs_src, inputs_dst, e_h))


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

        for k, _ in n_h.items():
            if g.number_of_dst_nodes(k) > 0:
                if k not in hs:
                    warn_msg = "Warning. Graph convolution returned empty " \
                        f"dictionary for nodes in type: {str(k)}. Please check your data" \
                        f" for no in-degree nodes in type: {str(k)}."
                    self.warning_once(warn_msg)
                    hs[k] = th.zeros((g.number_of_dst_nodes(k),
                                      self.out_feat),
                                     device=n_h[k].device)
                    # TODO the above might fail if the device is a different GPU
                else:
                    hs[k] = hs[k].view(hs[k].shape[0], hs[k].shape[1] * hs[k].shape[2])

        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}

class RelationalGATEncoder(GraphConvEncoder, GSgnnGNNEncoderInterface):
    """ Relational graph attention encoder.

    The ``RelationalGATEncoder`` employs several ``RelationalAttLayer`` as its encoding
    mechanism. The ``RelationalGATEncoder`` should be designated as the model's encoder
    within Graphstorm.

    .. versionchanged:: 0.4.1
        Add two new arguments ``edge_feat_name`` and ``edge_feat_mp_op`` in v0.4.1 to
        support edge features in RGAT encoder.


    Parameters
    -----------
    g: DistGraph
        The distributed graph.
    h_dim: int
        Hidden dimension.
    out_dim: int
        Output dimension.
    num_heads: int
        Number of attention heads.
    num_hidden_layers: int
        Number of hidden layers. Total GNN layers is equal to ``num_hidden_layers + 1``.
        Default: 1.
    edge_feat_name: dict of list of str
        User provided edge feature names in the format of {etype1:[feat1, feat2, ...],
        etype2:[...], ...}, or None if not provided.
    edge_feat_mp_op: str
        The opration method to combine source node embeddings with edge embeddings in message
        passing. Options include `concat`, `add`, `sub`, `mul`, and `div`.
        ``concat`` operation will concatenate the source node features with edge features;
        ``add`` operation will add the source node features with edge features together;
        ``sub`` operation will subtract the source node features by edge features;
        ``mul`` operation will multiply the source node features with edge features; and
        ``div`` operation will divide the source node features by edge features.
    dropout: float
        Dropout rate. Default: 0.
    use_self_loop: bool
        Whether to add selfloop. Default: True.
    last_layer_act: callable
        Activation for the last layer. Default: None.
    num_ffn_layers_in_gnn: int
        Number of fnn layers between GNN layers. Default: 0.
    norm: str
        Normalization methods. Options:``batch``, ``layer``, and ``None``. Default: None,
        meaning no normalization.

    Examples:
    ----------

    .. code:: python

        # Build model and do full-graph inference on RelationalGATEncoder
        from graphstorm import get_node_feat_size
        from graphstorm.model import RelationalGATEncoder
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

        gnn_encoder = RelationalGATEncoder(g, 4, 4,
                                           num_heads=2,
                                           num_hidden_layers=1,
                                           dropout=0,
                                           use_self_loop=True,
                                           norm="batch")
        model.set_gnn_encoder(gnn_encoder)
        model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))

        h = do_full_graph_inference(model, np_data)

    .. warning::
        To use edge feature in message passing computation, please ensure the node and edge
        features have the same dimension. Users can use GraphStorm's ``GSNodeEncoderInputLayer``,
        and ``GSEdgeEncoderInputLayer`` to transfer node and edge feature dimensions.

    """
    def __init__(self,
                 g,
                 h_dim, out_dim, num_heads,
                 num_hidden_layers=1,
                 edge_feat_name=None,
                 edge_feat_mp_op='concat',
                 dropout=0,
                 use_self_loop=True,
                 last_layer_act=False,
                 num_ffn_layers_in_gnn=0,
                 norm=None):
        super(RelationalGATEncoder, self).__init__(h_dim, out_dim, num_hidden_layers,
                                                   edge_feat_name, edge_feat_mp_op)
        self.num_heads = num_heads
        # h2h
        for _ in range(num_hidden_layers):
            self.layers.append(RelationalAttLayer(
                h_dim, h_dim, g.canonical_etypes, self.num_heads,
                edge_feat_name=edge_feat_name, edge_feat_mp_op=edge_feat_mp_op,
                activation=F.relu, self_loop=use_self_loop,
                dropout=dropout, num_ffn_layers_in_gnn=num_ffn_layers_in_gnn,
                fnn_activation=F.relu, norm=norm))
        # h2o
        self.layers.append(RelationalAttLayer(
            h_dim, out_dim, g.canonical_etypes, self.num_heads,
            edge_feat_name=edge_feat_name, edge_feat_mp_op=edge_feat_mp_op,
            activation=F.relu if last_layer_act else None,
            self_loop=use_self_loop, norm=norm if last_layer_act else None))

    def is_support_edge_feat(self):
        """ Overwrite ``RelationalAttLayer`` class' method, indicating RelationalGATEncoder
       supports edge feature.
        """
        return True

    def skip_last_selfloop(self):
        self.last_selfloop = self.layers[-1].self_loop
        self.layers[-1].self_loop = False

    def reset_last_selfloop(self):
        self.layers[-1].self_loop = self.last_selfloop

    def forward(self, blocks, n_h, e_hs=None):
        """ RGAT encoder forward computation.

        .. versionchanged:: 0.4.1
            Change inputs into blocks, n_h and e_hs in v0.4.1 to support edge feature
            in RGAT encoder.

        Parameters
        ----------
        blocks: list of DGL MFGs
            Sampled subgraph in the list of DGL message flow graphs (MFGs) format. More
            detailed information about DGL MFG can be found in `DGL Neighbor Sampling
            Overview
            <https://docs.dgl.ai/stochastic_training/neighbor_sampling_overview.html>`_.
        n_h: dict of Tensor
            Input node features for each node type in the format of {ntype: tensor}.
        e_hs: list of dict of Tensor
            Input edge features for each edge type in the format of [{etype: tensor}, ...],
            or [{}, {}. ...] for zero number of edges in input blocks. The length of e_hs
            should be equal to the number of gnn layers. Default is None.

        Returns
        ----------
        h: dict of Tensor
            Output node embeddings for each node type in the format of {ntype: tensor}.
        """
        if e_hs is not None:
            assert len(e_hs) == len(blocks), 'The size of the list of edge features should ' + \
                f'be equal to the number of blocks, but got {len(e_hs)} layers of edge ' + \
                f'features and {len(blocks)} blocks.'

            for layer, block, e_h in zip(self.layers, blocks, e_hs):
                n_h = layer(block, n_h, e_h)
        else:
            for layer, block in zip(self.layers, blocks):
                n_h = layer(block, n_h)

        return n_h

class HeteroGraphConv(nn.Module):
    r"""A generic module for computing convolution on heterogeneous graphs.

    Parameters
    ----------
    mods: dict[str, nn.Module]
        Modules associated with every edge types. The forward function of each
        module must have a `DGLGraph` object as the first argument, and
        its second argument is either a tensor object representing the node
        features or a pair of tensor object representing the source and destination
        node features.
    aggregate: str, callable, optional
        Method for aggregating node features generated by different relations.
        Allowed string values are 'sum', 'max', 'min', 'mean', 'stack'.
        The 'stack' aggregation is performed along the second dimension, whose order
        is deterministic.
        User can also customize the aggregator by providing a callable instance.
        For example, aggregation by summation is equivalent to the follows:

    Attributes
    ----------
    mods: dict[str, nn.Module]
        Modules associated with every edge types.
    """

    def __init__(self, mods, aggregate="sum"):
        super(HeteroGraphConv, self).__init__()
        self.mod_dict = mods
        mods = {str(k): v for k, v in mods.items()}
        # Register as child modules
        self.mods = nn.ModuleDict(mods)
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(
                v, "set_allow_zero_in_degree", None
            )
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def _get_module(self, etype):
        mod = self.mod_dict.get(etype, None)
        if mod is not None:
            return mod
        if isinstance(etype, tuple):
            # etype is canonical
            _, etype, _ = etype
            return self.mod_dict[etype]
        raise KeyError("Cannot find module with edge type %s" % etype)

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation

        Invoke the forward function with each module and aggregate their results.

        .. versionchanged:: 0.4.1
            Modify the argument inputs to accept a tuple of dict[str, Tensor] to support
            edge features in graph convolution.

        Parameters
        ----------
        g: DGLGraph
            Graph data.
        inputs: dict[str, Tensor] or tuple of dict[str, Tensor]
            Input node features, and edge feature if provided.
        mod_args: dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs: dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.

        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple) and len(inputs)==3:
                src_inputs, dst_inputs, edge_inputs = inputs
            elif isinstance(inputs, tuple) and len(inputs)==2:
                src_inputs, dst_inputs = inputs
                edge_inputs = {}
            else:
                src_inputs = inputs
                dst_inputs = {
                    k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
                }
                edge_inputs = {}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                # check if the edge type has inputs
                if (stype, etype, dtype) in edge_inputs:
                    dstdata = self._get_module((stype, etype, dtype))(
                        rel_graph,
                        (src_inputs[stype], dst_inputs[dtype], edge_inputs[(stype, etype, dtype)]),
                        *mod_args.get((stype, etype, dtype), ()),
                        **mod_kwargs.get((stype, etype, dtype), {})
                    )
                else:
                    dstdata = self._get_module((stype, etype, dtype))(
                        rel_graph,
                        (src_inputs[stype], dst_inputs[dtype]),
                        *mod_args.get((stype, etype, dtype), ()),
                        **mod_kwargs.get((stype, etype, dtype), {})
                    )
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata = self._get_module((stype, etype, dtype))(
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get((stype, etype, dtype), ()),
                    **mod_kwargs.get((stype, etype, dtype), {})
                )
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts

class GATConvwithEdgeFeat(nn.Module):
    """ Graph attention layer with edge feature supported in message passing computation.

    .. versionadded:: 0.4.1
        Add `GATConvwithEdgeFeat` class in v0.4.1 to support edge feature
        in message passing computation.

    Parameters
    ----------
    in_feat: int
        Input feature size.
    out_feat: int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    edge_feat_mp_op: str
        The opration method to combine source node embeddings with edge embeddings in message
        passing. Options include `concat`, `add`, `sub`, `mul`, and `div`.
        ``concat`` operation will concatenate the source node features with edge features;
        ``add`` operation will add the source node features with edge features together;
        ``sub`` operation will subtract the source node features by edge features;
        ``mul`` operation will multiply the source node features with edge features; and
        ``div`` operation will divide the source node features by edge features.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    bias: bool
        Whether to add bias. Default: True.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        edge_feat_mp_op='concat',
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        self_loop=True,
        activation=None,
        bias=True,
    ):
        super(GATConvwithEdgeFeat, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        assert edge_feat_mp_op in BUILTIN_EDGE_FEAT_MP_OPS, (
                'GraphStorm only support edge' + \
                f' message passing operation in {BUILTIN_EDGE_FEAT_MP_OPS}, bug got ' + \
                f'{edge_feat_mp_op}.')
        self.edge_feat_mp_op = edge_feat_mp_op
        if edge_feat_mp_op in ['concat']:
            self.fc_src = nn.Linear(
                self._in_src_feats * 2, out_feats * num_heads, bias=False
            )
        else:
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.fc_dst = nn.Linear(
            self._in_dst_feats, out_feats * num_heads, bias=False
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats,))
            )

        self.reset_parameters()
        self.activation = activation
        self.self_loop = self_loop

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
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, rel_graph, inputs,  get_attention=False, weight=None, edge_weight=None):
        """ GAT conv forward computation with edge feature.

        Parameters
        ----------
        rel_graph: DGLGraph
            Input DGL heterogenous graph with one edge type only.
        inputs: tuple of dict of Tensor
            Node features for each node type in the format of {ntype: tensor}.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.
        weight: dict of Tensor
            optional external node weight tensor. Not implemented. Reserved for future use.
        edge_weight: Tensor
            optional external edge weight tensor. Not implemented. Reserved for future use.

        Returns
        -------
        h: Tensor
            New node embeddings for destination node type.
        """

        # A corner case: no edge of this rel in this block. Will create an all 0s message and
        # multiply it with project weights as outputs, which is an all 0s tensor with output dim
        if rel_graph.num_edges() == 0:
            _, dst_inputs = inputs
            rst = th.zeros((dst_inputs.shape[0], self._num_heads, self._out_feats),
                           device=dst_inputs.device)
            if get_attention:
                return rst, None
            else:
                return rst
        else:
            assert len(inputs) == 3, 'For using edge features in message passing, you need to ' + \
                                     'provide 3 inputs in a tuple, the format is (src_inputs, ' + \
                                     f'dst_inputs, edge_inputs). but got {len(inputs)} inputs.'
            src_inputs, dst_inputs, edge_inputs = inputs
            assert src_inputs.shape[1:] == edge_inputs.shape[1:], \
                'To use edge feature in message passing computation, the node and edge ' + \
                'features should have the same dimensions, but got node feature dimension: ' + \
                f'{src_inputs.shape[1:]} and edge feature dimension: {edge_inputs.shape[1:]}.'

            with rel_graph.local_scope():
                dst_prefix_shape = dst_inputs.shape[:-1]
                edge_prefix_shape = edge_inputs.shape[:-1]
                rel_graph.srcdata['n_h'] = self.feat_drop(src_inputs)
                rel_graph.edata['e_h'] = self.feat_drop(edge_inputs)

                # u {edge_feat_mp_op} v
                if self.edge_feat_mp_op == 'concat':
                    rel_graph.apply_edges(lambda edges: {
                        'm': self.fc_src(
                            th.concat([edges.src['n_h'], edges.data['e_h']], dim=1))})
                elif self.edge_feat_mp_op == 'add':
                    rel_graph.apply_edges(
                        lambda edges: {'m': self.fc_src(edges.src['n_h'] + edges.data['e_h'])})
                elif self.edge_feat_mp_op == 'sub':
                    rel_graph.apply_edges(
                        lambda edges: {'m': self.fc_src(edges.src['n_h'] - edges.data['e_h'])})
                elif self.edge_feat_mp_op == 'mul':
                    rel_graph.apply_edges(
                        lambda edges: {'m': self.fc_src(edges.src['n_h'] * edges.data['e_h'])})
                elif self.edge_feat_mp_op == 'div':
                    rel_graph.apply_edges(
                        lambda edges: {'m': self.fc_src(edges.src['n_h'] / edges.data['e_h'])})
                else:
                    raise ValueError('Unknown edge message passing operation: ' + \
                                     f'{self.edge_feat_mp_op}. It should be one of ' + \
                                     f'{BUILTIN_EDGE_FEAT_MP_OPS}.')

                # projection for the dst nodes
                rel_graph.dstdata['n_h'] = self.fc_dst(self.feat_drop(dst_inputs)).view(
                    *dst_prefix_shape, self._num_heads, self._out_feats
                )

                rel_graph.edata['m'] = rel_graph.edata['m'].view(*edge_prefix_shape,
                                                                 self._num_heads, self._out_feats)

                # compute attention
                rel_graph.apply_edges(fn.e_add_v('m', 'n_h', 'm_attn'))
                m_attn = self.attn_drop(
                    edge_softmax(rel_graph,
                                 self.leaky_relu(rel_graph.edata.pop('m_attn'))))

                # message passing
                rel_graph.update_all(lambda edges: {'attn': (edges.data['m'] * m_attn)},
                                     fn.sum('attn', 'ft'))

                # extract outputs
                rst = rel_graph.dstdata['ft']

                # bias
                if self.bias:
                    rst = rst + self.bias.view(
                        *((1,) * len(dst_prefix_shape)),
                        self._num_heads,
                        self._out_feats
                    )

                # activation
                if self.activation:
                    rst = self.activation(rst)

                if get_attention:
                    return rst, m_attn
                else:
                    return rst
