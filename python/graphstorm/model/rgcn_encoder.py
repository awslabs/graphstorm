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

    RGCN layer implementation.
"""
import logging

import torch as th
from torch import nn
import torch.nn.functional as F
import dgl.nn as dglnn

from dgl.nn.pytorch.hetero import get_aggregate_fn
from .ngnn_mlp import NGNNMLP
from .gnn_encoder_base import (GraphConvEncoder,
                               GSgnnGNNEncoderInterface)


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer from `Modeling Relational Data
    with Graph Convolutional Networks <https://arxiv.org/abs/1703.06103>`__.

    A generic module for computing convolution on heterogeneous graphs.

    The relational graph convolution layer applies GraphConv on the heterogeneous graphs,
    which reads the features from source nodes and writes the updated ones to destination nodes.
    If multiple relations have the same destination node types, their results
    are aggregated by the specified method. If the heterogeneous graph has no edge,
    the corresponding module will not be called.

    Mathematically for the GraphConv it is defined as follows:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ji}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the set of neighbors of node :math:`i`,
    :math:`c_{ji}` is the product of the square root of node degrees
    (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`),
    and :math:`\sigma` is an activation function.

    Note:
    ******
    * The implementation of ``RelGraphConvLayer`` selects `right` as the norm, which divides
      the aggregated messages by each node's in-degrees, equivalent to averaging the received
      messages.

    Examples:
    ----------

    .. code:: python

        # suppose graph and input_feature are ready
        from graphstorm.model import RelGraphConvLayer

        layer = RelGraphConvLayer(
                in_feat=h_dim, out_feat=h_dim, rel_names=g.canonical_etypes,
                num_bases=num_bases, self_loop,
                dropout, num_ffn_layers_in_gnn,
                ffn_activation, norm)
        h = layer(g, input_feature)

    Parameters
    ----------
    in_feat: int
        Input feature size.
    out_feat: int
        Output feature size.
    rel_names: list of tuple
        Relation type list in the format of [('src_ntyp1', 'etype1', 'dst_ntype1`), ...].
    num_bases: int
        Number of bases. If is None, use number of relation types. Default: None.
    weight: bool
        Whether to apply a linear layer after message passing. Default: True.
    bias: bool
        Whether to add bias. Default: True.
    activation: callable
        Activation function. Default: None.
    self_loop: bool
        Whether to include self loop message. Default: True.
    dropout: float
        Dropout rate. Default: 0.
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
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0,
                 num_ffn_layers_in_gnn=0,
                 ffn_activation=F.relu,
                 norm=None):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = HeteroGraphConv({
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
                                     num_ffn_layers_in_gnn, ffn_activation, dropout)

        self.dropout = nn.Dropout(dropout)
        self.warn_msg = set()

    def warning_once(self, warn_msg):
        """ Print same warning msg only once

        Parameters
        ----------
        warn_msg: str
            Warning message.
        """
        if warn_msg in self.warn_msg:
            # Skip printing warning
            return
        self.warn_msg.add(warn_msg)
        logging.warning(warn_msg)

    # pylint: disable=invalid-name
    def forward(self, g, inputs):
        """ RGCN layer forward computation.

        Parameters
        ----------
        g: DGLHeteroGraph
            Input DGL heterogenous graph.
        inputs: dict of Tensor
            Node features for each node type in the format of {ntype: tensor}.

        Returns
        -------
        dict of Tensor: New node embeddings for each node type in the format of {ntype: tensor}.
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
            # DGL's message passing module requires to access the destination node embeddings.
            inputs_dst = {}
            for k in g.dsttypes:
                # If the destination node type exists in the input embeddings,
                # we can get from the input node embeddings directly because
                # the input nodes of DGL's block also contain the destination nodes
                if k in inputs:
                    inputs_dst[k] = inputs[k][:g.number_of_dst_nodes(k)]
                else:
                    # If the destination node type doesn't exist (this may happen if
                    # we use RGCN to construct node features), we should create a zero
                    # tensor. This tensor won't be used for computing embeddings.
                    # We need this just to fulfill the requirements of DGL message passing
                    # modules.
                    if g.num_dst_nodes(k) > 0:
                        assert not self.self_loop, \
                                f"We cannot allow self-loop if node {k} has no input features."
                    inputs_dst[k] = th.zeros((g.num_dst_nodes(k), self.in_feat),
                                             dtype=th.float32, device=g.device)
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, (inputs_src, inputs_dst), mod_kwargs=wdict)

        def _apply(ntype, h):
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
                    warn_msg = "Warning. Graph convolution returned empty " \
                        f"dictionary for nodes in type: {str(k)}. Please check your data" \
                        f" for no in-degree nodes in type: {str(k)}."
                    self.warning_once(warn_msg)
                    hs[k] = th.zeros((g.number_of_dst_nodes(k),
                                      self.out_feat),
                                     device=inputs[k].device)
                    # TODO the above might fail if the device is a different GPU
        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}


class RelationalGCNEncoder(GraphConvEncoder, GSgnnGNNEncoderInterface):
    """ Relational graph conv encoder.

    The ``RelationalGCNEncoder`` employs several ``RelGraphConvLayer`` as its encoding
    mechanism. The ``RelationalGCNEncoder`` should be designated as the model's encoder
    within Graphstorm.

    Parameters
    ----------
    g: DistGraph
        The distributed graph.
    h_dim: int
        Hidden dimension.
    out_dim: int
        Output dimension.
    num_bases: int
        Number of bases. If is None, use number of relation types. Default: None.
    num_hidden_layers: int
        Number of hidden layers. Total GNN layers is equal to ``num_hidden_layers + 1``.
        Default: 1.
    dropout: float
        Dropout rate. Default 0.
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

        # Build model and do full-graph inference on RelationalGCNEncoder
        from graphstorm import get_node_feat_size
        from graphstorm.model import RelationalGCNEncoder
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

        gnn_encoder = RelationalGCNEncoder(g, 4, 4,
                                           num_hidden_layers=1,
                                           dropout=0,
                                           use_self_loop=True,
                                           norm="batch")
        model.set_gnn_encoder(gnn_encoder)
        model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))

        h = do_full_graph_inference(model, np_data)
    """
    def __init__(self,
                 g,
                 h_dim, out_dim,
                 num_bases=-1,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=True,
                 last_layer_act=False,
                 num_ffn_layers_in_gnn=0,
                 norm=None):
        super(RelationalGCNEncoder, self).__init__(h_dim, out_dim, num_hidden_layers)
        if num_bases < 0 or num_bases > len(g.canonical_etypes):
            self.num_bases = len(g.canonical_etypes)
        else:
            self.num_bases = num_bases

        # h2h
        for _ in range(num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                h_dim, h_dim, g.canonical_etypes,
                self.num_bases, activation=F.relu, self_loop=use_self_loop,
                dropout=dropout, num_ffn_layers_in_gnn=num_ffn_layers_in_gnn,
                ffn_activation=F.relu, norm=norm))
        # h2o
        self.layers.append(RelGraphConvLayer(
            h_dim, out_dim, g.canonical_etypes,
            self.num_bases, activation=F.relu if last_layer_act else None,
            self_loop=use_self_loop, norm=norm if last_layer_act else None))

    def skip_last_selfloop(self):
        self.last_selfloop = self.layers[-1].self_loop
        self.layers[-1].self_loop = False

    def reset_last_selfloop(self):
        self.layers[-1].self_loop = self.last_selfloop

    # TODO(zhengda) refactor this to support edge features.
    def forward(self, blocks, h):
        """ RGCN encoder forward computation.

        Parameters
        ----------
        blocks: list of DGL MFGs
            Sampled subgraph in the list of DGL message flow graphs (MFGs) format. More
            detailed information about DGL MFG can be found in `DGL Neighbor Sampling
            Overview
            <https://docs.dgl.ai/stochastic_training/neighbor_sampling_overview.html>`_.
        h: dict of Tensor
            Input node features for each node type in the format of {ntype: tensor}.

        Returns
        ----------
        h: dict of Tensor
            New node embeddings for each node type in the format of {ntype: tensor}.
        """
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h


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

        Parameters
        ----------
        g: DGLGraph
            Graph data.
        inputs: dict[str, Tensor] or pair of dict[str, Tensor]
            Input node features.
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
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {
                    k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
                }

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
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
