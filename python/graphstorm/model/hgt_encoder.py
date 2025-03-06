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

    Heterogeneous Graph Transformer (HGT) layer implementation
"""
import logging
import math
import torch as th
import torch.nn.functional as F
import dgl.function as fn

from torch import nn
from dgl.nn.functional import edge_softmax
from ..config import BUILDIN_GNN_BATCH_NORM, BUILDIN_GNN_LAYER_NORM, BUILTIN_GNN_NORM
from .ngnn_mlp import NGNNMLP
from .gnn_encoder_base import (GraphConvEncoder,
                               GSgnnGNNEncoderInterface)
from ..config import BUILTIN_EDGE_FEAT_MP_OPS


class HGTLayer(nn.Module):
    r"""Heterogenous graph transformer (HGT) layer from `Heterogeneous Graph
    Transformer <https://arxiv.org/abs/2003.01332>`__.

    Given a graph :math:`G(V, E)` and input node features :math:`H^{(l-1)}` in the :math:`l-1`
    layer, it computes the new node features in the :math:`l` layer as follows:

    Compute a multi-head attention score for each edge :math:`(s, e, t)` in the graph:

    .. math::

      Attention(s, e, t) = \text{Softmax}\left(||_{i\in[1,h]}ATT-head^i(s, e, t)\right) \\
      ATT-head^i(s, e, t) = \left(K^i(s)W^{ATT}_{\phi(e)}Q^i(t)^{\top}\right)\cdot
        \frac{\mu_{(\tau(s),\phi(e),\tau(t)}}{\sqrt{d}} \\
      K^i(s) = \text{K-Linear}^i_{\tau(s)}(H^{(l-1)}[s]) \\
      Q^i(t) = \text{Q-Linear}^i_{\tau(t)}(H^{(l-1)}[t]) \\

    Compute the message to send on each edge :math:`(s, e, t)`:

    .. math::

      Message(s, e, t) = ||_{i\in[1, h]} MSG-head^i(s, e, t) \\
      MSG-head^i(s, e, t) = \text{M-Linear}^i_{\tau(s)}(H^{(l-1)}[s])W^{MSG}_{\phi(e)} \\

    Send messages to target nodes :math:`t` and aggregate:

    .. math::

      \tilde{H}^{(l)}[t] = \sum_{\forall s\in \mathcal{N}(t)}\left( Attention(s,e,t)
      \cdot Message(s,e,t)\right)

    Compute new node features:

    .. math::

      H^{(l)}[t]=\text{A-Linear}_{\tau(t)}(\sigma(\tilde{H}^{(l)}[t])) + H^{(l-1)}[t]

    Note:
    -----
    * Different from DGL's ``HGTConv``, this implementation is based on heterogeneous graphs.
      Other hyperparameters' default values are same as the DGL's ``HGTConv`` setting.

    * The cross-relation aggregation function of this implementation is ``mean``, which was chosen
      by authors of the HGT paper in their contribution to DGL.

    Examples:
    ----------

    .. code:: python

        # suppose graph and input_feature are ready
        from graphstorm.model import HGTLayer

        layer = HGTLayer(hid_dim, out_dim, g.ntypes, g.canonical_etypes,
                         num_heads, activation, dropout, norm)
        h = layer(g, input_feature)

    Parameters
    ----------
    in_dim: int
        Input dimension size.
    out_dim: int
        Output dimension size.
    ntypes: list of str
        List of node types in the format of [ntype1, ntype2, ...].
    canonical_etypes: list of tuple
        List of canonical edge types in the format of [('src_ntyp1', 'etype1', 'dst_ntype1`),
        ...].
    num_heads: int
        Number of attention heads.
    activation: callable
        Activation function. Default: None.
    dropout: float
        Dropout rate. Default: 0.2.
    norm: str
        Normalization methods. Options:``batch``, ``layer``, and ``None``. Default: ``layer``.
    num_ffn_layers_in_gnn: int
        Number of fnn layers between gnn layers. Default: 0.
    ffn_actication: torch.nn.functional
        Activation for ffn. Default: relu.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 ntypes,
                 canonical_etypes,
                 num_heads,
                 activation=None,
                 dropout=0.2,
                 norm=BUILDIN_GNN_LAYER_NORM,
                 num_ffn_layers_in_gnn=0,
                 fnn_activation=F.relu):
        super(HGTLayer, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        assert (out_dim % num_heads) == 0, f'The output dimension: {out_dim} should be divisible \
                                             by the number of heads: {num_heads}.'
        self.d_k = out_dim // num_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.activation = activation

        # normalization
        if norm in BUILTIN_GNN_NORM:
            self.use_norm = True
        else:
            self.use_norm = False

        # Node type parameters
        self.k_linears, \
            self.q_linears, \
            self.v_linears, \
            self.a_linears, \
            self.norms, \
            self.skip = self._create_node_parameters(ntypes, in_dim, out_dim,
                                                     self.use_norm, norm)

        # Edge type parameters
        self.relation_pri, \
            self.relation_att, \
            self.relation_msg = self._create_edge_parameters(canonical_etypes,
                                                             self.num_heads,
                                                             self.d_k)

        # ngnn
        self.num_ffn_layers_in_gnn = num_ffn_layers_in_gnn
        self.ngnn_mlp = NGNNMLP(out_dim, out_dim,
                                num_ffn_layers_in_gnn, fnn_activation, dropout)

        # Dropout
        self.drop = nn.Dropout(dropout)
        self.warn_msg = set()

    def _create_node_parameters(self, ntypes, in_dim, out_dim, use_norm, norm):
        """ Internal method for creating node type related HGT parameters.
        """
        k_linears = {}
        q_linears = {}
        v_linears = {}
        a_linears = {}
        norms = {}
        skip = {}
        for ntype in ntypes:
            # prepare the key, query, value transformer parameters for each node type
            k_linears[ntype] = nn.Linear(in_dim, out_dim)
            q_linears[ntype] = nn.Linear(in_dim, out_dim)
            v_linears[ntype] = nn.Linear(in_dim, out_dim)
            # prepare the attention transformer parameters for each node type
            a_linears[ntype] = nn.Linear(in_dim, out_dim)
            if use_norm:
                if norm == BUILDIN_GNN_BATCH_NORM:
                    norms[ntype] = nn.BatchNorm1d(out_dim)
                elif norm == BUILDIN_GNN_LAYER_NORM:
                    norms[ntype] = nn.LayerNorm(out_dim)
            skip[ntype] = nn.Parameter(th.ones(1))

        # wrap the key, query, value transformer parameters into a para dict
        para_k_linears = nn.ParameterDict(k_linears)
        para_q_linears = nn.ParameterDict(q_linears)
        para_v_linears = nn.ParameterDict(v_linears)
        # wrap the attention transformer parameters into a para dict
        para_a_linears = nn.ParameterDict(a_linears)
        # set normalization parameters for each node type
        if use_norm:
            para_norms = nn.ParameterDict(norms)
        else:
            para_norms = None
        # set skip jump parameters for each node type
        para_skip = nn.ParameterDict(skip)

        return para_k_linears, para_q_linears, para_v_linears, \
               para_a_linears, para_norms, para_skip

    def _create_edge_parameters(self, canonical_etypes, num_heads, d_k):
        """ Internal method for creating edge type related HGT parameters.
        """
        relation_pri = {}
        relation_att = {}
        relation_msg = {}
        for canonical_etype in canonical_etypes:
            c_etype_str = '_'.join(canonical_etype)
            # prepare the primary, attension, message transformer parameters for each edge type
            relation_pri[c_etype_str] = nn.Parameter(th.ones(num_heads))
            relation_att[c_etype_str] = nn.init.xavier_uniform_(
                nn.Parameter(th.Tensor(num_heads, d_k, d_k)))
            relation_msg[c_etype_str] = nn.init.xavier_uniform_(
                nn.Parameter(th.Tensor(num_heads, d_k, d_k)))

        # wrap the primary, attension, message transformer parameters into para dict
        para_relation_pri = nn.ParameterDict(relation_pri)
        para_relation_att = nn.ParameterDict(relation_att)
        para_relation_msg = nn.ParameterDict(relation_msg)

        return para_relation_pri, para_relation_att, para_relation_msg

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

    def forward(self, g, h):
        """ HGT layer forward computation.

        Parameters
        ----------
        g: DGLHeteroGraph
            Input DGL heterogenous graph.
        h: dict of Tensor
            Node features for each node type in the format of {ntype: tensor}.

        Returns
        -------
        new_h: dict of Tensor
            New node embeddings for each node type in the format of {ntype: tensor}.
        """
        # pylint: disable=no-member
        with g.local_scope():
            edge_fn = {}
            for srctype, etype, dsttype in g.canonical_etypes:
                c_etype_str = '_'.join((srctype, etype, dsttype))
                # extract each relation as a sub graph
                sub_graph = g[srctype, etype, dsttype]

                k_linear = self.k_linears[srctype]
                v_linear = self.v_linears[srctype]
                q_linear = self.q_linears[dsttype]

                k_val = k_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
                v_val = v_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
                if g.is_block:
                    q_val = q_linear(h[dsttype][:sub_graph.num_dst_nodes()]).view(-1,
                                                                            self.num_heads,
                                                                            self.d_k)
                else:
                    q_val = q_linear(h[dsttype]).view(-1, self.num_heads, self.d_k)

                relation_att = self.relation_att[c_etype_str]
                relation_pri = self.relation_pri[c_etype_str]
                relation_msg = self.relation_msg[c_etype_str]

                k_val = th.einsum("bij,ijk->bik", k_val, relation_att)
                v_val = th.einsum("bij,ijk->bik", v_val, relation_msg)

                sub_graph.srcdata['k'] = k_val
                sub_graph.dstdata['q'] = q_val
                sub_graph.srcdata[f'v_{c_etype_str}'] = v_val

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')
                sub_graph.edata[f't_{c_etype_str}'] = attn_score.unsqueeze(-1)

                edge_fn[srctype, etype, dsttype] = (fn.u_mul_e(f'v_{c_etype_str}',
                                                               f't_{c_etype_str}', 'm'),
                                                    fn.sum('m', 't'))

            g.multi_update_all(edge_fn, cross_reducer="mean")

            new_h = {}
            for k, _ in h.items():
                if g.num_dst_nodes(k) > 0:
                    alpha = th.sigmoid(self.skip[k])
                    if g.dstnodes[k].data.get('t') is not None:
                        t = g.dstnodes[k].data['t'].view(-1, self.out_dim)
                        trans_out = self.drop(t)
                        if g.is_block:
                            trans_out = trans_out * alpha + \
                                self.a_linears[k](h[k][:g.num_dst_nodes(k)]) * (1-alpha)
                        else:
                            trans_out = trans_out * alpha + self.a_linears[k](h[k]) * (1-alpha)
                    else:                       # Nodes not really in destination side.
                        warn_msg = "Warning. Graph convolution returned empty " \
                            f"dictionary for nodes in type: {str(k)}. Please check your data" \
                            f" for no in-degree nodes in type: {str(k)}."
                        self.warning_once(warn_msg)
                        # So add psudo self-loop for the destination nodes with its own feature.
                        dst_h = self.a_linears[k](h[k][:g.num_dst_nodes(k)])
                        trans_out = self.drop(dst_h)
                        trans_out = trans_out * alpha + dst_h * (1-alpha)
                else:
                    # Handle zero number of dst nodes, which is an extreme case
                    if g.dstnodes[k].data.get('t') is not None:
                        trans_out = self.a_linears[k](h[k])
                    else:
                        continue

                if self.use_norm:
                    new_h[k] = self.norms[k](trans_out)
                else:
                    new_h[k] = trans_out
                if self.activation:
                    new_h[k] = self.activation(new_h[k])
                if self.num_ffn_layers_in_gnn > 0:
                    new_h[k] = self.ngnn_mlp(new_h[k])

            return new_h


class HGTEncoder(GraphConvEncoder, GSgnnGNNEncoderInterface):
    r""" Heterogenous Graph Transformer (HGT) encoder.

    The ``HGTEncoder`` employs several ``HGTLayer`` as its encoding mechanism.
    The ``HGTEncoder`` should be designated as the model's encoder within Graphstorm.

    .. versionchanged:: 0.4.1
        Add two new arguments ``edge_feat_name`` and ``edge_feat_mp_op`` in v0.4.1 to
        support edge features in HGT encoder.

    Parameters
    -----------
    g: DistGraph
        The input distributed graph.
    hid_dim: int
        Hidden dimension size.
    out_dim: int
        Output dimension size.
    num_hidden_layers: int
        Number of hidden layers. Total GNN layers is equal to ``num_hidden_layers + 1``.
    num_heads: int
        Number of attention heads.
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
        Dropout rate. Default: 0.2.
    norm: str
        Normalization methods. Options:``batch``, ``layer``, and ``None``. Default: ``layer``.
    num_ffn_layers_in_gnn: int
        Number of fnn layers between GNN layers. Default: 0.

    Examples:
    ----------

    .. code:: python

        # Build model and do full-graph inference on HGTEncoder
        from graphstorm import get_node_feat_size
        from graphstorm.model import HGTEncoder
        from graphstorm.model import MLPEdgeDecoder
        from graphstorm.model import GSgnnEdgeModel, GSNodeEncoderInputLayer
        from graphstorm.dataloading import GSgnnData
        from graphstorm.model import do_full_graph_inference

        np_data = GSgnnData(...)

        model = GSgnnEdgeModel(alpha_l2norm=0)
        feat_size = get_node_feat_size(np_data.g, "feat")
        encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                          dropout=0,
                                          use_node_embeddings=True)
        model.set_node_input_encoder(encoder)

        gnn_encoder = HGTEncoder(g,
                                 hid_dim=4,
                                 out_dim=4,
                                 num_hidden_layers=1,
                                 num_heads=2,
                                 dropout=0.0,
                                 norm="layer",
                                 num_ffn_layers_in_gnn=0)
        model.set_gnn_encoder(gnn_encoder)
        model.set_decoder(MLPEdgeDecoder(model.gnn_encoder.out_dims,
                                         3, multilabel=False, target_etype=("n0", "r1", "n1"),
                                         num_ffn_layers=num_ffn_layers))

        h = do_full_graph_inference(model, np_data)
    """
    def __init__(self,
                 g,
                 hid_dim,
                 out_dim,
                 num_hidden_layers,
                 num_heads,
                 edge_feat_name=None,
                 edge_feat_mp_op='concat',
                 dropout=0.2,
                 norm=BUILDIN_GNN_LAYER_NORM,
                 num_ffn_layers_in_gnn=0):
        super(HGTEncoder, self).__init__(hid_dim, out_dim, num_hidden_layers)

        # check edge type string format
        if edge_feat_name:
            for etype, _ in edge_feat_name.items():
                assert len(etype) == 3, 'The edge type should be in canonical type format:' + \
                    f'(src_ntype, etype, dst_ntype), but got \"{etype}\".'

        self.layers = nn.ModuleList()
        # h2h
        for _ in range(num_hidden_layers):
            if edge_feat_name is None:
                self.layers.append(HGTLayer(hid_dim,
                                            hid_dim,
                                            g.ntypes,
                                            g.canonical_etypes,
                                            activation=F.relu,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            norm=norm,
                                            num_ffn_layers_in_gnn=num_ffn_layers_in_gnn,
                                            fnn_activation=F.relu))
            else:
                self.layers.append(HGTLayerwithEdgeFeat(
                                            hid_dim,
                                            hid_dim,
                                            g.ntypes,
                                            g.canonical_etypes,
                                            activation=F.relu,
                                            num_heads=num_heads,
                                            edge_feat_name=edge_feat_name,
                                            edge_feat_mp_op=edge_feat_mp_op,
                                            dropout=dropout,
                                            norm=norm,
                                            num_ffn_layers_in_gnn=num_ffn_layers_in_gnn,
                                            fnn_activation=F.relu))
        # h2o
        if edge_feat_name is None:
            self.layers.append(HGTLayer(hid_dim,
                                        out_dim,
                                        g.ntypes,
                                        g.canonical_etypes,
                                        num_heads=num_heads,
                                        activation=F.relu,
                                        dropout=dropout,
                                        norm=norm))
        else:
            self.layers.append(HGTLayerwithEdgeFeat(
                                        hid_dim,
                                        out_dim,
                                        g.ntypes,
                                        g.canonical_etypes,
                                        activation=F.relu,
                                        num_heads=num_heads,
                                        edge_feat_name=edge_feat_name,
                                        edge_feat_mp_op=edge_feat_mp_op,
                                        dropout=dropout,
                                        norm=norm,
                                        num_ffn_layers_in_gnn=num_ffn_layers_in_gnn,
                                        fnn_activation=F.relu))

    def skip_last_selfloop(self):
        # HGT does not have explicit self-loop
        pass

    def reset_last_selfloop(self):
        # HGT does not have explicit self-loop
        pass

    def is_support_edge_feat(self):
        """ Overwrite ``GraphConvEncoder`` class' method, indicating HGTEncoder
       supports edge features.
        """
        return True

    def forward(self, blocks, n_h, e_hs=None):
        """HGT encoder forward computation.

        .. versionchanged:: 0.4.1
            Change inputs into blocks, n_h and e_hs in v0.4.1 to support edge feature
            in HGT encoder.

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
            Input edge features for each edge type in the format of [{etype1: tensor,
            etype2: tensor, ...}, ...], or [{}, {}. ...] for zero number of edges in input
            blocks. The length of e_hs should be equal to the number of gnn layers.
            Default is None.

        Returns
        ----------
        h: dict of Tensor
            New node embeddings for each node type in the format of {ntype: tensor}.
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


class HGTLayerwithEdgeFeat(HGTLayer):
    r""" Heterogenous graph transformer (HGT) layer with edge feature supported.

    .. versionadded:: 0.4.1
        In version 0.4.1, add a new HGT layer that supports edge features.

    This class extends from `HGTLayer`.

    Implementation in this class uses a simple idea to include edge features into the original
    HGT model, i.e., combine embeddings of source node with embeddings of edge as the new `K`
    and `V`, then use this new `K` and `V` in HGT formulas. And the way of combination is same
    as the RGCN conv model, including `concat`, `add`, `sub`, `mul`, and `div`. Then the formula
    of computing the message to send on each edge :math:`(s, e, t)` become:

    .. math::

      MSG-head^i(s, e, t) = \text{M-Linear}^i_{\tau(s)}(H^{(l-1)}[s] \text{ op } EF_{e})
      W^{MSG}_{\phi(e)} \\

    where :math:`\text{op}` is one of the `add`, `sub`, `mul`, and `div` operators, and the
    :math:`EF_{e}` is the edge feature of the :math:`\phi(e)` edge type.
    
    For the `concat` operator, the formula is

    .. math::

      MSG-head^i(s, e, t) = (\text{M-Linear}^i_{\tau(s)}(H^{(l-1)}[s]) + 
      \text{EF-Linear}^i_{\phi(e)}(EF_{e}))W^{MSG}_{\phi(e)} \\

    where :math:`\text{EF-Linear}^i_{\phi(e)}` is an additional weight for the :math:`\phi(e)`
    edge type. 
    
    This formula uses a linear algebra trick to implement concatenation operation.
    That is, a linear computation of :math:`concat([e1, e2], dim=-1) @ w` equals to the
    computation of :math:`e1 @ w1 + e2 @ w2`, where embedding :math:`e1` and :math:`e2` have
    the same dimension :math:`(N * in\_dim)`, weight :math:`w` has the dimension
    :math:`(in\_dim * 2, out\_dim)`, and weight :math:`w1` and :math:`w2` have the same dimension
    :math:`(in\_dim, out\_dim)`. Based on this trick, instead of concatenating the source node
    embeddings and edge embeddings first, and then use an edge type specific weights with
    dimension :math:`(in\_dim * 2, out\_dim)` for linear transformation, this implementation uses
    two separated weights, i.e., one for source node type, and one for edge type, for their
    linear transformation first, and then add the transformed embeddings togethor.

    For other HGT formulas, please refer to the `HGTLayer`.

    Parameters
    ----------
    in_dim: int
        Input dimension size.
    out_dim: int
        Output dimension size.
    ntypes: list of str
        List of node types in the format of [ntype1, ntype2, ...].
    canonical_etypes: list of tuple
        List of canonical edge types in the format of [('src_ntyp1', 'etype1', 'dst_ntype1`),
        ...].
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
    activation: callable
        Activation function. Default: None.
    dropout: float
        Dropout rate. Default: 0.2.
    norm: str
        Normalization methods. Options:``batch``, ``layer``, and ``None``. Default: ``layer``.
    num_ffn_layers_in_gnn: int
        Number of fnn layers between gnn layers. Default: 0.
    ffn_actication: torch.nn.functional
        Activation for ffn. Default: relu.    
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 ntypes,
                 canonical_etypes,
                 num_heads,
                 edge_feat_name=None,
                 edge_feat_mp_op='concat',
                 activation=None,
                 dropout=0.2,
                 norm=BUILDIN_GNN_LAYER_NORM,
                 num_ffn_layers_in_gnn=0,
                 fnn_activation=F.relu):
        # initialize base HGT parameters
        super(HGTLayerwithEdgeFeat, self).__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            ntypes=ntypes,
            canonical_etypes=canonical_etypes,
            num_heads=num_heads,
            activation=activation,
            dropout=dropout,
            norm=norm,
            num_ffn_layers_in_gnn=num_ffn_layers_in_gnn,
            fnn_activation=fnn_activation
        )
        # set edge feature related variables
        assert edge_feat_name, 'To use HGTLayerwithEdgeFeat, must provide non-empty edge, ' + \
            f'feature name, but got {edge_feat_name}.'
        self.edge_feat_name=edge_feat_name

        assert edge_feat_mp_op in BUILTIN_EDGE_FEAT_MP_OPS, 'GraphStorm only support edge' + \
            f' message passing operation in {BUILTIN_EDGE_FEAT_MP_OPS}, bug got ' + \
            f'{edge_feat_mp_op}.'
        self.edge_feat_mp_op=edge_feat_mp_op

        # initialize edge feature related parameters for `concat` op
        if edge_feat_mp_op in ['concat']:
            self.ef_linears = self._create_ef_parameters(in_dim, out_dim, canonical_etypes,
                                                         edge_feat_name)

    def _create_ef_parameters(self, in_dim, out_dim, canonical_etypes, edge_feat_name):
        """ Create edge feature specific parameters when message passing operator is `concat`.

        With the design idea of combining edge feature into HGT algorithm, only when the
        message passing operator is `concat`, will we need addition weight parameters. For
        other operators, there is no addition parameters required because edge embeddings have
        the same dimension as the embeddings of source nodes so can be directly added, substracted,
        mutilied or devided.

        In this implementation, we use a linear algebra trick for concatenation operation, i.e.,
        concat([e1, e2], dim=-1) @ w ==  e1 @ w1 + e2 @ w2, where e1 and e2 have the same
        dimension (N * in_dim), w1 and w2 have the same dimension (in_dim, out_dim), and w has
        the dimension (in_dim * 2, out_dim).
        
        Based on this trick, we define edge type specifc parameters to do the linear
        transformation for edge embeddings. For source node embeddings, its weights are defined in
        the `HGTLayer`'s `_create_node_parameters()` function.
        """
        ef_linears = {}
        for canonical_etype in canonical_etypes:
            # TODO: Due to the undetermistic nature of mini-batch sampling, it is possible that
            #       some edge type parameters may not be used at all during model training because
            #       no such edge types are sampled in a block. This could cause weird Pytorch
            #       errors, which are hard to solve because we need to define the weights anyway.
            #  Purpose of this TODO is to leave a clue of this possible error source.
            if canonical_etype in edge_feat_name:
                c_etype_str = '_'.join(canonical_etype)
                ef_linears[c_etype_str] = nn.Linear(in_dim, out_dim)

        para_ef_linears = nn.ParameterDict(ef_linears)

        return para_ef_linears

    def forward(self, g, h, e_h=None):
        """ HGT with edge feature support layer forward computation.

        Parameters
        ----------
        g: DGLHeteroGraph
            Input DGL heterogenous graph.
        h: dict of Tensor
            Node features for each node type in the format of {ntype: tensor}.
        e_h: dict of Tensor
            edge features for each edge type in the format of {etype: tensor}. Default is None.

        Returns
        -------
        dict of Tensor: New node embeddings for each node type in the format of {ntype: tensor}.
        """
        # A corner case, there is 0 edges of edge type with edge features. So no input e_h will
        # be given.
        if e_h is None or len(e_h) == 0:
            total_num_edge = 0
            for can_etype in self.edge_feat_name.keys():
                total_num_edge += g.num_edges(etype=can_etype)
            assert total_num_edge == 0, f"No edge features provided for {total_num_edge} " + \
                "edges in HGTLayerwithEdgeFeat, please check the edge feature information " + \
                "specified in the \"edge_feat_name\" argument during initialization " + \
                "or check the \"e_h\" argument of the forward function.."
            e_h = {}

        # pylint: disable=no-member
        with g.local_scope():
            edge_fn = {}
            for srctype, etype, dsttype in g.canonical_etypes:
                c_etype_str = '_'.join((srctype, etype, dsttype))
                # extract each relation as a sub graph
                sub_graph = g[srctype, etype, dsttype]

                # extract source, destination, and edge embeds
                src_nh = h[srctype]
                sub_graph.srcdata['src_nh'] = src_nh

                # copy source embed to edges, and extract from edata
                sub_graph.apply_edges(fn.copy_u('src_nh', 'src_eh'))
                src_eh = sub_graph.edata['src_eh']

                # extract src, dst, and edge parameters
                k_linear = self.k_linears[srctype]
                v_linear = self.v_linears[srctype]
                q_linear = self.q_linears[dsttype]

                relation_att = self.relation_att[c_etype_str]
                relation_pri = self.relation_pri[c_etype_str]
                relation_msg = self.relation_msg[c_etype_str]

                # combine src_eh with e_h
                if (srctype, etype, dsttype) in e_h.keys():
                    edge_h = e_h[(srctype, etype, dsttype)]
                    if self.edge_feat_mp_op == 'concat':
                        ef_linear = self.ef_linears[c_etype_str]

                        src_k_val = k_linear(src_eh)
                        edge_val = ef_linear(edge_h)
                        k_val = src_k_val + edge_val
                        src_v_val = v_linear(src_eh)
                        v_val = src_v_val + edge_val
                    elif self.edge_feat_mp_op == 'add':
                        new_src_h = src_eh + edge_h
                        k_val = k_linear(new_src_h)
                        v_val = v_linear(new_src_h)
                    elif self.edge_feat_mp_op == 'sub':
                        new_src_h = src_eh - edge_h
                        k_val = k_linear(new_src_h)
                        v_val = v_linear(new_src_h)
                    elif self.edge_feat_mp_op == 'mul':
                        new_src_h = src_eh * edge_h
                        k_val = k_linear(new_src_h)
                        v_val = v_linear(new_src_h)
                    elif self.edge_feat_mp_op == 'div':
                        new_src_h = src_eh / edge_h
                        k_val = k_linear(new_src_h)
                        v_val = v_linear(new_src_h)
                    else:
                        raise ValueError('Unknown edge message passing operation: ' + \
                                        f'{self.edge_feat_mp_op}. It should be one of ' + \
                                        f'{BUILTIN_EDGE_FEAT_MP_OPS}.')

                    k_val = k_val.view(-1, self.num_heads, self.d_k)
                    v_val = v_val.view(-1, self.num_heads, self.d_k)
                else:
                    k_val = k_linear(src_eh).view(-1, self.num_heads, self.d_k)
                    v_val = v_linear(src_eh).view(-1, self.num_heads, self.d_k)

                k_val = th.einsum("bij,ijk->bik", k_val, relation_att)
                v_val = th.einsum("bij,ijk->bik", v_val, relation_msg)

                if g.is_block:
                    q_val = q_linear(h[dsttype][:sub_graph.num_dst_nodes()]).view(-1,
                                                                            self.num_heads,
                                                                            self.d_k)
                else:
                    q_val = q_linear(h[dsttype]).view(-1, self.num_heads, self.d_k)

                # compute attention scores
                sub_graph.edata['k'] = k_val
                sub_graph.dstdata['q'] = q_val
                sub_graph.apply_edges(fn.v_dot_e('q', 'k', 't'))

                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')
                attn_v_val = v_val * attn_score.unsqueeze(-1)
                sub_graph.edata[f'h_{c_etype_str}'] = attn_v_val

                edge_fn[srctype, etype, dsttype] = (fn.copy_e(f'h_{c_etype_str}', 'm'),
                                                    fn.sum('m', 't'))

            g.multi_update_all(edge_fn, cross_reducer="mean")

            new_h = {}
            for k, _ in h.items():
                if g.num_dst_nodes(k) > 0:
                    alpha = th.sigmoid(self.skip[k])
                    if g.dstnodes[k].data.get('t') is not None:
                        t = g.dstnodes[k].data['t'].view(-1, self.out_dim)
                        trans_out = self.drop(t)
                        if g.is_block:
                            trans_out = trans_out * alpha + \
                                self.a_linears[k](h[k][:g.num_dst_nodes(k)]) * (1-alpha)
                        else:
                            trans_out = trans_out * alpha + self.a_linears[k](h[k]) * (1-alpha)
                    else:                       # Nodes not really in destination side.
                        warn_msg = "Warning. Graph convolution returned empty " \
                            f"dictionary for nodes in type: {str(k)}. Please check your data" \
                            f" for no in-degree nodes in type: {str(k)}."
                        self.warning_once(warn_msg)
                        # So add psudo self-loop for the destination nodes with its own feature.
                        dst_h = self.a_linears[k](h[k][:g.num_dst_nodes(k)])
                        trans_out = self.drop(dst_h)
                        trans_out = trans_out * alpha + dst_h * (1-alpha)
                else:
                    # Handle zero number of dst nodes, which is an extreme case
                    if g.dstnodes[k].data.get('t') is not None:
                        trans_out = self.a_linears[k](h[k])
                    else:
                        continue

                if self.use_norm:
                    new_h[k] = self.norms[k](trans_out)
                else:
                    new_h[k] = trans_out
                if self.activation:
                    new_h[k] = self.activation(new_h[k])
                if self.num_ffn_layers_in_gnn > 0:
                    new_h[k] = self.ngnn_mlp(new_h[k])

            return new_h
