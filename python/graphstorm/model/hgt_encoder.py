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
import torch
import torch.nn.functional as F
import dgl.function as fn

from torch import nn
from dgl.nn.functional import edge_softmax
from ..config import BUILDIN_GNN_BATCH_NORM, BUILDIN_GNN_LAYER_NORM, BUILTIN_GNN_NORM
from .ngnn_mlp import NGNNMLP
from .gnn_encoder_base import (GraphConvEncoder,
                               GSgnnGNNEncoderInterface)


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
        k_linears = {}
        q_linears = {}
        v_linears = {}
        a_linears = {}
        norms = {}
        skip = {}
        for ntype in ntypes:
            k_linears[ntype] = nn.Linear(in_dim, out_dim)
            q_linears[ntype] = nn.Linear(in_dim, out_dim)
            v_linears[ntype] = nn.Linear(in_dim, out_dim)
            a_linears[ntype] = nn.Linear(in_dim, out_dim)
            if self.use_norm:
                if norm == BUILDIN_GNN_BATCH_NORM:
                    norms[ntype] = nn.BatchNorm1d(out_dim)
                elif norm == BUILDIN_GNN_LAYER_NORM:
                    norms[ntype] = nn.LayerNorm(out_dim)
            skip[ntype] = nn.Parameter(torch.ones(1))

        self.k_linears = nn.ParameterDict(k_linears)
        self.q_linears = nn.ParameterDict(q_linears)
        self.v_linears = nn.ParameterDict(v_linears)
        self.a_linears = nn.ParameterDict(a_linears)
        if self.use_norm:
            self.norms = nn.ParameterDict(norms)
        self.skip = nn.ParameterDict(skip)

        # Edge type parameters
        relation_pri = {}
        relation_att = {}
        relation_msg = {}
        for canonical_etype in canonical_etypes:
            c_etype_str = '_'.join(canonical_etype)
            relation_pri[c_etype_str] = nn.Parameter(torch.ones(self.num_heads))
            relation_att[c_etype_str] = nn.init.xavier_uniform_(
                nn.Parameter(torch.Tensor(self.num_heads, self.d_k, self.d_k)))
            relation_msg[c_etype_str] = nn.init.xavier_uniform_(
                nn.Parameter(torch.Tensor(self.num_heads, self.d_k, self.d_k)))

        self.relation_pri = nn.ParameterDict(relation_pri)
        self.relation_att = nn.ParameterDict(relation_att)
        self.relation_msg = nn.ParameterDict(relation_msg)

        # ngnn
        self.num_ffn_layers_in_gnn = num_ffn_layers_in_gnn
        self.ngnn_mlp = NGNNMLP(out_dim, out_dim,
                                num_ffn_layers_in_gnn, fnn_activation, dropout)

        # Dropout
        self.drop = nn.Dropout(dropout)
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

                k_val = torch.einsum("bij,ijk->bik", k_val, relation_att)
                v_val = torch.einsum("bij,ijk->bik", v_val, relation_msg)

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
                    alpha = torch.sigmoid(self.skip[k])
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
                 dropout=0.2,
                 norm=BUILDIN_GNN_LAYER_NORM,
                 num_ffn_layers_in_gnn=0):
        super(HGTEncoder, self).__init__(hid_dim, out_dim, num_hidden_layers)

        self.layers = nn.ModuleList()
        # h2h
        for _ in range(num_hidden_layers):
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
        # h2o
        self.layers.append(HGTLayer(hid_dim,
                                    out_dim,
                                    g.ntypes,
                                    g.canonical_etypes,
                                    num_heads=num_heads,
                                    activation=F.relu,
                                    dropout=dropout,
                                    norm=norm))

    def skip_last_selfloop(self):
        # HGT does not have explicit self-loop
        pass

    def reset_last_selfloop(self):
        # HGT does not have explicit self-loop
        pass

    def forward(self, blocks, h):
        """HGT encoder forward computation.

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
