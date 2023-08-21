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


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from dgl.nn.functional import edge_softmax
from .ngnn_mlp import NGNNMLP
from .gnn_encoder_base import GraphConvEncoder


class HGTLayer(nn.Module):
    r"""Heterogenous graph transformer (HGT) layer.

    Parameters
    ----------
    in_dim : int
        Input dimension size.
    out_dim : int
        Output dimension size.

    num_heads : int
        Number of attention heads
    dropout : float, optional
        Dropout rate. Default: 0.0
    num_ffn_layers_in_gnn: int, optional
        Number of layers of ngnn between gnn layers
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                #  node_dict,         # node type and id in order, e.g., {'author': 0, 'paper': 1, 'subject': 2}
                #  edge_dict,         # edge type and id in order, e.g., {'writing': 0, 'cited': 1, 'citing': 2}
                 ntypes,
                 canonical_etypes,
                 num_heads,
                 dropout=0.0,
                 use_norm=True,
                 num_ffn_layers_in_gnn=0
                 ):
        super(HGTLayer, self).__init__()
        # self.in_dim        = in_dim
        # self.out_dim       = out_dim
        # self.node_dict     = node_dict
        # self.edge_dict     = edge_dict
        # self.num_ntypes    = len(node_dict)
        # self.num_etypes    = len(edge_dict)
        self.num_heads       = num_heads
        self.d_k           = out_dim // num_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm    = use_norm

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()

        for _ in range(self.num_ntypes):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_etypes, self.num_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_etypes, num_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_etypes, num_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_ntypes))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, graph, h):
        if graph.is_block:
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in graph.canonical_etypes:
                # extract each relation as a sub graph
                sub_graph = graph[srctype, etype, dsttype]

                # check if no edges exist for this can_etype
                if sub_graph.num_edges() == 0:
                    continue

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
                q = q_linear(h[dsttype][:sub_graph.num_dst_nodes()]).view(-1, self.num_heads, self.d_k)

                e_id = self.edge_dict[(srctype, etype, dsttype)]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata[f'v_{e_id :d}'] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            edge_fn = {}
            for etype, e_id in edge_dict.items():
                if etype not in graph.canonical_etypes:
                    continue
                else:
                    edge_fn[etype] = (fn.u_mul_e(f'v_{e_id :d}', 't', 'm'), fn.sum('m', 't'))
            graph.multi_update_all(edge_fn, cross_reducer = 'mean')

            new_h = {}
            for ntype in graph.dsttypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                if graph.num_dst_nodes(ntype) == 0:
                    continue

                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = graph.dstnodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype][: graph.num_dst_nodes(ntype)] * (1-alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out

            return new_h


class HGTEncoder(GraphConvEncoder):
    r"""Heterogenous graph transformer (HGT) encoder

    Parameters
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
    use_norm: boolean
        If use normalization or not, default is True
    num_ffn_layers_in_gnn: int
        Number of ngnn gnn layers between GNN layers
    """
    def __init__(self,
                 g,
                 hid_dim,
                 out_dim,
                 num_hidden_layers,
                 num_heads,
                 dropout=0.0,
                 use_norm=True,
                 num_ffn_layers_in_gnn=0
                 ):
        super(HGTEncoder, self).__init__()

        self.layers = nn.ModuleList()
        # h2h
        for _ in range(num_hidden_layers):
            self.layers.append(HGTLayer(g,
                                        hid_dim,
                                        hid_dim,
                                        num_heads,
                                        dropout,
                                        use_norm,
                                        num_ffn_layers_in_gnn))
        # h2o
            self.layers.append(HGTLayer(g,
                                        hid_dim,
                                        out_dim,
                                        num_heads,
                                        dropout,
                                        use_norm,
                                        num_ffn_layers_in_gnn))

    def forward(self, blocks, h):
        """Forward computation

        Parameters
        ----------
        blocks: DGL MFGs
            Sampled subgraph in DGL MFG
        h: dict[str, torch.Tensor]
            Input node feature for each node type.
        """
        for i in range(self.num_layers):
            h = self.layers[i](blocks[i], h)

        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)

        return h


class RelationalGATEncoder(GraphConvEncoder):
    r"""Relational graph attention encoder

    Parameters
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
    """
    def __init__(self,
                 g,
                 h_dim, out_dim, num_heads,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=True,
                 last_layer_act=False,
                 num_ffn_layers_in_gnn=0):
        super(RelationalGATEncoder, self).__init__(h_dim, out_dim, num_hidden_layers)
        self.num_heads = num_heads
        # h2h
        for _ in range(num_hidden_layers):
            self.layers.append(RelationalAttLayer(
                h_dim, h_dim, g.canonical_etypes,
                self.num_heads, activation=F.relu, self_loop=use_self_loop,
                dropout=dropout, num_ffn_layers_in_gnn=num_ffn_layers_in_gnn,
                fnn_activation=F.relu))
        # h2o
        self.layers.append(RelationalAttLayer(
            h_dim, out_dim, g.canonical_etypes,
            self.num_heads, activation=F.relu if last_layer_act else None,
            self_loop=use_self_loop))

    def forward(self, blocks, h):
        """Forward computation

        Parameters
        ----------
        blocks: DGL MFGs
            Sampled subgraph in DGL MFG
        h: dict[str, torch.Tensor]
            Input node feature for each node type.
        """
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h
