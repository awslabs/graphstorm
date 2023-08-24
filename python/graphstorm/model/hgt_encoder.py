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
    
    Different from DGL's HGTConv, this implementation is based on heterogeneous graph.
    Other hyperparameters' default values are same as the DGL's setting.
    
    Parameters
    ----------
    in_dim : int
        Input dimension size.
    out_dim : int
        Output dimension size.
    ntypes: list[str]
        List of node types
    canonical_etypes: list[(str, str, str)]
        List of canonical edge types
    num_heads : int
        Number of attention heads
    dropout : float, optional
        Dropout rate. Default: 0.2
    use_norm: boolean
        If use layer normalization or not, default is True
    num_ffn_layers_in_gnn: int, optional
        Number of layers of ngnn between gnn layers
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 ntypes,
                 canonical_etypes,
                 num_heads,
                 dropout=0.2,
                 use_norm=True,
                 num_ffn_layers_in_gnn=0,
                 fnn_activation=F.relu):
        super(HGTLayer, self).__init__()
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        # Define node type parameters
        k_linears = {}
        q_linears = {}
        v_linears = {}
        a_linears = {}
        norms = {}
        skip = {}
        for ntype in ntypes:
            k_linears[ntype] = (nn.Linear(in_dim, out_dim))
            q_linears[ntype] = (nn.Linear(in_dim, out_dim))
            v_linears[ntype] = (nn.Linear(in_dim, out_dim))
            a_linears[ntype] = (nn.Linear(in_dim, out_dim))
            if use_norm:
                norms[ntype] = (nn.LayerNorm(out_dim))
            skip[ntype] = nn.Parameter(torch.ones(1))


        self.k_linears = nn.ModuleDict(k_linears)
        self.q_linears = nn.ModuleDict(q_linears)
        self.v_linears = nn.ModuleDict(v_linears)
        self.a_linears = nn.ModuleDict(a_linears)
        if use_norm:
            self.norms = nn.ModuleDict(norms)

        # Define edge type parameters
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

        self.relation_pri = nn.ModuleDict(relation_pri)
        self.relation_att = nn.ModuleDict(relation_att)
        self.relation_msg = nn.ModuleDict(relation_msg)
        self.skip = nn.ModuleDict(skip)

        # ngnn
        self.num_ffn_layers_in_gnn = num_ffn_layers_in_gnn
        self.ngnn_mlp = NGNNMLP(out_dim, out_dim,
                                num_ffn_layers_in_gnn, fnn_activation, dropout)

        # Define dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, g, h):
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
        if g.is_block:
            h_src = {}
            h_dst = {}
            for srctype, etype, dsttype in g.canonical_etypes:
                # extract each relation as a sub graph
                sub_graph = g[srctype, etype, dsttype]
                
                h_src[srctype] = h[srctype]
                h_dst[dsttype] = h[dsttype][:sub_graph.num_dst_nodes()]
        else:
            h_src = {}
            h_dst = {}
            for srctype, etype, dsttype in g.canonical_etypes:
                # extract each relation as a sub graph
                sub_graph = g[srctype, etype, dsttype]
                
                h_src[srctype] = h[srctype]
                h_dst[dsttype] = h[dsttype]

        edge_fn = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            # extract each relation as a sub graph
            sub_graph = g[srctype, etype, dsttype]

            # check if no edges exist for this can_etype
            if sub_graph.num_edges() == 0:
                continue

            k_linear = self.k_linears[srctype]
            v_linear = self.v_linears[srctype]
            q_linear = self.q_linears[dsttype]

            k = k_linear(h_src[srctype]).view(-1, self.num_heads, self.d_k)
            v = v_linear(h_src[srctype]).view(-1, self.num_heads, self.d_k)
            q = q_linear(h_dst[dsttype]).view(-1, self.num_heads, self.d_k)

            c_etype_str = '_'.join((srctype, etype, dsttype))

            relation_att = self.relation_att[c_etype_str]
            relation_pri = self.relation_pri[c_etype_str]
            relation_msg = self.relation_msg[c_etype_str]

            k = torch.einsum("bij,ijk->bik", k, relation_att)
            v = torch.einsum("bij,ijk->bik", v, relation_msg)

            sub_graph.srcdata['k'] = k
            sub_graph.dstdata['q'] = q
            sub_graph.srcdata[f'v_{c_etype_str}'] = v

            sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
            attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
            attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

            sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            edge_fn[(srctype, etype, dsttype)] = (fn.u_mul_e(f'v_{c_etype_str}', 't', 'm'), fn.sum('m', 't'))

        g.multi_update_all(edge_fn, cross_reducer = 'mean')

        new_h = {}
        for ntype in g.dsttypes:
            if g.num_dst_nodes(ntype) == 0:
                continue

            alpha = torch.sigmoid(self.skip[ntype])
            t = g.dstnodes[ntype].data['t'].view(-1, self.out_dim)
            trans_out = self.drop(self.a_linears[ntype](t))
            trans_out = trans_out * alpha + h_dst[ntype] * (1-alpha)
            if self.use_norm:
                new_h[ntype] = self.norms[ntype](trans_out)
            else:
                new_h[ntype] = trans_out

            if self.num_ffn_layers_in_gnn > 0:
                new_h[ntype] = self.ngnn_mlp(new_h[ntype])

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
        If use layer normalization or not, default is False
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
                 use_norm=False,
                 num_ffn_layers_in_gnn=0):
        super(HGTEncoder, self).__init__()

        self.layers = nn.ModuleList()
        # h2h
        for _ in range(num_hidden_layers):
            self.layers.append(HGTLayer(hid_dim,
                                        hid_dim,
                                        g.ntypes,
                                        g.canonical_etypes,
                                        num_heads,
                                        dropout,
                                        use_norm,
                                        num_ffn_layers_in_gnn,
                                        fnn_activation=F.relu))
        # h2o
        self.layers.append(HGTLayer(hid_dim,
                                    out_dim,
                                    g.ntypes,
                                    g.canonical_etypes,
                                    num_heads,
                                    dropout,
                                    use_norm))

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
