from graphstorm import model as gsmodel

from model_utils import get_unique_nfields
from model_utils import to_per_field_nfeats
from model_utils import rel_field_map
from model_utils import rel_name_map
from model_utils import average_over_fields
from model_utils import get_unique_etype_triplet

from model_utils import merge_multi_blocks, get_temporal_ordered_etypes

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GATConv

class TemporalRelationalGraphEncoder(gsmodel.gnn_encoder_base.GraphConvEncoder):
    """ Relational temporal graph conv encoder.

    Parameters
    ----------
    g: DistGraph
        The distributed graph
    h_dim : int
        Hidden dimension
    out_dim : int
        Output dimension
    num_hidden_layers : int
        Number of hidden layers. Total GNN layers is equal to num_hidden_layers + 1. Default 1
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout. Default 0.
    """
    def __init__(self, g, h_dim, out_dim, num_hidden_layers, num_heads=0, dropout=0):
        super(TemporalRelationalGraphEncoder, self).__init__(h_dim, out_dim, num_hidden_layers)
        self.node_fields = get_unique_nfields(g.etypes)
        for _ in range(num_hidden_layers):
            self.layers.append(
                ResidualTemporalRelationalGraphConv(
                    g,
                    in_feat=h_dim,
                    out_feat=h_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=nn.LeakyReLU(),
                )
            )
        self.layers.append(
            ResidualTemporalRelationalGraphConv(
                g,
                in_feat=h_dim,
                out_feat=out_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=nn.LeakyReLU(),
            )
        )

    def forward(self, blocks, inputs):
        """Forward computation.

        Parameters
        ----------
        blocks: DGL MFGs
            Sampled subgraph in DGL MFG
        inputs: dict[str, torch.Tensor]
            Input node feature for each node type.
        """
        h = to_per_field_nfeats(inputs, self.node_fields)
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        h = average_over_fields(h)
        return h

class ResidualTemporalRelationalGraphConv(nn.Module):
    """ Relational temporal graph conv layer with residual connection.

    Parameters
    ----------
    g: DistGraph
        The distributed graph
    in_feat : int
        Input dimension
    out_dim : int
        Output dimension
    num_heads : int
        Number of attention heads
    activation: torch.nn.Module
        Activation function.
    dropout : float
        Dropout. Default 0.
    """
    def __init__(self, g, in_feat, out_feat, num_heads, activation=None, dropout=0.0):
        super(ResidualTemporalRelationalGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.conv = TemporalRelationalGraphConv(g, in_feat, out_feat, num_heads)
        self.loop_weights = nn.Linear(in_feat, out_feat, bias=True)

    def forward(self, g, inputs):
        """Forward computation. Residual connection is used.

        Parameters
        ----------
        g: DGL MFGs
            Sampled subgraph in DGL MFG
        inputs: dict[str, dict[str, torch.Tensor]]
            Input node feature for each node type.
        """
        g = g.local_var()
        hs = self.conv(g, inputs)

        # Add residual connection, activation, dropout
        outputs = {}
        for ntype, ntype_inputs in inputs.items():
            outputs[ntype] = {}
            for nfield, ntype_nfield_inputs in ntype_inputs.items():
                # get the residual feats
                h = self.loop_weights(
                    ntype_nfield_inputs[: g.number_of_dst_nodes(ntype)]
                )
                if ntype in hs and nfield in hs[ntype]:
                    h = h + hs[ntype][nfield]
                if self.activation != None:
                    h = self.activation(h)
                outputs[ntype][nfield] = self.dropout(h)

        return outputs

class TemporalRelationalGraphConv(nn.Module):
    """ Relational temporal graph conv layer without residual connection.

    Parameters
    ----------
    g: DistGraph
        The distributed graph
    in_feat : int
        Input dimension
    out_dim : int
        Output dimension
    num_heads : int
        Number of attention heads
    """
    def __init__(self, g, in_feat, out_feat, num_heads):
        super(TemporalRelationalGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.node_fields = get_unique_nfields(g.etypes)

        self.conv = nn.ModuleDict()
        for canonical_etypes in get_unique_etype_triplet(g.canonical_etypes):
            # The "canonical_etypes" does not include timestamps, i.e., all timestamps
            # will utilize the same message passing model.
            conv_name = "%s-%s-%s" % (
                canonical_etypes[0],
                canonical_etypes[1],
                canonical_etypes[2],
            )
            gc_layer = GATConv(
                in_feat,
                out_feat // num_heads,
                num_heads,
                bias=False,
                allow_zero_in_degree=True,
            )
            self.conv[conv_name] = gc_layer

    def forward(self, g, inputs):
        """Forward computation.

        Parameters
        ----------
        g: DGL MFGs
            Sampled subgraph in DGL MFG
        inputs: dict[str, torch.Tensor]
            Input node feature for each node type.
        """
        mapping = get_temporal_ordered_etypes(g.canonical_etypes)

        outputs = {}
        for ntype in g.dsttypes:
            outputs[ntype] = {f"{t}_feat": [] for t in self.node_fields}

        for stype, etype, dtype in g.canonical_etypes:
            merge_canonical_etypes = mapping[stype, etype, dtype]
            # construct a new DGLBlock for temporal aggregation
            rel_graph, src_inputs, dst_inputs = merge_multi_blocks(
                g, inputs, merge_canonical_etypes
            )

            nfield = f"{rel_field_map(etype)}_feat"
            conv_name = "%s-%s-%s" % (stype, rel_name_map(etype), dtype)
            dstdata = self.conv[conv_name](
                rel_graph,
                (src_inputs, dst_inputs),
            )

            if nfield in outputs[dtype]:
                outputs[dtype][nfield].append(dstdata.view(dstdata.size(0), -1))

        # then, we aggregate information of each field
        final_outputs = {}

        for ntype, ntype_outputs in outputs.items():
            for nfield, ntype_nfield_outputs in ntype_outputs.items():
                if len(ntype_nfield_outputs) == 0:
                    continue

                dstdata = torch.stack(ntype_nfield_outputs, dim=0).mean(dim=0)

                if ntype not in final_outputs:
                    final_outputs[ntype] = {}
                final_outputs[ntype][nfield] = dstdata

        return final_outputs
