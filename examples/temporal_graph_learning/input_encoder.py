import torch
import torch.nn as nn
import torch.nn.functional as F

from graphstorm import model as gsmodel
from graphstorm.utils import get_rank

from model_utils import get_unique_nfields
from model_utils import get_trainable_params
from model_utils import to_per_field_nfeats
from model_utils import concat_over_fields

import dgl


class NodeEncoderInputLayer(gsmodel.embed.GSNodeInputLayer):
    """The temporal input encoder layer for all nodes in a heterogeneous graph.

    The input layer adds a linear layer on nodes with node features and the linear layer projects the node
    features to a specified dimension.
    Then, the input layer adds learnable embeddings as temporal encoding on the projected node features.


    Parameters
    ----------
    g: DistGraph
        The distributed graph
    feat_size : dict of int
        The original feat sizes of each node type
    embed_size : int
        The embedding size
    dropout : float
        The dropout parameter
    """
    def __init__(
        self,
        g,
        out_size,
        feat_size,
        dropout=0.0,
    ):
        super(NodeEncoderInputLayer, self).__init__(g)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_size = out_size
        self.feat_size = feat_size

        self.fields = get_unique_nfields(g.etypes)
        # "field_embeds" is a collection of trainable vectors used for time encoding purposes.

        # Each timestamp is associated with a learnable vector, which will be subsequently added
        # to the original node features. This will help models in distinguishing the timestamp
        # associated with each node feature. It's important to clarify that `field_embeds` does
        # not inherently contain information about the sequence order. Instead, we rely on neural
        # networks to learn this sequencing information through temporal aggregation
        # during the training process.

        self.field_embeds = nn.ParameterDict()
        for t in self.fields:
            self.field_embeds[f"{t}_feat"] = get_trainable_params(1, out_size)

        self.feats_proj_matrix = nn.ParameterDict()
        for ntype in g.ntypes:
            self.feats_proj_matrix[ntype] = get_trainable_params(feat_size[ntype], out_size)

    def forward(
        self, input_feats, input_nodes
    ):
        """Forward computation

        Parameters
        ----------
        input_feats: dict
            input features
        input_nodes: dict
            input node ids

        Returns
        -------
        a dict of Tensor: the node embeddings.
        """
        assert isinstance(input_nodes, dict), "The input node IDs should be in a dict."

        for ntype, feats in input_feats.items():
            feats = feats @ self.feats_proj_matrix[ntype]
            input_feats[ntype] = feats.repeat(1, len(self.fields))
        input_feats = to_per_field_nfeats(input_feats, self.fields)

        embs = {}
        for ntype in input_nodes:

            # add field_embeds to node embeds to capture temporal information
            embs[ntype] = {}
            for t in self.fields:
                field = f"{t}_feat"
                embs[ntype][field] = input_feats[ntype][field]
                embs[ntype][field] = embs[ntype][field] + self.field_embeds[field].expand(
                    embs[ntype][field].size(0), -1
                )
                embs[ntype][field] = self.activation(embs[ntype][field])
                embs[ntype][field] = self.dropout(embs[ntype][field])

        embs = concat_over_fields(embs, self.fields)
        return embs

    @property
    def out_dims(self):
        """The number of output dimensions."""
        return self.out_size
