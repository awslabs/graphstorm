"""Embedding layer implementation"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from .utils import sparse_emb_initializer

def init_emb(shape, dtype):
    arr = th.zeros(shape, dtype=dtype)
    nn.init.uniform_(arr, -1.0, 1.0)
    return arr

class DistGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""
    def __init__(self,
                 g,
                 feat_size,
                 text_feat_ntypes,
                 embed_size,
                 bert_dim = 768,
                 embed_name='embed',
                 activation=None,
                 dropout=0.0,
                 self_loop_init=False,
                 use_node_embeddings=False):
        """

        Parameters
        ----------
        g: DistGraph
        feat_size : the original feat size
        text_feat_ntypes : the text feat size
        embed_size : the embedding size
        bert_dim : the bert dimension
        embed_name : the name
        activation : the activation function
        dropout : the dropout parameter
        self_loop_init :  whether we will initialize the model with only the self loop matrix
        use_node_embeddings : whether we will use the node embeddings for individual nodes even when node features are
                              available.
        """
        super(DistGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.use_node_embeddings = use_node_embeddings

        # create weight embeddings for each node for each relation
        self.sparse_embeds = {}
        self.proj_matrix = nn.ParameterDict() if self.use_node_embeddings else None
        self.input_projs = nn.ParameterDict()
        for ntype in g.ntypes:
            feat_dim = 0
            if feat_size[ntype] > 0:
                feat_dim += feat_size[ntype]
            if ntype in text_feat_ntypes:
                feat_dim += bert_dim[ntype] if isinstance(bert_dim, dict) else bert_dim
            if feat_dim > 0:
                if g.rank() == 0:
                    print('Node {} has {} features.'.format(ntype, feat_dim))
                input_projs = nn.Parameter(th.Tensor(feat_dim, self.embed_size))
                nn.init.xavier_uniform_(input_projs, gain=nn.init.calculate_gain('relu'))
                self.input_projs[ntype] = input_projs
                if self_loop_init:
                    nn.init.eye_(self.input_projs[ntype])
                if self.use_node_embeddings:
                    if g.rank() == 0:
                        print('Use sparse embeddings on node {}'.format(ntype))
                    part_policy = g.get_node_partition_policy(ntype)
                    self.sparse_embeds[ntype] = dgl.distributed.DistEmbedding(g.number_of_nodes(ntype),
                                                                              self.embed_size,
                                                                              embed_name + '_' + ntype,
                                                                              init_emb,
                                                                              part_policy)
                    proj_matrix = nn.Parameter(th.Tensor(2 * self.embed_size, self.embed_size))
                    nn.init.xavier_uniform_(proj_matrix, gain=nn.init.calculate_gain('relu'))
                    self.proj_matrix[ntype] = proj_matrix
            else:
                part_policy = g.get_node_partition_policy(ntype)
                if g.rank() == 0:
                    print('Use sparse embeddings on node {}'.format(ntype))
                self.sparse_embeds[ntype] = dgl.distributed.DistEmbedding(g.number_of_nodes(ntype),
                                self.embed_size,
                                embed_name + '_' + ntype,
                                init_emb,
                                part_policy)

    def forward(self, input=None, target_ntype=None, input_nodes=None):
        """Forward computation

        Parameters
        ----------
        input: dict
            input_feats
        target_ntype: str
            if target ntype is specified, only compute the node embedding of the corresponding ntype
        input_nodes: dict
            input node ids

        Returns
        -------
        DGLHeteroGraph
            The block graph fed with embeddings.
        """
        embs = {}
        for ntype in self.g.ntypes:
            if target_ntype is not None and ntype != target_ntype:
                continue
            if ntype in self.input_projs: # nodes have input features
                if ntype not in input:
                    continue
                emb = input[ntype] @ self.input_projs[ntype]
                if self.use_node_embeddings:
                    node_emb = self.sparse_embeds[ntype](input_nodes[ntype])
                    concat_emb=th.cat((emb, node_emb),dim=1)
                    emb = concat_emb @ self.proj_matrix[ntype]
            else: # nodes do not have input features
                # if the input node of a node type is empty
                # skip this node type directly
                if len(input_nodes[ntype]) == 0:
                    continue
                emb = self.sparse_embeds[ntype](input_nodes[ntype])
            if self.activation is not None:
                emb = self.activation(emb)
            emb = self.dropout(emb)
            embs[ntype] = emb

        return embs
