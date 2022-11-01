"""Link prediction decoders
"""
import torch as th
import numpy as np
import torch.nn as nn
import dgl

from ..eval.utils import calc_distmult_pos_score, calc_dot_pos_score

class DenseBiDecoder(nn.Module):
    r"""Dense bi-linear decoder.
    Dense implementation of the bi-linear decoder used in GCMC. Suitable when
    the graph can be efficiently represented by a pair of arrays (one for source
    nodes; one for destination nodes).

    Parameters
    ----------
    in_units : int
        Size of input user and movie features
    num_classes : int
        Number of classes.
    num_basis : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    target_etype :  is the target etype for prediction
    regression : if this is true then we perform regression
    """
    def __init__(self,
                 in_units,
                 num_classes,
                 target_etype,
                 num_basis=2,
                 dropout_rate=0.0,
                 regression=False):
        super().__init__()

        basis_out = in_units if regression else num_classes
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.P = nn.Parameter(th.randn(num_basis, in_units, in_units))
        self.combine_basis = nn.Linear(self._num_basis, basis_out, bias=False)
        self.reset_parameters()
        self.regression = regression
        self.target_etype = target_etype
        if regression:
            self.regression_head = nn.Linear(basis_out, 1, bias=True)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, g, h):
        """Forward function.
        Compute logits for each pair ``(ufeat[i], ifeat[i])``.
        Parameters
        ----------
        g : the minibatch graph
        h : the dictionary containing the embeddings
        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge. Shape: (B, num_classes)
        """
        with g.local_scope():
            u, v = g.edges(etype=self.target_etype)
            (src_type, e_type, dest_type) = g.to_canonical_etype(etype=self.target_etype)
            ufeat = h[src_type][u]

            ifeat = h[dest_type][v]
            out = self.predict(ufeat, ifeat)

        return out


    def predict(self, ufeat, ifeat):
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        out = th.einsum('ai,bij,aj->ab', ufeat, self.P.to(ifeat.device), ifeat)
        out = self.combine_basis(out)
        if self.regression:
            out = self.regression_head(out)
        return out


class MLPEdgeDecoder(nn.Module):
    """ MLP based edge classificaiton/regression decoder

    Parameters
    ----------
    h_dim : int
        Size of input dim of decoder. It is the dim of [src_emb || dst_emb]
    out_dim : int
        Output dim. e.g., number of classes
    target_etype : tuple of str
        Target etype for prediction
    num_hidden_layers: int
        Number of layers
    regression : Bool
        If this is true then we perform regression
    dropout: float
        Dropout
    """
    def __init__(self,
                 h_dim,
                 out_dim,
                 target_etype,
                 num_hidden_layers=1,
                 dropout=0,
                 regression=False):
        super(MLPEdgeDecoder, self).__init__()
        self.h_dim = h_dim
        self.out_dim = h_dim if regression else out_dim
        self.decoder = nn.Parameter(th.randn(h_dim, out_dim))
        self.target_etype = target_etype
        assert num_hidden_layers == 1, "More than one layers not supported"
        nn.init.xavier_uniform_(self.decoder,
                                gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(dropout)
        self.regression = regression
        if regression:
            self.regression_head = nn.Linear(self.out_dim, 1, bias=True)

    def forward(self, g, h):
        """Forward function.
        Compute logits for each pair ``(ufeat[i], ifeat[i])``.
        Parameters
        ----------
        g : the minibatch graph
        h : the dictionary containing the embeddings
        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge. Shape: (B, num_classes)
        """
        with g.local_scope():
            u, v = g.edges(etype=self.target_etype)
            (src_type, e_type, dest_type) = g.to_canonical_etype(etype=self.target_etype)
            ufeat = h[src_type][u]

            ifeat = h[dest_type][v]
            out = self.predict(ufeat, ifeat)

        return out

    def predict(self, ufeat, ifeat):
        h = th.cat([ufeat, ifeat], dim=1)
        out = th.matmul(h, self.decoder)
        if self.regression:
            out = self.regression_head(out)
        return out

class DenseBiDecoderWithEdgeFeats(nn.Module):
    r"""Dense bi-linear decoder.
    Dense implementation of the bi-linear decoder used in GCMC. Suitable when
    the graph can be efficiently represented by a pair of arrays (one for source
    nodes; one for destination nodes).
    Parameters
    ----------
    g : graph
    in_units : int
        Size of input user and movie features
    num_classes : int
        Number of classes.
    num_basis : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    efeat_name : the name of the edge features
    target_etype :  is the target etype for prediction
    regression : if this is true then we perform regression
    """
    def __init__(self,
                 g,
                 in_units,
                 num_classes,
                 target_etype,
                 num_basis=2,
                 efeat_name='embeddings',
                 dropout_rate=0.1,
                 regression=False,
                 only_efeats=False):
        super().__init__()

        basis_out = in_units if regression else num_classes
        self.target_etype = target_etype
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.P = nn.Parameter(th.randn(num_basis, in_units, in_units))
        self.combine_basis = nn.Linear(self._num_basis, basis_out, bias=False)

        self.efeat_name = efeat_name
        self.only_efeats = only_efeats
        self.efeat_dim = g.edges[self.target_etype[1]].data[efeat_name].shape[1]
        self.efeat_transform = nn.Sequential(
            nn.Linear(self.efeat_dim, in_units),
            nn.ReLU(),
            nn.Linear(in_units, in_units),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(in_units, in_units),
        )

        if only_efeats:
            print("Only the edge features will be considered during training.")
            self.combine_w_efeats = nn.Linear(in_units, in_units, bias=False)
        else:
            self.combine_w_efeats = nn.Linear(2*in_units, in_units, bias=False)
        self.efeats = g.edges[self.target_etype[1]].data[efeat_name]

        self.reset_parameters()
        self.regression = regression

        if regression:
            self.regression_head = nn.Linear(basis_out, 1, bias=True)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, g, h):
        """Forward function.
        Compute logits for each pair ``(ufeat[i], ifeat[i])``.
        Parameters
        ----------
        g : the minibatch graph
        h : the dictionary containing the embeddings
        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge. Shape: (B, num_classes)
        """
        with g.local_scope():
            u, v = g.edges(etype=self.target_etype)
            (src_type, e_type, dest_type) = g.to_canonical_etype(etype=self.target_etype)
            ufeat = h[src_type][u]
            ifeat = h[dest_type][v]
            # get the edge EIDs
            seeds = g.edges[self.target_etype[1]].data[dgl.EID]
            efeats = self.efeats[seeds]
            out = self.predict(ufeat, ifeat, efeats)

        return out


    def predict(self, ufeat, ifeat, efeats):
        efeats = efeats.to(self.P.device)
        efeats_transf = self.efeat_transform(efeats)
        efeats_transf = self.dropout(efeats_transf)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        if self.only_efeats:
            ufeat = self.combine_w_efeats(th.cat([efeats_transf], dim=1))
            ifeat = self.combine_w_efeats(th.cat([efeats_transf], dim=1))
        else:
            ufeat = self.combine_w_efeats(th.cat([ufeat, efeats_transf], dim=1))
            ifeat = self.combine_w_efeats(th.cat([ifeat, efeats_transf], dim=1))
        out = th.einsum('ai,bij,aj->ab', ufeat, self.P.to(ifeat.device), ifeat)
        out = self.combine_basis(out)
        if self.regression:
            out = self.regression_head(out)
        return out

class MLPEdgeDecoderWithEdgeFeats(nn.Module):
    def __init__(self,
                 g,
                 h_dim,
                 out_dim,
                 target_etype,
                 efeat_name='embeddings',
                 num_hidden_layers=1,
                 dropout=0.1,
                 only_efeats=False):
        super(MLPEdgeDecoderWithEdgeFeats, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.efeat_name = efeat_name
        self.target_etype = target_etype
        self.only_efeats = only_efeats
        self.efeat_dim = g.edges[self.target_etype[1]].data[efeat_name].shape[1]
        self.efeat_transform = nn.Sequential(
            nn.Linear(self.efeat_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )
        if self.only_efeats:
            print("Only the edge features will be considered during training with {} dimension".
                  format(self.efeat_dim))
            self.decoder = nn.Parameter(th.randn(h_dim, out_dim))
        else:
            print("Edge features will be added during training with {} dimension".
                  format(self.efeat_dim))
            self.decoder = nn.Parameter(th.randn(2*h_dim, out_dim))
        self.efeats = g.edges[self.target_etype[1]].data[efeat_name]

        assert num_hidden_layers == 1, "More than one layers not supported"

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, g, h):
        """Forward function.
        Compute logits for each pair ``(ufeat[i], ifeat[i])``.
        Parameters
        ----------
        g : the minibatch graph
        h : the dictionary containing the embeddings
        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge. Shape: (B, num_classes)
        """
        with g.local_scope():
            u, v = g.edges(etype=self.target_etype)
            (src_type, e_type, dest_type) = g.to_canonical_etype(etype=self.target_etype)
            ufeat = h[src_type][u]
            ifeat = h[dest_type][v]
            seeds = g.edges[self.target_etype[1]].data[dgl.EID]
            efeats = self.efeats[seeds]

            out = self.predict(ufeat, ifeat, efeats)

        return out

    def predict(self, ufeat, ifeat, efeats):
        efeats = efeats.to(self.decoder.device)
        efeats_transf = self.efeat_transform(efeats)
        ufeat = self.dropout(ufeat)
        efeats_transf = self.dropout(efeats_transf)
        ifeat = self.dropout(ifeat)
        if self.only_efeats:
            h = efeats_transf
        else:
            h = th.cat([ufeat, ifeat, efeats_transf], dim=1)
        return th.matmul(h, self.decoder)


class LinkPredictDotDecoder(nn.Module):
    """ Link prediction decoder with the score function of dot product
    """
    def __init__(self):
        """

        Parameters
        ----------
        """
        super(LinkPredictDotDecoder, self).__init__()


    def forward(self, g, h):
        with g.local_scope():
            scores = []

            for etype in g.etypes:
                if g.num_edges(etype) == 0:
                    continue # the block might contain empty edge types

                (src_type,e_type,dest_type) = g.to_canonical_etype(etype=etype)
                u, v = g.edges(etype=etype)
                src_emb = h[src_type][u]
                dest_emb = h[dest_type][v]
                scores_etype = calc_dot_pos_score(src_emb, dest_emb)
                scores.append(scores_etype)

            scores=th.cat(scores)
            return scores

class LinkPredictDistMultDecoder(nn.Module):
    """ Link prediction decoder with the score function of DistMult

    Parameters
    ----------
    """
    def __init__(self,
                 g,
                 h_dim,
                 gamma=40.):
        super(LinkPredictDistMultDecoder, self).__init__()
        self.num_rels = len(g.etypes)
        self.etype2rid = {etype: i for i, etype in enumerate(g.etypes)}
        self._w_relation = nn.Embedding(self.num_rels, h_dim)
        self.trained_rels = np.zeros(self.num_rels)
        emb_init = gamma / h_dim
        nn.init.uniform_(self._w_relation.weight, -emb_init, emb_init)
        self.relids = th.arange(self.num_rels)#.to(self.device)

    def get_relemb(self, etype):
        i = self.etype2rid[etype]
        assert self.trained_rels[i] > 0, 'The relation {} is not trained'.format(etype)
        return self._w_relation(th.tensor(i).to(self._w_relation.weight.device))

    def get_relembs(self):
        return self._w_relation.weight, self.etype2rid

    def forward(self, g, h):
        with g.local_scope():
            scores=[]

            for etype in g.etypes:
                if g.num_edges(etype) == 0:
                    continue # the block might contain empty edge types

                i = self.etype2rid[etype]
                self.trained_rels[i] += 1
                rel_embedding = self._w_relation(th.tensor(i).to(self._w_relation.weight.device))
                rel_embedding = rel_embedding.unsqueeze(dim=1)
                (src_type,e_type,dest_type) = g.to_canonical_etype(etype=etype)
                u, v = g.edges(etype=etype)
                src_emb = h[src_type][u]

                dest_emb = h[dest_type][v]
                rel_embedding = rel_embedding.repeat(1,dest_emb.shape[0]).T
                scores_etype = calc_distmult_pos_score(src_emb, dest_emb, rel_embedding)
                scores.append(scores_etype)
            scores=th.cat(scores)
            return scores
