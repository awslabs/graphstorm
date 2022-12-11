""" Decoders for edge predictions.
"""
import numpy as np
import torch as th
from torch import nn

from .gs_layer import GSLayer, GSLayerNoParam
from ..eval.utils import calc_distmult_pos_score, calc_dot_pos_score

# TODO(zhengda) we need to split it into classifier and regression.
class DenseBiDecoder(GSLayer):
    r"""Dense bi-linear decoder.
    Dense implementation of the bi-linear decoder used in GCMC. Suitable when
    the graph can be efficiently represented by a pair of arrays (one for source
    nodes; one for destination nodes).

    Parameters
    ----------
    in_units : int
        The input node feature size
    num_classes : int
        Number of classes.
    multilabel : bool
        Whether this is a multilabel classification.
    num_basis : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    target_etype : tuple of str
        The target etype for prediction
    regression : bool
        Whether this is true then we perform regression
    """
    def __init__(self,
                 in_units,
                 num_classes,
                 multilabel,
                 target_etype,
                 num_basis=2,
                 dropout_rate=0.0,
                 regression=False):
        super().__init__()

        basis_out = in_units if regression else num_classes
        self._in_units = in_units
        self._num_classes = num_classes
        self._multilabel = multilabel
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.basis_para = nn.Parameter(th.randn(num_basis, in_units, in_units))
        self.combine_basis = nn.Linear(self._num_basis, basis_out, bias=False)
        self.reset_parameters()
        self.regression = regression
        self.target_etype = target_etype
        if regression:
            self.regression_head = nn.Linear(basis_out, 1, bias=True)

    def reset_parameters(self):
        """Reset all parameters in this decoder with xavier uniform method
        """
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, g, h):
        """Forward function.
        Compute logits for each pair ``(ufeat[i], ifeat[i])``.
        Parameters
        ----------
        g : DGLBlock
            The minibatch graph
        h : dict of Tensors
            The dictionary containing the embeddings
        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge. Shape: (B, num_classes)
        """
        with g.local_scope():
            u, v = g.edges(etype=self.target_etype)
            (src_type, _, dest_type) = g.to_canonical_etype(etype=self.target_etype)
            ufeat = h[src_type][u]
            ifeat = h[dest_type][v]

            ufeat = self.dropout(ufeat)
            ifeat = self.dropout(ifeat)
            out = th.einsum('ai,bij,aj->ab', ufeat, self.basis_para.to(ifeat.device), ifeat)
            out = self.combine_basis(out)
            if self.regression:
                out = self.regression_head(out)

        return out

    def predict(self, ufeat, ifeat, _):
        """predict function for this decoder

        Parameters
        ----------
        ufeat : Tensor
            The source node features.
        ifeat : Tensor
            The destination node features.

        Returns
        -------
        Tensor : the scores of each edge.
        """
        out = th.einsum('ai,bij,aj->ab', ufeat, self.basis_para.to(ifeat.device), ifeat)
        out = self.combine_basis(out)
        if self.regression:
            out = self.regression_head(out)
        elif not self._multilabel:
            out = out.argmax(dim=1)
        return out

    @property
    def in_dims(self):
        """ The number of input dimensions.

        Returns
        -------
        int : the number of input dimensions.
        """
        return self._in_units

    @property
    def out_dims(self):
        """ The number of output dimensions.

        Returns
        -------
        int : the number of output dimensions.
        """
        return 1 if self.regression else self._num_classes


class MLPEdgeDecoder(GSLayer):
    """ MLP based edge classificaiton/regression decoder

    Parameters
    ----------
    h_dim : int
        Size of input dim of decoder. It is the dim of [src_emb || dst_emb]
    out_dim : int
        Output dim. e.g., number of classes
    multilabel : bool
        Whether this is a multilabel classification.
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
                 multilabel,
                 target_etype,
                 num_hidden_layers=1,
                 dropout=0,
                 regression=False):
        super(MLPEdgeDecoder, self).__init__()
        self.h_dim = h_dim
        self.multilabel = multilabel
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
        g : DGLBlock
            The minibatch graph
        h : dict of Tensors
            The dictionary containing the embeddings
        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge. Shape: (B, num_classes)
        """
        with g.local_scope():
            u, v = g.edges(etype=self.target_etype)
            (src_type, _, dest_type) = g.to_canonical_etype(etype=self.target_etype)
            ufeat = h[src_type][u]
            ifeat = h[dest_type][v]

            h = th.cat([ufeat, ifeat], dim=1)
            out = th.matmul(h, self.decoder)
            if self.regression:
                out = self.regression_head(out)

        return out

    def predict(self, ufeat, ifeat, _):
        """predict function for this decoder

        Parameters
        ----------
        ufeat : Tensor
            The source node features.
        ifeat : Tensor
            The destination node features.

        Returns
        -------
        Tensor : the scores of each edge.
        """
        h = th.cat([ufeat, ifeat], dim=1)
        out = th.matmul(h, self.decoder)
        if self.regression:
            out = self.regression_head(out)
        elif self.multilabel:
            out = out.argmax(dim=1)
        return out

    @property
    def in_dims(self):
        """ The number of input dimensions.

        Returns
        -------
        int : the number of input dimensions.
        """
        return self.h_dim

    @property
    def out_dims(self):
        """ The number of output dimensions.

        Returns
        -------
        int : the number of output dimensions.
        """
        return 1 if self.regression else self.out_dim

class LinkPredictDotDecoder(GSLayerNoParam):
    """ Link prediction decoder with the score function of dot product
    """
    def __init__(self, in_dim):
        self._in_dim = in_dim

    def forward(self, g, h):    # pylint: disable=arguments-differ
        """Forward function.

        This computes the dot product score on every edge type.
        """
        with g.local_scope():
            scores = []

            for etype in g.etypes:
                if g.num_edges(etype) == 0:
                    continue # the block might contain empty edge types

                (src_type, _, dest_type) = g.to_canonical_etype(etype=etype)
                u, v = g.edges(etype=etype)
                src_emb = h[src_type][u]
                dest_emb = h[dest_type][v]
                scores_etype = calc_dot_pos_score(src_emb, dest_emb)
                scores.append(scores_etype)

            scores=th.cat(scores)
            return scores

    @property
    def in_dims(self):
        """ The number of input dimensions.

        Returns
        -------
        int : the number of input dimensions.
        """
        return self._in_dim

    @property
    def out_dims(self):
        """ The number of output dimensions.

        Returns
        -------
        int : the number of output dimensions.
        """
        return 1

class LinkPredictDistMultDecoder(GSLayer):
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
        self.h_dim = h_dim
        self.etype2rid = {etype: i for i, etype in enumerate(g.etypes)}
        self._w_relation = nn.Embedding(self.num_rels, h_dim)
        self.trained_rels = np.zeros(self.num_rels)
        emb_init = gamma / h_dim
        nn.init.uniform_(self._w_relation.weight, -emb_init, emb_init)
        self.relids = th.arange(self.num_rels)#.to(self.device)

    def get_relemb(self, etype):
        """retrieve trained embedding of the given edge type

        Parameters
        ----------
        etype : str
            The edge type.
        """
        i = self.etype2rid[etype]
        assert self.trained_rels[i] > 0, 'The relation {} is not trained'.format(etype)
        return self._w_relation(th.tensor(i).to(self._w_relation.weight.device))

    def get_relembs(self):
        """retrieve all edges' trained embedding and edge type id mapping
        """
        return self._w_relation.weight, self.etype2rid

    def forward(self, g, h):
        """Forward function.

        This computes the DistMult score on every edge type.

        Parameters
        ----------
        g : DGLGraph
            a DGL graph for the edge prediction.
        h : dict of Tensor
            The node data for the input graph.

        Returns
        -------
        Tensor : the prediction scores for all edges in the input graph.
        """
        with g.local_scope():
            scores=[]

            for etype in g.etypes:
                if g.num_edges(etype) == 0:
                    continue # the block might contain empty edge types

                i = self.etype2rid[etype]
                self.trained_rels[i] += 1
                rel_embedding = self._w_relation(th.tensor(i).to(self._w_relation.weight.device))
                rel_embedding = rel_embedding.unsqueeze(dim=1)
                (src_type, _, dest_type) = g.to_canonical_etype(etype=etype)
                u, v = g.edges(etype=etype)
                src_emb = h[src_type][u]

                dest_emb = h[dest_type][v]
                rel_embedding = rel_embedding.repeat(1,dest_emb.shape[0]).T
                scores_etype = calc_distmult_pos_score(src_emb, dest_emb, rel_embedding)
                scores.append(scores_etype)
            scores=th.cat(scores)
            return scores

    @property
    def in_dims(self):
        """ The number of input dimensions.

        Returns
        -------
        int : the number of input dimensions.
        """
        return self.h_dim

    @property
    def out_dims(self):
        """ The number of output dimensions.

        Returns
        -------
        int : the number of output dimensions.
        """
        return 1
