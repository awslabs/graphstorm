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

    Decoders for edge predictions.
"""
import numpy as np
import torch as th
from torch import nn

from .gs_layer import GSLayer, GSLayerNoParam
from ..dataloading import (BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                           BUILTIN_LP_JOINT_NEG_SAMPLER,
                           LP_DECODER_EDGE_WEIGHT,
                           EP_DECODER_EDGE_FEAT)
from ..eval.utils import calc_distmult_pos_score, calc_dot_pos_score
from ..eval.utils import calc_distmult_neg_head_score, calc_distmult_neg_tail_score

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
            src_type, _, dest_type = self.target_etype
            ufeat = h[src_type][u]
            ifeat = h[dest_type][v]

            ufeat = self.dropout(ufeat)
            ifeat = self.dropout(ifeat)
            out = th.einsum('ai,bij,aj->ab', ufeat, self.basis_para.to(ifeat.device), ifeat)
            out = self.combine_basis(out)
            if self.regression:
                out = self.regression_head(out)

        return out

    def predict(self, g, h):
        """predict function for this decoder

        Parameters
        ----------
        g : DGLBlock
            The minibatch graph
        h : dict of Tensors
            The dictionary containing the embeddings

        Returns
        -------
        Tensor : the maximum score of each edge.
        """
        with g.local_scope():
            u, v = g.edges(etype=self.target_etype)
            src_type, _, dest_type = self.target_etype
            ufeat = h[src_type][u]
            ifeat = h[dest_type][v]
            out = th.einsum('ai,bij,aj->ab', ufeat, self.basis_para.to(ifeat.device), ifeat)
            out = self.combine_basis(out)
            if self.regression:
                out = self.regression_head(out)
            elif self._multilabel:
                out = (th.sigmoid(out) > .5).long()
            else:  # not multilabel
                out = out.argmax(dim=1)
        return out

    def predict_proba(self, g, h):
        """predict function for this decoder

        Parameters
        ----------
        g : DGLBlock
            The minibatch graph
        h : dict of Tensors
            The dictionary containing the embeddings

        Returns
        -------
        Tensor : all the scores of each edge.
        """
        with g.local_scope():
            u, v = g.edges(etype=self.target_etype)
            src_type, _, dest_type = self.target_etype
            ufeat = h[src_type][u]
            ifeat = h[dest_type][v]
            out = th.einsum('ai,bij,aj->ab', ufeat, self.basis_para.to(ifeat.device), ifeat)
            out = self.combine_basis(out)
            if self.regression:
                out = self.regression_head(out)
            elif self._multilabel:
                out = th.sigmoid(out)
            else:
                out = th.softmax(out, 1)
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
        The input dim of decoder. It is the dim of source or destinatioin node embeddings.
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
        self.target_etype = target_etype
        self.regression = regression
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers

        self._init_model()

    def _init_model(self):
        """ Init decoder model
        """
        # Here we assume the source and destination nodes have the same dimension.
        self.decoder = nn.Parameter(th.randn(self.h_dim * 2, self.out_dim))
        assert self.num_hidden_layers == 1, "More than one layers not supported"
        nn.init.xavier_uniform_(self.decoder,
                                gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(self.dropout)
        if self.regression:
            self.regression_head = nn.Linear(self.out_dim, 1, bias=True)

    def _compute_logits(self, g, h):
        """ Compute forword output

            Parameters
            ----------
            g : DGLBlock
                The minibatch graph
            h : dict of Tensors
                The dictionary containing the embeddings
            Returns
            -------
            th.Tensor
                Output of forward
        """
        with g.local_scope():
            u, v = g.edges(etype=self.target_etype)
            src_type, _, dest_type = self.target_etype
            ufeat = h[src_type][u]
            ifeat = h[dest_type][v]

            h = th.cat([ufeat, ifeat], dim=1)
            out = th.matmul(h, self.decoder)
        return out

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
        out = self._compute_logits(g, h)

        if self.regression:
            out = self.regression_head(out)
        return out

    def predict(self, g, h):
        """Predict function for this decoder

        Parameters
        ----------
        g : DGLBlock
            The minibatch graph
        h : dict of Tensors
            The dictionary containing the embeddings

        Returns
        -------
        Tensor : the scores of each edge.
        """
        out = self._compute_logits(g, h)

        if self.regression:
            out = self.regression_head(out)
        elif self.multilabel:
            out = (th.sigmoid(out) > .5).long()
        else:  # not multilabel
            out = out.argmax(dim=1)
        return out

    def predict_proba(self, g, h):
        """Predict function for this decoder

        Parameters
        ----------
        g : DGLBlock
            The minibatch graph
        h : dict of Tensors
            The dictionary containing the embeddings

        Returns
        -------
        Tensor : the scores of each edge.
        """
        out = self._compute_logits(g, h)

        if self.regression:
            out = self.regression_head(out)
        elif self.multilabel:
            out = th.sigmoid(out)
        else:
            out = th.softmax(out, 1)
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

class MLPEFeatEdgeDecoder(MLPEdgeDecoder):
    """ MLP based edge classificaiton/regression decoder

    Parameters
    ----------
    h_dim : int
        The input dim of decoder. It is the dim of source or destinatioin node embeddings.
    feat_dim : int
        The input dim of edge features which are used with NN output.
    out_dim : int
        Output dim. e.g., number of classes
    multilabel : bool
        Whether this is a multilabel classification.
    target_etype : tuple of str
        Target etype for prediction
    regression : Bool
        If this is true then we perform regression
    dropout: float
        Dropout
    """
    def __init__(self,
                 h_dim,
                 feat_dim,
                 out_dim,
                 multilabel,
                 target_etype,
                 dropout=0,
                 regression=False):
        self.feat_dim = feat_dim
        super(MLPEFeatEdgeDecoder, self).__init__(h_dim=h_dim,
                                                  out_dim=out_dim,
                                                  multilabel=multilabel,
                                                  target_etype=target_etype,
                                                  dropout=dropout,
                                                  regression=regression)

    def _init_model(self):
        """ Init decoder model
        """
        self.relu = th.nn.ReLU()

        # [src_emb | dest_emb] @ W -> h_dim
        # Here we assume the source and destination nodes have the same dimension.
        self.nn_decoder = nn.Parameter(th.randn(self.h_dim * 2, self.h_dim))
        # [edge_feat] @ W -> h_dim
        self.feat_decoder = nn.Parameter(th.randn(self.feat_dim, self.h_dim))
        # combine output of nn_decoder and feat_decoder
        self.combine_decoder = nn.Parameter(th.randn(self.h_dim * 2, self.h_dim))
        self.decoder = nn.Parameter(th.randn(self.h_dim, self.out_dim))
        self.dropout = nn.Dropout(self.dropout)


        nn.init.xavier_uniform_(self.nn_decoder,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.feat_decoder,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.combine_decoder,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.decoder,
                                gain=nn.init.calculate_gain('relu'))
        if self.regression:
            self.regression_head = nn.Linear(self.out_dim, 1, bias=True)

    def _compute_logits(self, g, h):
        """ Compute forword output

            Parameters
            ----------
            g : DGLBlock
                The minibatch graph
            h : dict of Tensors
                The dictionary containing the embeddings
            Returns
            -------
            th.Tensor
                Output of forward
        """
        with g.local_scope():
            u, v = g.edges(etype=self.target_etype)
            src_type, _, dest_type = self.target_etype
            ufeat = h[src_type][u]
            ifeat = h[dest_type][v]
            efeat = g.edges[self.target_etype].data[EP_DECODER_EDGE_FEAT]

            # [src_emb | dest_emb] @ W -> h_dim
            h = th.cat([ufeat, ifeat], dim=1)
            nn_h = th.matmul(h, self.nn_decoder)
            nn_h = self.relu(nn_h)
            nn_h = self.dropout(nn_h)
            # [edge_feat] @ W -> h_dim
            feat_h = th.matmul(efeat, self.feat_decoder)
            feat_h = self.relu(feat_h)
            feat_h = self.dropout(feat_h)
            # [nn_h | feat_h] @ W -> h_dim
            combine_h = th.cat([nn_h, feat_h], dim=1)
            combine_h = th.matmul(combine_h, self.combine_decoder)
            combine_h = self.relu(combine_h)
            out = th.matmul(combine_h, self.decoder)

        return out

##################### Link Prediction Decoders #######################
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

            for canonical_etype in g.canonical_etypes:
                if g.num_edges(canonical_etype) == 0:
                    continue # the block might contain empty edge types

                src_type, _, dest_type = canonical_etype
                u, v = g.edges(etype=canonical_etype)
                src_emb = h[src_type][u]
                dest_emb = h[dest_type][v]
                scores_etype = calc_dot_pos_score(src_emb, dest_emb)
                scores.append(scores_etype)

            scores=th.cat(scores)
            return scores

    def calc_test_scores(self, emb, pos_neg_tuple, neg_sample_type, device):
        """ Compute scores for positive edges and negative edges

        Parameters
        ----------
        emb: dict of Tensor
            Node embeddings.
        pos_neg_tuple: dict of tuple
            Positive and negative edges stored in a tuple:
            tuple(positive source, negative source,
            postive destination, negatve destination).
            The positive edges: (positive source, positive desitnation)
            The negative edges: (positive source, negative desitnation) and
                                (negative source, positive desitnation)
        neg_sample_type: str
            Describe how negative samples are sampled.
                Uniform: For each positive edge, we sample K negative edges
                Joint: For one batch of positive edges, we sample
                       K negative edges
        device: th.device
            Device used to compute scores

        Return
        ------
        Dict of (Tensor, Tensor)
            Return a dictionary of edge type to
            (positive scores, negative scores)
        """
        assert isinstance(pos_neg_tuple, dict) and len(pos_neg_tuple) == 1, \
            "DotDecoder is only applicable to link prediction task with " \
            "single target training edge type"
        canonical_etype = list(pos_neg_tuple.keys())[0]
        pos_src, neg_src, pos_dst, neg_dst = pos_neg_tuple[canonical_etype]
        utype, _, vtype = canonical_etype
        pos_src_emb = emb[utype][pos_src].to(device)
        pos_dst_emb = emb[vtype][pos_dst].to(device)

        scores = {}
        pos_scores = calc_dot_pos_score(pos_src_emb, pos_dst_emb)
        neg_scores = []
        if neg_src is not None:
            neg_src_emb = emb[utype][neg_src.reshape(-1,)].to(device)
            if neg_sample_type == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
                neg_src_emb = neg_src_emb.reshape(
                    neg_src.shape[0], neg_src.shape[1], -1)
                pos_dst_emb = pos_dst_emb.reshape(
                    pos_dst_emb.shape[0], 1, pos_dst_emb.shape[1])
                neg_score = calc_dot_pos_score(neg_src_emb, pos_dst_emb)
            elif neg_sample_type == BUILTIN_LP_JOINT_NEG_SAMPLER:
                # joint sampled negative samples
                assert len(pos_dst_emb.shape) == 2, \
                    "For joint negative sampler, in evaluation" \
                    "positive src/dst embs should in shape of" \
                    "[eval_batch_size, dimension size]"
                assert len(neg_src_emb.shape) == 2, \
                    "For joint negative sampler, in evaluation" \
                    "negative src/dst embs should in shape of " \
                    "[number_of_negs, dimension size]"
                neg_src_emb = neg_src_emb.reshape(1, neg_src.shape[0], -1)
                pos_dst_emb = pos_dst_emb.reshape(
                    pos_dst_emb.shape[0], 1, pos_dst_emb.shape[1])
                neg_score = calc_dot_pos_score(neg_src_emb, pos_dst_emb)
            else:
                assert False, f"Unknow negative sample type {neg_sample_type}"
            assert len(neg_score.shape) == 2
            neg_scores.append(neg_score)

        if neg_dst is not None:
            if neg_sample_type == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
                neg_dst_emb = emb[vtype][neg_dst.reshape(-1,)].to(device)
                neg_dst_emb = neg_dst_emb.reshape(
                    neg_dst.shape[0], neg_dst.shape[1], -1)
                # uniform sampled negative samples
                pos_src_emb = pos_src_emb.reshape(
                    pos_src_emb.shape[0], 1, pos_src_emb.shape[1])
                neg_score = calc_dot_pos_score(pos_src_emb, neg_dst_emb)
            elif neg_sample_type == BUILTIN_LP_JOINT_NEG_SAMPLER:
                neg_dst_emb = emb[vtype][neg_dst].to(device)
                # joint sampled negative samples
                assert len(pos_src_emb.shape) == 2, \
                    "For joint negative sampler, in evaluation " \
                    "positive src/dst embs should in shape of" \
                    "[eval_batch_size, dimension size]"
                assert len(neg_dst_emb.shape) == 2, \
                    "For joint negative sampler, in evaluation" \
                    "negative src/dst embs should in shape of " \
                    "[number_of_negs, dimension size]"
                pos_src_emb = pos_src_emb.reshape(
                    pos_src_emb.shape[0], 1, pos_src_emb.shape[1])
                neg_dst_emb = neg_dst_emb.reshape(1, neg_dst.shape[0], -1)
                neg_score = calc_dot_pos_score(pos_src_emb, neg_dst_emb)
            else:
                assert False, f"Unknow negative sample type {neg_sample_type}"
            assert len(neg_score.shape) == 2
            neg_scores.append(neg_score)

        neg_scores = th.cat(neg_scores, dim=-1).detach().cpu()
        pos_scores = pos_scores.detach().cpu()
        scores[canonical_etype] = (pos_scores, neg_scores)
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
    etypes : list of tuples
        The canonical edge types of the graph
    h_dim : int
        The hidden dimension
    gamma : float
        The gamma value for initialization
    """
    def __init__(self,
                 etypes,
                 h_dim,
                 gamma=40.):
        super(LinkPredictDistMultDecoder, self).__init__()
        self.num_rels = len(etypes)
        self.h_dim = h_dim
        self.etype2rid = {etype: i for i, etype in enumerate(etypes)}
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

            for canonical_etype in g.canonical_etypes:
                if g.num_edges(canonical_etype) == 0:
                    continue # the block might contain empty edge types

                i = self.etype2rid[canonical_etype]
                self.trained_rels[i] += 1
                rel_embedding = self._w_relation(th.tensor(i).to(self._w_relation.weight.device))
                rel_embedding = rel_embedding.unsqueeze(dim=1)
                src_type, _, dest_type = canonical_etype
                u, v = g.edges(etype=canonical_etype)
                src_emb = h[src_type][u]

                dest_emb = h[dest_type][v]
                rel_embedding = rel_embedding.repeat(1,dest_emb.shape[0]).T
                scores_etype = calc_distmult_pos_score(src_emb, dest_emb, rel_embedding)
                scores.append(scores_etype)
            scores=th.cat(scores)
            return scores

    def calc_test_scores(self, emb, pos_neg_tuple, neg_sample_type, device):
        """ Compute scores for positive edges and negative edges

        Parameters
        ----------
        emb: dict of Tensor
            Node embeddings.
        pos_neg_tuple: dict of tuple
            Positive and negative edges stored in a tuple:
            tuple(positive source, negative source,
            postive destination, negatve destination).
            The positive edges: (positive source, positive desitnation)
            The negative edges: (positive source, negative desitnation) and
                                (negative source, positive desitnation)
        neg_sample_type: str
            Describe how negative samples are sampled.
                Uniform: For each positive edge, we sample K negative edges
                Joint: For one batch of positive edges, we sample
                       K negative edges
        device: th.device
            Device used to compute scores

        Return
        ------
        Dict of (Tensor, Tensor)
            Return a dictionary of edge type to
            (positive scores, negative scores)
        """
        assert isinstance(pos_neg_tuple, dict), \
            "DistMulti is only applicable to heterogeneous graphs." \
            "Otherwise please use dot product decoder"
        scores = {}
        for canonical_etype, (pos_src, neg_src, pos_dst, neg_dst) in pos_neg_tuple.items():
            utype, _, vtype = canonical_etype
            # pos score
            pos_src_emb = emb[utype][pos_src]
            pos_dst_emb = emb[vtype][pos_dst]
            rid = self.etype2rid[canonical_etype]
            rel_embedding = self._w_relation(
                th.tensor(rid).to(self._w_relation.weight.device))
            pos_scores = calc_distmult_pos_score(
                pos_src_emb, pos_dst_emb, rel_embedding, device)
            neg_scores = []

            if neg_src is not None:
                neg_src_emb = emb[utype][neg_src.reshape(-1,)]
                if neg_sample_type == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
                    neg_src_emb = neg_src_emb.reshape(neg_src.shape[0], neg_src.shape[1], -1)
                    # uniform sampled negative samples
                    pos_dst_emb = pos_dst_emb.reshape(
                        pos_dst_emb.shape[0], 1, pos_dst_emb.shape[1])
                    rel_embedding = rel_embedding.reshape(
                        1, 1, rel_embedding.shape[-1])
                    neg_score = calc_distmult_pos_score(
                        neg_src_emb, rel_embedding, pos_dst_emb, device)
                elif neg_sample_type == BUILTIN_LP_JOINT_NEG_SAMPLER:
                    # joint sampled negative samples
                    assert len(pos_dst_emb.shape) == 2, \
                        "For joint negative sampler, in evaluation" \
                        "positive src/dst embs should in shape of" \
                        "[eval_batch_size, dimension size]"
                    assert len(neg_src_emb.shape) == 2, \
                        "For joint negative sampler, in evaluation" \
                        "negative src/dst embs should in shape of " \
                        "[number_of_negs, dimension size]"
                    neg_score = calc_distmult_neg_head_score(
                        neg_src_emb, pos_dst_emb, rel_embedding,
                        1, pos_dst_emb.shape[0], neg_src_emb.shape[0],
                        device)
                    # shape (batch_size, num_negs)
                    neg_score = neg_score.reshape(-1, neg_src_emb.shape[0])
                else:
                    assert False, f"Unknow negative sample type {neg_sample_type}"
                assert len(neg_score.shape) == 2
                neg_scores.append(neg_score)

            if neg_dst is not None:
                if neg_sample_type == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
                    neg_dst_emb = emb[vtype][neg_dst.reshape(-1,)]
                    neg_dst_emb = neg_dst_emb.reshape(neg_dst.shape[0], neg_dst.shape[1], -1)
                    # uniform sampled negative samples
                    pos_src_emb = pos_src_emb.reshape(
                        pos_src_emb.shape[0], 1, pos_src_emb.shape[1])
                    rel_embedding = rel_embedding.reshape(
                        1, 1, rel_embedding.shape[-1])
                    neg_score = calc_distmult_pos_score(
                        pos_src_emb, rel_embedding, neg_dst_emb, device)
                elif neg_sample_type == BUILTIN_LP_JOINT_NEG_SAMPLER:
                    neg_dst_emb = emb[vtype][neg_dst]
                    # joint sampled negative samples
                    assert len(pos_src_emb.shape) == 2, \
                        "For joint negative sampler, in evaluation " \
                        "positive src/dst embs should in shape of" \
                        "[eval_batch_size, dimension size]"
                    assert len(neg_dst_emb.shape) == 2, \
                        "For joint negative sampler, in evaluation" \
                        "negative src/dst embs should in shape of " \
                        "[number_of_negs, dimension size]"
                    neg_score = calc_distmult_neg_tail_score(
                        pos_src_emb, neg_dst_emb, rel_embedding,
                        1, pos_src_emb.shape[0], neg_dst_emb.shape[0],
                        device)
                    # shape (batch_size, num_negs)
                    neg_score = neg_score.reshape(-1, neg_dst_emb.shape[0])
                else:
                    assert False, f"Unknow negative sample type {neg_sample_type}"
                assert len(neg_score.shape) == 2
                neg_scores.append(neg_score)
            neg_scores = th.cat(neg_scores, dim=-1).detach().cpu()
            pos_scores = pos_scores.detach().cpu()
            scores[canonical_etype] = (pos_scores, neg_scores)

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

def _get_edge_weight(g, weight_field, etype):
    """ Get the edge weight feature from g according to etype.
        If the corresponding edge type does not have edge weight, set the weight to 1.

        Parameters
        ----------
        g: DGLGraph
            Graph.
        weight_field: str
            Edge weight feature field in a graph
        etype: (str, str, str)
            Canonical etype
    """
    # edge_weight_fields is a str
    if weight_field in g.edges[etype].data:
        eid = g.edges(form="eid", etype=etype)
        weight = g.edges[etype].data[weight_field][eid]
        weight = weight.flatten()
        assert len(weight) == len(eid), \
                "Edge weight must be a tensor of shape (num_edges,) " \
            f"or (num_edges, 1). But get {g.edges[etype].data[weight_field].shape}"
    else:
        # current etype does not has weight
        weight = th.ones((g.num_edges(etype),))
    return weight

class LinkPredictWeightedDistMultDecoder(LinkPredictDistMultDecoder):
    """Link prediction decoder with the score function of DistMult
       with edge weight.

       When computing loss, edge weights are used to adjust the loss
    """
    def __init__(self, etypes, h_dim, gamma=40., edge_weight_fields=None):
        self._edge_weight_fields = edge_weight_fields
        super(LinkPredictWeightedDistMultDecoder, self).__init__(etypes, h_dim, gamma)

    def forward(self, g, h):
        """Forward function.

        This computes the DistMult score on every edge type.
        """
        with g.local_scope():
            scores=[]
            weights = []

            for canonical_etype in g.canonical_etypes:
                if g.num_edges(canonical_etype) == 0:
                    continue # the block might contain empty edge types

                i = self.etype2rid[canonical_etype]
                self.trained_rels[i] += 1
                rel_embedding = self._w_relation(th.tensor(i).to(self._w_relation.weight.device))
                rel_embedding = rel_embedding.unsqueeze(dim=1)
                src_type, _, dest_type = canonical_etype
                u, v = g.edges(etype=canonical_etype)
                src_emb = h[src_type][u]

                dest_emb = h[dest_type][v]
                rel_embedding = rel_embedding.repeat(1,dest_emb.shape[0]).T
                scores_etype = calc_distmult_pos_score(src_emb, dest_emb, rel_embedding)

                weight = _get_edge_weight(g, LP_DECODER_EDGE_WEIGHT, canonical_etype)
                weights.append(weight.to(scores_etype.device))
                scores.append(scores_etype)
            scores = th.cat(scores)
            weights = th.cat(weights)
            return (scores, weights)

class LinkPredictWeightedDotDecoder(LinkPredictDotDecoder):
    """Link prediction decoder with the score function of dot product
       with edge weight.

       When computing loss, edge weights are used to adjust the loss
    """
    def __init__(self, in_dim, edge_weight_fields):
        self._edge_weight_fields = edge_weight_fields
        super(LinkPredictWeightedDotDecoder, self).__init__(in_dim)

    def forward(self, g, h): # pylint: disable=arguments-differ
        """Forward function.

        This computes the dot product score on every edge type.
        """
        with g.local_scope():
            scores = []
            weights = []

            for canonical_etype in g.canonical_etypes:
                if g.num_edges(canonical_etype) == 0:
                    continue # the block might contain empty edge types

                src_type, _, dest_type = canonical_etype
                u, v = g.edges(etype=canonical_etype)
                src_emb = h[src_type][u]
                dest_emb = h[dest_type][v]
                scores_etype = calc_dot_pos_score(src_emb, dest_emb)

                weight = _get_edge_weight(g, LP_DECODER_EDGE_WEIGHT, canonical_etype)
                weights.append(weight.to(scores_etype.device))
                scores.append(scores_etype)

            scores = th.cat(scores)
            weights = th.cat(weights)
            return (scores, weights)
