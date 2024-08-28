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
import abc
import logging
import numpy as np
import torch as th
from torch import nn

from ..utils import get_backend, is_distributed
from .ngnn_mlp import NGNNMLP
from .gs_layer import GSLayer, GSLayerNoParam
from ..dataloading import (BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                           BUILTIN_LP_JOINT_NEG_SAMPLER,
                           BUILTIN_LP_FIXED_NEG_SAMPLER)

from ..eval.utils import calc_distmult_pos_score, calc_dot_pos_score, calc_rotate_pos_score
from ..eval.utils import (calc_distmult_neg_head_score,
                          calc_distmult_neg_tail_score,
                          calc_rotate_neg_head_score,
                          calc_rotate_neg_tail_score)


# TODO(zhengda) we need to split it into classifier and regression.
class GSEdgeDecoder(GSLayer):
    """ The abstract class of a GraphStorm edge decoder
    """
    @abc.abstractmethod
    def forward(self, g, h, e_h=None):
        """Forward function.

        Compute logits for each pair ``(ufeat[i], ifeat[i])``. The target
        edges are stored in g.

        Parameters
        ----------
        g: DGLGraph
            The graph that stores target edges to run edge prediction.
        h: dict of Tensors
            The dictionary containing the embeddings
        e_h: dict of tensors
            The dictionary containing the edge features for g.
        Returns
        -------
        th.Tensor
            Predicting scores for each edge in g.
            Shape: (B, num_classes) for classification
            Shape: (B, ) for regression
        """

    @abc.abstractmethod
    def predict(self, g, h, e_h=None):
        """predict function for this decoder

        Parameters
        ----------
        g: DGLGraph
            The graph that stores target edges to run edge prediction.
        h: dict of Tensors
            The dictionary containing the node embeddings.
        e_h: dict of tensors
            The dictionary containing the edge features for the ``g``.

        Returns
        -------
        Tensor: the maximum score of each edge.
        """

    @abc.abstractmethod
    def predict_proba(self, g, h, e_h=None):
        """predict function for this decoder

        Parameters
        ----------
        g: DGLGraph
            The graph that stores target edges to run edge prediction.
        h: dict of Tensors
            The dictionary containing the node embeddings.
        e_h: dict of tensors
            The dictionary containing the edge features for the ``g``.

        Returns
        -------
        Tensor: all the scores of each edge.
        """

class DenseBiDecoder(GSEdgeDecoder):
    r""" Dense bi-linear decoder for edge prediction tasks.

    ``DenseBiDecoder`` is the dense implementation of the bi-linear decoder used in GCMC.
    Suitable when the graph can be represented by a pair of lists (one for source
    node list and one for destination node list).

    Parameters
    ----------
    in_units: int
        The input dimension size. It is the dimension for both source and destination node.
        embeddings.
    num_classes: int
        Number of classes. For classification tasks only.
    multilabel: bool
        Whether this is a multi-label classification decoder.
    num_basis: int
        Number of basis. Default: 2.
    dropout_rate: float
        Dropout rate. Default: 0.
    target_etype: tuple of str
        The target etype for prediction in the format of (src_ntype, etype, dst_ntype).
    regression: bool
        Whether this decoder is for regression tasks. Default: False.
    norm: str
        Normalization methods. Not used, but reserved for complex DenseBiDecoder child class
        implementation. Default: None.
    """
    def __init__(self,
                 in_units,
                 num_classes,
                 multilabel,
                 target_etype,
                 num_basis=2,
                 dropout_rate=0.0,
                 regression=False,
                 norm=None):
        super().__init__()

        self.in_units = in_units
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.num_basis = num_basis
        self.regression = regression
        self.dropout = dropout_rate
        # TODO(xiangsx): The norm is not used here.
        self.norm = norm
        # TODO support multi target etypes
        # In the future we can accept both tuple and list of tuple
        assert isinstance(target_etype, tuple) and len(target_etype) == 3, \
            "Target etype must be a tuple of a canonical etype."
        self.target_etype = target_etype

        self._init_model()

    def _init_model(self):
        """ Init decoder model
        """
        if self.norm is not None:
            logging.warning("Embedding normalization (batch norm or layer norm) "
                            "is not supported in DenseBiDecoder")
        basis_out = self.in_units if self.regression else self.num_classes

        self.dropout = nn.Dropout(self.dropout)
        self.basis_para = nn.Parameter(
            th.randn(self.num_basis, self.in_units, self.in_units))
        self.combine_basis = nn.Linear(self.num_basis, basis_out, bias=False)
        self.reset_parameters()

        if self.regression:
            self.regression_head = nn.Linear(basis_out, 1, bias=True)

    def reset_parameters(self):
        """Reset all parameters in this decoder with xavier uniform method
        """
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    # pylint: disable=unused-argument
    def forward(self, g, h, e_h=None):
        """ Dense bi-linear edge decoder forward computation.

        Parameters
        ----------
        g: DGLGraph
            The graph of target edges.
        h: dict of Tensor
            The input node embeddings in the format of {ntype: emb}.
        e_h: dict of Tensor
            The input edge embeddings in the format of {(src_ntype, etype, dst_ntype): emb}.
            Not used, but reserved for future support of edge embeddings. Default: None.

        Returns
        -------
        out: Tensor
            The prediction results.
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

    # pylint: disable=unused-argument
    def predict(self, g, h, e_h=None):
        """ Dense bi-linear edge decoder predict computation.

        Parameters
        ----------
        g: DGLGraph
            The graph of target edges.
        h: dict of Tensor
            The input node embeddings in the format of {ntype: emb}.
        e_h: dict of Tensor
            The input edge embeddings in the format of {(src_ntype, etype, dst_ntype): emb}.
            Not used, but reserved for future support of edge embeddings. Default: None.

        Returns
        -------
        out: Tensor
            The prediction results.
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
            elif self.multilabel:
                out = (th.sigmoid(out) > .5).long()
            else:  # not multilabel
                out = out.argmax(dim=1)
        return out

    # pylint: disable=unused-argument
    def predict_proba(self, g, h, e_h=None):
        """ Dense bi-linear edge decoder predict computation and return the normalized
        prediction results if this decoder is for edge classification.

        Parameters
        ----------
        g: DGLGraph
            The graph of target edges.
        h: dict of Tensor
            The input node embeddings in the format of {ntype: emb}.
        e_h: dict of Tensor
            The input edge embeddings in the format of {(src_ntype, etype, dst_ntype): emb}.
            Not used, but reserved for future support of edge embeddings. Default: None.

        Returns
        -------
        out: Tensor
            The prediction results. If this decoder is for edge classification, return the
            normalized prediction results.
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
            elif self.multilabel:
                out = th.sigmoid(out)
            else:
                out = th.softmax(out, 1)
        return out

    @property
    def in_dims(self):
        """ Return the input dimension size, which is given in class initialization.
        """
        return self.in_units

    @property
    def out_dims(self):
        """ Return the output dimension size. If this decoder is for edge regression,
        will return ``1``.
        """
        return 1 if self.regression else self.num_classes


class MLPEdgeDecoder(GSEdgeDecoder):
    """ MLP-based decoder for edge prediction tasks.

    Parameters
    ----------
    h_dim: int
        The input dimension size. It is the dimension for both source and destination
        node embeddings.
    out_dim: int
        Output dimension size. If this decoder is for edge regression, the output
        dimension should be ``1``.
    multilabel: bool
        Whether this decoder is for multilabel edge classification.
    target_etype: tuple of str
        The target etype for prediction in the format of (src_ntype, etype, dst_ntype).
    num_hidden_layers: int
        Number of MLP layers. Default: 1.
    dropout: float
        Dropout rate. Default: 0.
    regression: bool
        Whether this decoder is for regression tasks. Default: False.
    num_ffn_layers: int
        Number of FFN layers added to the decoder. Default: 0
    norm: str
        Normalization methods. Not used, but reserved for complex MLPEdgeDecoder child class
        implementation. Default: None.
    """
    def __init__(self,
                 h_dim,
                 out_dim,
                 multilabel,
                 target_etype,
                 num_hidden_layers=1,
                 dropout=0,
                 regression=False,
                 num_ffn_layers=0,
                 norm=None):
        super(MLPEdgeDecoder, self).__init__()
        self.h_dim = h_dim
        self.multilabel = multilabel
        self.out_dim = h_dim if regression else out_dim
        self.regression = regression
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_ffn_layers = num_ffn_layers
        # TODO(xiangsx): The norm is not used here.
        self.norm = norm
        # TODO support multi target etypes
        # In the future we can accept both tuple and list of tuple
        assert isinstance(target_etype, tuple) and len(target_etype) == 3, \
            "Target etype must be a tuple of a canonical etype."
        self.target_etype = target_etype

        self._init_model()

    def _init_model(self):
        """ Init decoder model
        """
        if self.norm is not None:
            logging.warning("Embedding normalization (batch norm or layer norm) "
                            "is not supported in MLPEdgeDecoder")
        # ngnn layer
        self.ngnn_mlp = NGNNMLP(self.h_dim * 2, self.h_dim * 2,
                                self.num_ffn_layers,
                                th.nn.functional.relu,
                                self.dropout)

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
            g: DGLBlock
                The minibatch graph
            h: dict of Tensors
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
            if self.num_ffn_layers > 0:
                h = self.ngnn_mlp(h)
            out = th.matmul(h, self.decoder)
        return out

    # pylint: disable=unused-argument
    def forward(self, g, h, e_h=None):
        """ MLP-based edge decoder forward computation.

        Parameters
        ----------
        g: DGLGraph
            The graph of target edges.
        h: dict of Tensor
            The input node embeddings in the format of {ntype: emb}.
        e_h: dict of Tensor
            The input edge embeddings in the format of {(src_ntype, etype, dst_ntype): emb}.
            Not used, but reserved for future support of edge embeddings. Default: None.

        Returns
        -------
        out: Tensor
            The prediction results.
        """
        out = self._compute_logits(g, h)

        if self.regression:
            out = self.regression_head(out)
        return out

    # pylint: disable=unused-argument
    def predict(self, g, h, e_h=None):
        """ MLP-based edge decoder predict computation.

        Parameters
        ----------
        g: DGLGraph
            The graph of target edges.
        h: dict of Tensor
            The input node embeddings in the format of {ntype: emb}.
        e_h: dict of Tensor
            The input edge embeddings in the format of {(src_ntype, etype, dst_ntype): emb}.
            Not used, but reserved for future support of edge embeddings. Default: None.

        Returns
        -------
        out: Tensor
            The prediction results.
        """
        out = self._compute_logits(g, h)

        if self.regression:
            out = self.regression_head(out)
        elif self.multilabel:
            out = (th.sigmoid(out) > .5).long()
        else:  # not multilabel
            out = out.argmax(dim=1)
        return out

    # pylint: disable=unused-argument
    def predict_proba(self, g, h, e_h=None):
        """ MLP-based edge decoder prediction computation and return the normalized
        prediction results if this decoder is for edge classification.

        Parameters
        ----------
        g: DGLGraph
            The graph of target edges.
        h: dict of Tensor
            The input node embeddings in the format of {ntype: emb}.
        e_h: dict of Tensor
            The input edge embeddings in the format of {(src_ntype, etype, dst_ntype): emb}.
            Not used, but reserved for future support of edge embeddings. Default: None.

        Returns
        -------
        out: Tensor
            The prediction results. If this decoder is for edge classification, return the
            normalized prediction results.
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
        """ Return the input dimension size, which is given in class initialization.
        """
        return self.h_dim

    @property
    def out_dims(self):
        """ Return the output dimension size. If this decoder is for edge regression,
        will return ``1``.
        """
        return 1 if self.regression else self.out_dim

class MLPEFeatEdgeDecoder(MLPEdgeDecoder):
    """ MLP-based decoder for edge prediction tasks with edge features supported.

    Parameters
    ----------
    h_dim: int
        The input dimension size. It is the dimension for both source and destination
        node embeddings.
    feat_dim: int
        The input dimension size of edge features which are used for computing decoder output.
    out_dim: int
        Output dimension size. If this decoder is for edge regression, the output
        dimension should be ``1``.
    multilabel: bool
        Whether this decoder is for multilabel edge classification.
    target_etype: tuple of str
        The target etype for prediction in the format of (src_ntype, etype, dst_ntype).
    dropout: float
        Dropout rate. Default: 0.
    regression: bool
        Whether this decoder is for regression tasks. Default: False.
    num_ffn_layers: int
        Number of FFN layers added to the decoder. Default: 0
    norm: str
        Normalization methods. Not used, but reserved for complex MLPEFeatEdgeDecoder child
        class implementation. Default: None.
    """
    def __init__(self,
                 h_dim,
                 feat_dim,
                 out_dim,
                 multilabel,
                 target_etype,
                 dropout=0,
                 regression=False,
                 num_ffn_layers=0,
                 norm=None):
        self.feat_dim = feat_dim
        super(MLPEFeatEdgeDecoder, self).__init__(h_dim=h_dim,
                                                  out_dim=out_dim,
                                                  multilabel=multilabel,
                                                  target_etype=target_etype,
                                                  dropout=dropout,
                                                  regression=regression,
                                                  num_ffn_layers=num_ffn_layers,
                                                  norm=norm)

    def _init_model(self):
        """ Init decoder model
        """
        self.relu = th.nn.ReLU()

        # [src_emb | dest_emb] @ W -> h_dim
        # Here we assume the source and destination nodes have the same dimension.
        self.nn_decoder = nn.Parameter(th.randn(self.h_dim * 2, self.h_dim))
        # [edge_feat] @ W -> h_dim
        self.feat_decoder = nn.Parameter(th.randn(self.feat_dim, self.h_dim))

        # ngnn before combine layer
        self.ngnn_mlp = NGNNMLP(self.h_dim * 2, self.h_dim * 2,
                                self.num_ffn_layers,
                                th.nn.functional.relu,
                                self.dropout)

        # combine output of nn_decoder and feat_decoder
        self.combine_decoder = nn.Parameter(th.randn(self.h_dim * 2, self.h_dim))
        self.decoder = nn.Parameter(th.randn(self.h_dim, self.out_dim))
        self.dropout = nn.Dropout(self.dropout)

        self.nn_decoder_norm = None
        self.feat_decoder_norm = None
        self.combine_norm = None
        if self.norm == "batch":
            self.feat_decoder_norm = nn.BatchNorm1d(self.h_dim)
            self.nn_decoder_norm = nn.BatchNorm1d(self.h_dim)
            self.combine_norm = nn.BatchNorm1d(self.h_dim)
        elif self.norm == "layer":
            self.feat_decoder_norm = nn.LayerNorm(self.h_dim)
            self.nn_decoder_norm = nn.LayerNorm(self.h_dim)
            self.combine_norm = nn.LayerNorm(self.h_dim)

        logging.debug("MLPEFeatEdgeDecoder with decoder %s norm", self.norm)

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

    # pylint: disable=arguments-differ
    def _compute_logits(self, g, h, e_h):
        """ Compute forword output

            Parameters
            ----------
            g: DGLBlock
                The minibatch graph
            h: dict of Tensors
                The dictionary containing the embeddings
            Returns
            -------
            th.Tensor
                Output of forward
        """
        assert e_h is not None, "edge feature is required"
        with g.local_scope():
            u, v = g.edges(etype=self.target_etype)
            src_type, _, dest_type = self.target_etype
            ufeat = h[src_type][u]
            ifeat = h[dest_type][v]
            efeat = e_h[self.target_etype]

            # [src_emb | dest_emb] @ W -> h_dim
            h = th.cat([ufeat, ifeat], dim=1)
            nn_h = th.matmul(h, self.nn_decoder)
            if self.nn_decoder_norm is not None:
                nn_h = self.nn_decoder_norm(nn_h)
            nn_h = self.relu(nn_h)
            nn_h = self.dropout(nn_h)
            # [edge_feat] @ W -> h_dim
            feat_h = th.matmul(efeat, self.feat_decoder)
            if self.feat_decoder_norm is not None:
                feat_h = self.feat_decoder_norm(feat_h)
            feat_h = self.relu(feat_h)
            feat_h = self.dropout(feat_h)
            # [nn_h | feat_h] @ W -> h_dim
            combine_h = th.cat([nn_h, feat_h], dim=1)
            if self.num_ffn_layers > 0:
                combine_h = self.ngnn_mlp(combine_h)
            combine_h = th.matmul(combine_h, self.combine_decoder)
            if self.combine_norm is not None:
                combine_h = self.combine_norm(combine_h)
            combine_h = self.relu(combine_h)
            out = th.matmul(combine_h, self.decoder)

        return out

    # pylint: disable=signature-differs
    def forward(self, g, h, e_h):
        """ MLP-based edge feature supported edge decoder forward computation.

        Parameters
        ----------
        g: DGLGraph
            The graph of target edges.
        h: dict of Tensor
            The input node embeddings in the format of {ntype: emb}.
        e_h: dict of Tensor
            The input edge embeddings in the format of {(src_ntype, etype, dst_ntype): emb}.

        Returns
        -------
        out: Tensor
            The prediction results.
        """
        out = self._compute_logits(g, h, e_h)

        if self.regression:
            out = self.regression_head(out)
        return out

    # pylint: disable=signature-differs
    def predict(self, g, h, e_h):
        """ MLP-based edge feature supported edge decoder predict computation.

        Parameters
        ----------
        g: DGLGraph
            The graph of target edges.
        h: dict of Tensor
            The input node embeddings in the format of {ntype: emb}.
        e_h: dict of Tensor
            The input edge embeddings in the format of {(src_ntype, etype, dst_ntype): emb}.

        Returns
        -------
        out: Tensor
            The prediction results.
        """
        out = self._compute_logits(g, h, e_h)

        if self.regression:
            out = self.regression_head(out)
        elif self.multilabel:
            out = (th.sigmoid(out) > .5).long()
        else:  # not multilabel
            out = out.argmax(dim=1)
        return out

    # pylint: disable=signature-differs
    def predict_proba(self, g, h, e_h):
        """ MLP-based edge feature supported edge decoder predict computation and return
        the normalized prediction results if this decoder is for edge classification.

        Parameters
        ----------
        g: DGLGraph
            The graph of target edges.
        h: dict of Tensor
            The input node embeddings in the format of {ntype: emb}.
        e_h: dict of Tensor
            The input edge embeddings in the format of {(src_ntype, etype, dst_ntype): emb}.

        Returns
        -------
        out: Tensor
            The prediction results. If this decoder is for edge classification, return the
            normalized prediction results.
        """
        out = self._compute_logits(g, h, e_h)

        if self.regression:
            out = self.regression_head(out)
        elif self.multilabel:
            out = th.sigmoid(out)
        else:
            out = th.softmax(out, 1)
        return out

##################### Link Prediction Decoders #######################
class LinkPredictNoParamDecoder(GSLayerNoParam):
    """ Abstract class for Link prediction decoder without trainable parameters
    """

    # pylint: disable=arguments-differ
    @abc.abstractmethod
    def forward(self, g, h, e_h=None):
        """Link prediction decoder forward function.

        This computes the edge score on every edge type.

        Parameters
        ----------
        g: DGLGraph
            The target edge graph
        h: dict of Tensors
            The dictionary containing the node embeddings
        e_h: dict of tensors
            The dictionary containing the edge features for g.

        Returns
        -------
        dict of th.Tensor
            The scores for edges of each edge type
            in the input graph.
        """

class LinkPredictLearnableDecoder(GSLayer):
    """ Abstract class for Link prediction decoder with trainable parameters
    """

    # pylint: disable=arguments-differ
    @abc.abstractmethod
    def forward(self, g, h, e_h=None):
        """Link prediction decoder forward function.

        This computes the edge score on every edge type.

        Parameters
        ----------
        g: DGLGraph
            The target edge graph
        h: dict of Tensors
            The dictionary containing the node embeddings
        e_h: dict of tensors
            The dictionary containing the edge features for g.

        Returns
        -------
        dict of th.Tensor
            The scores for edges of each edge type
            in the input graph.
        """

class LinkPredictDotDecoder(LinkPredictNoParamDecoder):
    """ Decoder for link prediction using the dot product as the score function.

    Parameters
    ----------
    in_dim: int
        The input dimension size. It is the dimension for both source and destination
        node embeddings.
    """
    def __init__(self, in_dim):
        self._in_dim = in_dim

    # pylint: disable=unused-argument
    def forward(self, g, h, e_h=None):
        """ Link prediction dot product decoder forward function.

        This function computes the edge scores on all edge types of the input graph.

        Parameters
        ----------
        g: DGLGraph
            The input graph.
        h: dict of Tensor
            The input node embeddings in the format of {ntype: emb}.
        e_h: dict of Tensor
            The input edge embeddings in the format of {(src_ntype, etype, dst_ntype): emb}.
            Not used, but reserved for future support of edge embeddings. Default: None.

        Returns
        -------
        scores: dict of Tensor
            The scores for edges of all edge types in the input graph in the format of
            {(src_ntype, etype, dst_ntype): score}.
        """
        with g.local_scope():
            scores = {}

            for canonical_etype in g.canonical_etypes:
                if g.num_edges(canonical_etype) == 0:
                    continue # the block might contain empty edge types

                src_type, _, dest_type = canonical_etype
                u, v = g.edges(etype=canonical_etype)
                src_emb = h[src_type][u]
                dest_emb = h[dest_type][v]
                scores_etype = calc_dot_pos_score(src_emb, dest_emb)
                scores[canonical_etype] = scores_etype

            return scores

    def calc_test_scores(self, emb, pos_neg_tuple, neg_sample_type, device):
        """ Compute scores for positive edges and negative edges.

        Parameters
        ----------
        emb: dict of Tensor
            Node embeddings in the format of {ntype: emb}.
        pos_neg_tuple: dict of tuple
            Positive and negative edges stored in a dict of tuple in the format of
            {("src_ntype1", "etype1", "dst_ntype1" ): (pos_src_idx, neg_src_idx,
            pos_dst_idx, neg_dst_idx)}.

            The `pos_src_idx` represents the postive source node indexes in the format
            of Torch.Tensor. The `neg_src_idx` represents the negative source node indexes
            in the format of Torch.Tensor. The `pos_dst_idx` represents the postive destination
            node indexes in the format of Torch.Tensor. The `neg_dst_idx` represents the
            negative destination node indexes in the format of Torch.Tensor.

            We define positive and negative edges as:

            * The positive edges: (pos_src_idx, pos_dst_idx)
            * The negative edges: (pos_src_idx, neg_dst_idx) and
              (neg_src_idx, pos_dst_idx)

        neg_sample_type: str
            Describe how negative samples are sampled. There are two options:

            * ``Uniform``: For each positive edge, we sample K negative edges.
            * ``Joint``: For one batch of positive edges, we sample K negative edges.

        device: th.device
            Device used to compute scores.

        Returns
        --------
        scores: dict of tuple
            Return a dictionary of edge type's positive scores and negative scores in the format
            of {(src_ntype, etype, dst_ntype): (pos_scores, neg_scores)}
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
            if neg_sample_type in [BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                                   BUILTIN_LP_FIXED_NEG_SAMPLER]:
                # fixed negative sample is similar to uniform negative sample
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
            if neg_sample_type in [BUILTIN_LP_UNIFORM_NEG_SAMPLER, \
                                   BUILTIN_LP_FIXED_NEG_SAMPLER]:
                # fixed negative sample is similar to uniform negative sample
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

        neg_scores = th.cat(neg_scores, dim=-1).detach()
        # gloo with cpu will consume less GPU memory
        neg_scores = neg_scores.cpu() \
            if is_distributed() and get_backend() == "gloo" \
            else neg_scores
        pos_scores = pos_scores.detach()
        pos_scores = pos_scores.cpu() \
            if is_distributed() and get_backend() == "gloo" \
            else pos_scores
        scores[canonical_etype] = (pos_scores, neg_scores)
        return scores

    @property
    def in_dims(self):
        """ Return the input dimension size, which is given in class initialization.
        """
        return self._in_dim

    @property
    def out_dims(self):
        """ Return ``1`` for link prediction tasks.
        """
        return 1

class LinkPredictContrastiveDotDecoder(LinkPredictDotDecoder):
    """ Decoder for link prediction designed for contrastive loss by
    using the dot product as the score function.

    Note:
    -----
    This class is specifically implemented for contrastive loss. But
    it could also be used by other pair-wise loss functions for link
    prediction tasks.

    Parameters
    ----------
    in_dim: int
        The input dimension size. It is the dimension for both source and destination
        node embeddings.
    """
    # TODO(xiang): Develop a better solution for supporting pair-wise
    # loss functions in link prediction tasks. The
    # LinkPredictContrastiveDotDecoder is implemented based on the
    # assumption that the same decoder.forward will be called twice
    # with a positive graph and negative graph respectively. And
    # the positive and negative graphs are compatible. We can simply
    # sort the edges in postive and negative graphs to create <pos, neg>
    # pairs. This implementation makes strong assumption of the correlation
    # between the Dataloader, Decoder and the Loss function. We should
    # find a better implementation.

    # pylint: disable=unused-argument
    def forward(self, g, h, e_h=None):
        with g.local_scope():
            scores = {}

            for canonical_etype in g.canonical_etypes:
                if g.num_edges(canonical_etype) == 0:
                    continue # the block might contain empty edge types

                src_type, _, dest_type = canonical_etype
                u, v = g.edges(etype=canonical_etype)
                # Sort edges according to source node ids
                # The same function is invoked by computing both pos scores
                # and neg scores, by sorting edges according to source nids
                # the output scores of pos_score and neg_score are compatible.
                #
                # For example:
                #
                # pos pairs   |  neg pairs
                # (10, 20)    |  (10, 3), (10, 1), (10, 0), (10, 22)
                # (13, 6)     |  (13, 3), (13, 1), (13, 0), (13, 22)
                # (29, 8)     |  (29, 3), (29, 1), (29, 0), (29, 22)
                # TODO: use stable to keep the order of negatives. This may not
                # be necessary.
                u_sort_idx = th.argsort(u, stable=True)
                u = u[u_sort_idx]
                v = v[u_sort_idx]
                src_emb = h[src_type][u]
                dest_emb = h[dest_type][v]
                scores_etype = calc_dot_pos_score(src_emb, dest_emb)
                scores[canonical_etype] = scores_etype

            return scores

class LinkPredictMultiRelationLearnableDecoder(LinkPredictLearnableDecoder):
    """ Decoder for link prediction with multiple relation types (relation embeddings).

    Parameters
    ----------
    etypes: list of tuples
        The canonical edge types of the graph in the format of
        [(src_ntype1, etype1, dst_ntype1), ...]
    h_dim: int
        The input dimension size. It is the dimension for both source and destination
        node embeddings.
    gamma: float
        The gamma value for model weight initialization. Default: 40.
    """
    def __init__(self,
                 etypes,
                 h_dim,
                 gamma=40.):
        super(LinkPredictMultiRelationLearnableDecoder, self).__init__()
        self.num_rels = len(etypes)
        self.h_dim = h_dim
        self.etype2rid = {etype: i for i, etype in enumerate(etypes)}
        self.relids = th.arange(self.num_rels)

        # used to track whether a relation embedding is trained.
        self.trained_rels = np.zeros(self.num_rels)
        self.gamma = gamma
        self.init_w_relation()

    # pylint: disable=arguments-differ
    @abc.abstractmethod
    def init_w_relation(self):
        """ Initialize learnable relation embeddings.

            An example:

            .. code:: python

                def init_w_relation(self):
                    self._w_relation = nn.Embedding(self.num_rels, self.h_dim)

                    nn.init.uniform_(self._w_relation.weight, -1., 1.)
        """

    def get_relemb(self, etype):
        """ Retrieve trained embedding of the given edge type.

        Parameters
        ----------
        etype: tuple
            An edge type in the format of (src_ntype, etype, dst_ntype).

        Returns
        -------
        Tensor: the output embeddings of the given edge type.
        """
        i = self.etype2rid[etype]
        assert self.trained_rels[i] > 0, 'The relation {} is not trained'.format(etype)
        return self._w_relation(th.tensor(i).to(self._w_relation.weight.device))

    def get_relembs(self):
        """ Retrieve all edge types' trained weights and the edge type ID mapping.

        Returns
        -------
        dict of Tensor: the trained weights (relation embeddings) of all edge types.
        dict of int: edge type ID mapping in the format of
        {((src_ntype1, etype1, dst_ntype1)): id}
        """
        return self._w_relation.weight, self.etype2rid

class LinkPredictRotatEDecoder(LinkPredictMultiRelationLearnableDecoder):
    r""" Decoder for link prediction using the RotatE as the score function.

        Score function of RotateE measures the angular distance between
        head and tail elements. The angular distance is defined as:

        .. math:

            d_r(h, t)=\|h\circ r-t\|

        The RotatE score function is defined as:

        .. math:

            gamma - \|h\circ r-t\|^2

        where gamma is a margin.

        For more details, please refer to https://arxiv.org/abs/1902.10197
        or https://dglke.dgl.ai/doc/kg.html#rotatee.

        Note: The relation embedding of RotatE has two parts,
        one for real numbers and one for complex numbers.
        Each has the dimension size as half of the input dimension size.

    Parameters
    ----------
    etypes: list of tuples
        The canonical edge types of the graph in the format of
        [(src_ntype1, etype1, dst_ntype1), ...]
    h_dim: int
        The input dimension size. It is the dimension for both source and destination
        node embeddings.
    gamma: float
        The gamma value for model initialization and score function. Default: 4.
    """
    def __init__(self,
                 etypes,
                 h_dim,
                 gamma=4.):
        self.rel_dim = h_dim // 2
        super(LinkPredictRotatEDecoder, self).__init__(etypes, h_dim, gamma)

    def init_w_relation(self):
        self._w_relation = nn.Embedding(self.num_rels, self.rel_dim)
        self.emb_init = self.gamma / self.rel_dim
        nn.init.uniform_(self._w_relation.weight, -self.emb_init, self.emb_init)

    # pylint: disable=unused-argument
    def forward(self, g, h, e_h=None):
        """ Link prediction decoder forward function using the RotatE as the score function.

        This computes the edge score on every edge type.

        Parameters
        ----------
        g: DGLGraph
            The input graph.
        h: dict of Tensor
            The input node embeddings in the format of {ntype: emb}.
        e_h: dict of Tensor
            The input edge embeddings in the format of {(src_ntype, etype, dst_ntype): emb}.
            Not used, but reserved for future support of edge embeddings. Default: None.

        Returns
        -------
        scores: dict of Tensor
            The scores for edges of all edge types in the input graph in the format of
            {(src_ntype, etype, dst_ntype): score}.
        """
        with g.local_scope():
            scores = {}

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
                scores_etype = calc_rotate_pos_score(src_emb,
                                                     dest_emb,
                                                     rel_embedding,
                                                     self.emb_init,
                                                     self.gamma)
                scores[canonical_etype] = scores_etype

            return scores

    def calc_test_scores(self, emb, pos_neg_tuple, neg_sample_type, device):
        """ Compute scores for positive edges and negative edges.

        Parameters
        ----------
        emb: dict of Tensor
            Node embeddings in the format of {ntype: emb}.
        pos_neg_tuple: dict of tuple
            Positive and negative edges stored in a dict of tuple in the format of
            {("src_ntype1", "etype1", "dst_ntype1" ): (pos_src_idx, neg_src_idx,
            pos_dst_idx, neg_dst_idx)}.

            The `pos_src_idx` represents the postive source node indexes in the format
            of Torch.Tensor. The `neg_src_idx` represents the negative source node indexes
            in the format of Torch.Tensor. The `pos_dst_idx` represents the postive destination
            node indexes in the format of Torch.Tensor. The `neg_dst_idx` represents the
            negative destination node indexes in the format of Torch.Tensor.

            We define positive and negative edges as:

            * The positive edges: (pos_src_idx, pos_dst_idx)
            * The negative edges: (pos_src_idx, neg_dst_idx) and
              (neg_src_idx, pos_dst_idx)

        neg_sample_type: str
            Describe how negative samples are sampled. There are two options:

            * ``Uniform``: For each positive edge, we sample K negative edges.
            * ``Joint``: For one batch of positive edges, we sample K negative edges.

        device: th.device
            Device used to compute scores.

        Returns
        --------
        scores: dict of tuple
            Return a dictionary of edge type's positive scores and negative scores in the format
            of {(src_ntype, etype, dst_ntype): (pos_scores, neg_scores)}.
        """
        assert isinstance(pos_neg_tuple, dict), \
            "RotatE is only applicable to heterogeneous graphs." \
            "Otherwise please use dot product decoder."
        scores = {}
        for canonical_etype, (pos_src, neg_src, pos_dst, neg_dst) in pos_neg_tuple.items():
            utype, _, vtype = canonical_etype
            # pos score
            pos_src_emb = emb[utype][pos_src]
            pos_dst_emb = emb[vtype][pos_dst]
            rid = self.etype2rid[canonical_etype]
            rel_embedding = self._w_relation(
                th.tensor(rid).to(self._w_relation.weight.device))
            pos_scores = calc_rotate_pos_score(pos_src_emb,
                                               pos_dst_emb,
                                               rel_embedding,
                                               self.emb_init,
                                               self.gamma,
                                               device)
            neg_scores = []

            if neg_src is not None:
                neg_src_emb = emb[utype][neg_src.reshape(-1,)]
                if neg_sample_type in [BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                                       BUILTIN_LP_FIXED_NEG_SAMPLER]:
                    # fixed negative sample is similar to uniform negative sample
                    neg_src_emb = neg_src_emb.reshape(neg_src.shape[0], neg_src.shape[1], -1)
                    # uniform sampled negative samples
                    pos_dst_emb = pos_dst_emb.reshape(
                        pos_dst_emb.shape[0], 1, pos_dst_emb.shape[1])
                    rel_embedding = rel_embedding.reshape(
                        1, 1, rel_embedding.shape[-1])
                    neg_score = calc_rotate_pos_score(neg_src_emb,
                                                      pos_dst_emb,
                                                      rel_embedding,
                                                      self.emb_init,
                                                      self.gamma,
                                                      device)
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
                    neg_score = calc_rotate_neg_head_score(
                        neg_src_emb, pos_dst_emb, rel_embedding,
                        1, pos_dst_emb.shape[0], neg_src_emb.shape[0],
                        self.emb_init, self.gamma,
                        device)
                    # shape (batch_size, num_negs)
                    neg_score = neg_score.reshape(-1, neg_src_emb.shape[0])
                else:
                    assert False, f"Unknow negative sample type {neg_sample_type}"
                assert len(neg_score.shape) == 2
                neg_scores.append(neg_score)

            if neg_dst is not None:
                if neg_sample_type in [BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                                       BUILTIN_LP_FIXED_NEG_SAMPLER]:
                    # fixed negative sample is similar to uniform negative sample
                    neg_dst_emb = emb[vtype][neg_dst.reshape(-1,)]
                    neg_dst_emb = neg_dst_emb.reshape(neg_dst.shape[0], neg_dst.shape[1], -1)
                    # uniform sampled negative samples
                    pos_src_emb = pos_src_emb.reshape(
                        pos_src_emb.shape[0], 1, pos_src_emb.shape[1])
                    rel_embedding = rel_embedding.reshape(
                        1, 1, rel_embedding.shape[-1])
                    neg_score = calc_rotate_pos_score(pos_src_emb,
                                                      neg_dst_emb,
                                                      rel_embedding,
                                                      self.emb_init,
                                                      self.gamma,
                                                      device)
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
                    neg_score = calc_rotate_neg_tail_score(
                        pos_src_emb, neg_dst_emb, rel_embedding,
                        1, pos_src_emb.shape[0], neg_dst_emb.shape[0],
                        self.emb_init, self.gamma,
                        device)
                    # shape (batch_size, num_negs)
                    neg_score = neg_score.reshape(-1, neg_dst_emb.shape[0])
                else:
                    assert False, f"Unknow negative sample type {neg_sample_type}"
                assert len(neg_score.shape) == 2
                neg_scores.append(neg_score)
            neg_scores = th.cat(neg_scores, dim=-1).detach()
            # gloo with cpu will consume less GPU memory
            neg_scores = neg_scores.cpu() \
                if is_distributed() and get_backend() == "gloo" \
                else neg_scores

            pos_scores = pos_scores.detach()
            pos_scores = pos_scores.cpu() \
                if is_distributed() and get_backend() == "gloo" \
                else pos_scores
            scores[canonical_etype] = (pos_scores, neg_scores)

        return scores

    @property
    def in_dims(self):
        """ Return the input dimension size, which is given in class initialization.
        """
        return self.h_dim

    @property
    def out_dims(self):
        """ Return ``1`` for link prediction tasks.
        """
        return 1

class LinkPredictContrastiveRotatEDecoder(LinkPredictRotatEDecoder):
    """ Decoder for link prediction designed for contrastive loss
        using the RotatE as the score function.

    Note:
    ------
    This class is specifically implemented for contrastive loss. But
    it could also be used by other pair-wise loss functions for link
    prediction tasks.

    Parameters
    ----------
    etypes: list of tuples
        The canonical edge types of the graph in the format of
        [(src_ntype1, etype1, dst_ntype1), ...]
    h_dim: int
        The input dimension size. It is the dimension for both source and destination
        node embeddings.
    gamma: float
        The gamma value for model weight initialization. Default: 4.
    """
    # pylint: disable=unused-argument
    def forward(self, g, h, e_h=None):
        with g.local_scope():
            scores = {}

            for canonical_etype in g.canonical_etypes:
                if g.num_edges(canonical_etype) == 0:
                    continue # the block might contain empty edge types

                i = self.etype2rid[canonical_etype]
                self.trained_rels[i] += 1
                rel_embedding = self._w_relation(th.tensor(i).to(self._w_relation.weight.device))
                rel_embedding = rel_embedding.unsqueeze(dim=1)
                src_type, _, dest_type = canonical_etype
                u, v = g.edges(etype=canonical_etype)
                # Sort edges according to source node ids
                # The same function is invoked by computing both pos scores
                # and neg scores, by sorting edges according to source nids
                # the output scores of pos_score and neg_score are compatible.
                #
                # For example:
                #
                # pos pairs   |  neg pairs
                # (10, 20)    |  (10, 3), (10, 1), (10, 0), (10, 22)
                # (13, 6)     |  (13, 3), (13, 1), (13, 0), (13, 22)
                # (29, 8)     |  (29, 3), (29, 1), (29, 0), (29, 22)
                #
                # TODO: use stable to keep the order of negatives. This may not
                # be necessary
                u_sort_idx = th.argsort(u, stable=True)
                u = u[u_sort_idx]
                v = v[u_sort_idx]
                src_emb = h[src_type][u]
                dest_emb = h[dest_type][v]
                rel_embedding = rel_embedding.repeat(1,dest_emb.shape[0]).T
                scores_etype = calc_rotate_pos_score(src_emb,
                                                     dest_emb,
                                                     rel_embedding,
                                                     self.emb_init,
                                                     self.gamma)
                scores[canonical_etype] = scores_etype

            return scores

class LinkPredictWeightedRotatEDecoder(LinkPredictRotatEDecoder):
    """Link prediction decoder with the score function of RotatE
       with edge weight.

       When computing loss, edge weights are used to adjust the loss.

    Parameters
    ----------
    etypes: list of tuples
        The canonical edge types of the graph in the format of
        [(src_ntype1, etype1, dst_ntype1), ...]
    h_dim: int
        The input dimension size. It is the dimension for both source and destination
        node embeddings.
    gamma: float
        The gamma value for model weight initialization. Default: 4.
    edge_weight_fields: dict of str
        The edge feature field(s) storing the edge weights.
    """
    def __init__(self, etypes, h_dim, gamma=4., edge_weight_fields=None):
        self._edge_weight_fields = edge_weight_fields
        super(LinkPredictWeightedRotatEDecoder, self).__init__(etypes, h_dim, gamma)

    # pylint: disable=signature-differs
    def forward(self, g, h, e_h):
        """Forward function.

        This computes the RotatE score on every edge type.
        """
        with g.local_scope():
            scores = {}

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
                scores_etype = calc_rotate_pos_score(src_emb,
                                                     dest_emb,
                                                     rel_embedding,
                                                     self.emb_init,
                                                     self.gamma)

                if e_h is not None and canonical_etype in e_h.keys():
                    weight = e_h[canonical_etype]
                    assert th.is_tensor(weight), \
                        "The edge weight for Link prediction must be a torch tensor." \
                        "LinkPredictWeightedRotatEDecoder only accepts a 1D edge " \
                        "feature as edge weight."
                    weight = weight.flatten()
                else:
                    # current etype does not have weight
                    weight = th.ones((g.num_edges(canonical_etype),),
                                     device=scores_etype.device)
                scores[canonical_etype] = (scores_etype,
                                           weight)

            return scores


class LinkPredictDistMultDecoder(LinkPredictMultiRelationLearnableDecoder):
    """ Decoder for link prediction using the DistMult as the score function.

    Parameters
    ----------
    etypes: list of tuples
        The canonical edge types of the graph in the format of
        [(src_ntype1, etype1, dst_ntype1), ...]
    h_dim: int
        The input dimension size. It is the dimension for both source and destination
        node embeddings.
    gamma: float
        The gamma value for model weight initialization. Default: 40.
    """

    def init_w_relation(self):
        self._w_relation = nn.Embedding(self.num_rels, self.h_dim)
        emb_init = self.gamma / self.h_dim
        nn.init.uniform_(self._w_relation.weight, -emb_init, emb_init)

    # pylint: disable=unused-argument
    def forward(self, g, h, e_h=None):
        """ Link prediction decoder forward function using the DistMult as the score function.

        This computes the edge score on every edge type.

        Parameters
        ----------
        g: DGLGraph
            The input graph.
        h: dict of Tensor
            The input node embeddings in the format of {ntype: emb}.
        e_h: dict of Tensor
            The input edge embeddings in the format of {(src_ntype, etype, dst_ntype): emb}.
            Not used, but reserved for future support of edge embeddings. Default: None.

        Returns
        -------
        scores: dict of Tensor
            The scores for edges of all edge types in the input graph in the format of
            {(src_ntype, etype, dst_ntype): score}.
        """
        with g.local_scope():
            scores = {}

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
                scores[canonical_etype] = scores_etype

            return scores

    def calc_test_scores(self, emb, pos_neg_tuple, neg_sample_type, device):
        """ Compute scores for positive edges and negative edges.

        Parameters
        ----------
        emb: dict of Tensor
            Node embeddings in the format of {ntype: emb}.
        pos_neg_tuple: dict of tuple
            Positive and negative edges stored in a dict of tuple in the format of
            {("src_ntype1", "etype1", "dst_ntype1" ): (pos_src_idx, neg_src_idx,
            pos_dst_idx, neg_dst_idx)}.

            The `pos_src_idx` represents the postive source node indexes in the format
            of Torch.Tensor. The `neg_src_idx` represents the negative source node indexes
            in the format of Torch.Tensor. The `pos_dst_idx` represents the postive destination
            node indexes in the format of Torch.Tensor. The `neg_dst_idx` represents the
            negative destination node indexes in the format of Torch.Tensor.

            We define positive and negative edges as:

            * The positive edges: (pos_src_idx, pos_dst_idx)
            * The negative edges: (pos_src_idx, neg_dst_idx) and
              (neg_src_idx, pos_dst_idx)

        neg_sample_type: str
            Describe how negative samples are sampled. There are two options:

            * ``Uniform``: For each positive edge, we sample K negative edges.
            * ``Joint``: For one batch of positive edges, we sample K negative edges.

        device: th.device
            Device used to compute scores.

        Returns
        --------
        scores: dict of tuple
            Return a dictionary of edge type's positive scores and negative scores in the format
            of {(src_ntype, etype, dst_ntype): (pos_scores, neg_scores)}
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
                if neg_sample_type in [BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                                       BUILTIN_LP_FIXED_NEG_SAMPLER]:
                    # fixed negative sample is similar to uniform negative sample
                    neg_src_emb = neg_src_emb.reshape(neg_src.shape[0], neg_src.shape[1], -1)
                    # uniform sampled negative samples
                    pos_dst_emb = pos_dst_emb.reshape(
                        pos_dst_emb.shape[0], 1, pos_dst_emb.shape[1])
                    rel_embedding = rel_embedding.reshape(
                        1, 1, rel_embedding.shape[-1])
                    neg_score = calc_distmult_pos_score(
                        neg_src_emb, pos_dst_emb, rel_embedding, device)
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
                if neg_sample_type in [BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                                       BUILTIN_LP_FIXED_NEG_SAMPLER]:
                    # fixed negative sample is similar to uniform negative sample
                    neg_dst_emb = emb[vtype][neg_dst.reshape(-1,)]
                    neg_dst_emb = neg_dst_emb.reshape(neg_dst.shape[0], neg_dst.shape[1], -1)
                    # uniform sampled negative samples
                    pos_src_emb = pos_src_emb.reshape(
                        pos_src_emb.shape[0], 1, pos_src_emb.shape[1])
                    rel_embedding = rel_embedding.reshape(
                        1, 1, rel_embedding.shape[-1])
                    neg_score = calc_distmult_pos_score(
                        pos_src_emb, neg_dst_emb, rel_embedding, device)
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
            neg_scores = th.cat(neg_scores, dim=-1).detach()
            # gloo with cpu will consume less GPU memory
            neg_scores = neg_scores.cpu() \
                if is_distributed() and get_backend() == "gloo" \
                else neg_scores

            pos_scores = pos_scores.detach()
            pos_scores = pos_scores.cpu() \
                if is_distributed() and get_backend() == "gloo" \
                else pos_scores
            scores[canonical_etype] = (pos_scores, neg_scores)

        return scores

    @property
    def in_dims(self):
        """ Return the input dimension size, which is given in class initialization.
        """
        return self.h_dim

    @property
    def out_dims(self):
        """ Return ``1`` for link prediction tasks.
        """
        return 1

class LinkPredictContrastiveDistMultDecoder(LinkPredictDistMultDecoder):
    """ Decoder for link prediction designed for contrastive loss
        using the DistMult as the score function.

    Note:
    ------
    This class is specifically implemented for contrastive loss. But
    it could also be used by other pair-wise loss functions for link
    prediction tasks.

    Parameters
    ----------
    etypes: list of tuples
        The canonical edge types of the edges used during model training in the format of
        [(src_ntype1, etype1, dst_ntype1), ...]
    h_dim: int
        The input dimension size. It is the dimension for both source and destination
        node embeddings.
    gamma: float
        The gamma value for model weight initialization. Default: 40.
    """
    # TODO(xiang): Develop a better solution for supporting pair-wise
    # loss functions in link prediction tasks. The
    # LinkPredictContrastiveDotDecoder is implemented based on the
    # assumption that the same decoder. Forward will be called twice
    # with a positive graph and negative graph respectively. And
    # the positive and negative graphs are compatible. We can simply
    # sort the edges in postive and negative graphs to create <pos, neg>
    # pairs. This implementation makes strong assumption of the correlation
    # between the Dataloader, Decoder and the Loss function. We should
    # find a better implementation.

    # pylint: disable=unused-argument
    def forward(self, g, h, e_h=None):
        with g.local_scope():
            scores = {}

            for canonical_etype in g.canonical_etypes:
                if g.num_edges(canonical_etype) == 0:
                    continue # the block might contain empty edge types

                i = self.etype2rid[canonical_etype]
                self.trained_rels[i] += 1
                rel_embedding = self._w_relation(th.tensor(i).to(self._w_relation.weight.device))
                rel_embedding = rel_embedding.unsqueeze(dim=1)
                src_type, _, dest_type = canonical_etype
                u, v = g.edges(etype=canonical_etype)
                # Sort edges according to source node ids
                # The same function is invoked by computing both pos scores
                # and neg scores, by sorting edges according to source nids
                # the output scores of pos_score and neg_score are compatible.
                #
                # For example:
                #
                # pos pairs   |  neg pairs
                # (10, 20)    |  (10, 3), (10, 1), (10, 0), (10, 22)
                # (13, 6)     |  (13, 3), (13, 1), (13, 0), (13, 22)
                # (29, 8)     |  (29, 3), (29, 1), (29, 0), (29, 22)
                #
                # TODO: use stable to keep the order of negatives. This may not
                # be necessary
                u_sort_idx = th.argsort(u, stable=True)
                u = u[u_sort_idx]
                v = v[u_sort_idx]
                src_emb = h[src_type][u]
                dest_emb = h[dest_type][v]
                rel_embedding = rel_embedding.repeat(1,dest_emb.shape[0]).T
                scores_etype = calc_distmult_pos_score(src_emb, dest_emb, rel_embedding)
                scores[canonical_etype] = scores_etype

            return scores

class LinkPredictWeightedDistMultDecoder(LinkPredictDistMultDecoder):
    """Link prediction decoder with the score function of DistMult
       with edge weight.

       When computing loss, edge weights are used to adjust the loss

    Parameters
    ----------
    etypes: list of tuples
        The canonical edge types of the graph in the format of
        [(src_ntype1, etype1, dst_ntype1), ...]
    h_dim: int
        The input dimension size. It is the dimension for both source and destination
        node embeddings.
    gamma: float
        The gamma value for model weight initialization. Default: 40.
    edge_weight_fields: dict of str
        The edge feature field(s) storing the edge weights.
    """
    def __init__(self, etypes, h_dim, gamma=40., edge_weight_fields=None):
        self._edge_weight_fields = edge_weight_fields
        super(LinkPredictWeightedDistMultDecoder, self).__init__(etypes, h_dim, gamma)

    # pylint: disable=signature-differs
    def forward(self, g, h, e_h):
        """Forward function.

        This computes the DistMult score on every edge type.
        """
        with g.local_scope():
            scores = {}

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

                if e_h is not None and canonical_etype in e_h.keys():
                    weight = e_h[canonical_etype]
                    assert th.is_tensor(weight), \
                        "The edge weight for Link prediction must be a torch tensor." \
                        "LinkPredictWeightedDistMultDecoder only accepts a single edge " \
                        "feature as edge weight."
                    weight = weight.flatten()
                else:
                    # current etype does not has weight
                    weight = th.ones((g.num_edges(canonical_etype),),
                                     device=scores_etype.device)
                scores[canonical_etype] = (scores_etype,
                                           weight)

            return scores

class LinkPredictWeightedDotDecoder(LinkPredictDotDecoder):
    """ Decoder for link prediction using the dot product with edge weight as the score
    function.

    When computing loss, edge weights are used to adjust the loss

    Parameters
    ----------
    in_dim: int
        The input dimension size. It is the dimension for both source and destination
        node embeddings.
    edge_weight_fields: dict of str
        The edge feature field(s) storing the edge weights.
    """
    def __init__(self, in_dim, edge_weight_fields):
        self._edge_weight_fields = edge_weight_fields
        super(LinkPredictWeightedDotDecoder, self).__init__(in_dim)

    # pylint: disable=signature-differs
    def forward(self, g, h, e_h):
        """Forward function.

        This computes the dot product score on every edge type.
        """
        with g.local_scope():
            scores = {}

            for canonical_etype in g.canonical_etypes:
                if g.num_edges(canonical_etype) == 0:
                    continue # the block might contain empty edge types

                src_type, _, dest_type = canonical_etype
                u, v = g.edges(etype=canonical_etype)
                src_emb = h[src_type][u]
                dest_emb = h[dest_type][v]
                scores_etype = calc_dot_pos_score(src_emb, dest_emb)

                if e_h is not None and canonical_etype in e_h.keys():
                    weight = e_h[canonical_etype]
                    assert th.is_tensor(weight), \
                        "The edge weight for Link prediction must be a torch tensor." \
                        "LinkPredictWeightedDotDecoder only accepts a single edge " \
                        "feature as edge weight."
                    weight = weight.flatten()
                else:
                    # current etype does not has weight
                    weight = th.ones((g.num_edges(canonical_etype),),
                                     device=scores_etype.device)
                scores[canonical_etype] = (scores_etype,
                                           weight)
            return scores
