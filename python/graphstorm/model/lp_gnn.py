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

    GNN model for link prediction in GraphStorm.
"""
import abc
import torch as th

from .gnn import GSgnnModel, GSgnnModelBase
from ..dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from ..dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER
from ..eval.utils import calc_ranking

class GSgnnLinkPredictionModelInterface:
    """ The interface for GraphStorm link prediction model.

    This interface defines two main methods for training and inference.
    """
    @abc.abstractmethod
    def forward(self, blocks, pos_graph, neg_graph,
        node_feats, edge_feats, input_nodes=None):
        """ The forward function for link prediction.

        This method is used for training. It takes a mini-batch, including
        the graph structure, node features and edge features and
        computes the loss of the model in the mini-batch.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        pos_graph : a DGLGraph
            The graph that contains the positive edges.
        neg_graph : a DGLGraph
            The graph that contains the negative edges.
        node_feats : dict of Tensors
            The input node features of the message passing graphs.
        edge_feats : dict of Tensors
            The input edge features of the message passing graphs.
        input_nodes: dict of Tensors
            The input nodes of a mini-batch.

        Returns
        -------
        The loss of prediction.
        """

class GSgnnLinkPredictionModelBase(GSgnnModelBase,  # pylint: disable=abstract-method
                                   GSgnnLinkPredictionModelInterface):
    """ The base class for link-prediction GNN

    When a user wants to define a link prediction GNN model and train the model
    in GraphStorm, the model class needs to inherit from this base class.
    A user needs to implement some basic methods including `forward`, `predict`,
    `save_model`, `restore_model` and `create_optimizer`.
    """


class GSgnnLinkPredictionModel(GSgnnModel, GSgnnLinkPredictionModelInterface):
    """ GraphStorm GNN model for link prediction

    Parameters
    ----------
    alpha_l2norm : float
        The alpha for L2 normalization.
    """
    def __init__(self, alpha_l2norm):
        super(GSgnnLinkPredictionModel, self).__init__()
        self.alpha_l2norm = alpha_l2norm

    def forward(self, blocks, pos_graph,
        neg_graph, node_feats, _, input_nodes=None):
        """ The forward function for link prediction.

        This model doesn't support edge features for now.
        """
        alpha_l2norm = self.alpha_l2norm
        if blocks is None or len(blocks) == 0:
            # no GNN message passing
            encode_embs = self.comput_input_embed(input_nodes, node_feats)
        else:
            # GNN message passing
            encode_embs = self.compute_embed_step(blocks, node_feats)

        # TODO add w_relation in calculating the score. The current is only valid for
        # homogenous graph.
        pos_score = self.decoder(pos_graph, encode_embs)
        neg_score = self.decoder(neg_graph, encode_embs)
        pred_loss = self.loss_func(pos_score, neg_score)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # weighted addition to the total loss
        return pred_loss + alpha_l2norm * reg_loss


def get_embs(emb, node_list, neg_sample_type, canonical_etype):
    """ Fetch node embeddings for mini-batch prediction.

        Parameters
        ----------
        emb: dict of Tensor
            Node embeddings.
        node_list: tuple of tensor
            list of positive/negative source and destination nodes
        neg_sample_type: str
            Describe how negative samples are sampled.
                Uniform: For each positive edge, we sample K negative edges
                Joint: For one batch of positive edges, we sample
                       K negative edges
        canonical_etype: str
            Relation type

        Return
        ------
        tuple of tensors
            node embeddings stored in a tuple:
            tuple(positive source embeddings, negative source embeddings,
            postive destination embeddings, negatve destination embeddings).
    """
    utype, _, vtype = canonical_etype
    pos_src, neg_src, pos_dst, neg_dst = node_list
    pos_src_emb = emb[utype][pos_src]
    pos_dst_emb = emb[vtype][pos_dst]
    neg_src_emb = None
    neg_dst_emb = None

    if neg_src is not None:
        neg_src_emb = emb[utype][neg_src.reshape(-1,)]
        if neg_sample_type == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
            neg_src_emb = neg_src_emb.reshape(neg_src.shape[0], neg_src.shape[1], -1)
    if neg_dst is not None:
        if neg_sample_type == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
            neg_dst_emb = emb[vtype][neg_dst.reshape(-1,)]
            neg_dst_emb = neg_dst_emb.reshape(neg_dst.shape[0], neg_dst.shape[1], -1)
        elif neg_sample_type == BUILTIN_LP_JOINT_NEG_SAMPLER:
            neg_dst_emb = emb[vtype][neg_dst]
        else:
            assert False, f"Unknow negative sample type {neg_sample_type}"
    return (pos_src_emb, neg_src_emb, pos_dst_emb, neg_dst_emb)


def compute_batch_score(decoder, ranking, batch_emb, neg_sample_type, device):
    """ Compute scores for positive edges and negative edges using batch embeddings"""
    score = decoder.calc_test_scores(batch_emb, neg_sample_type, device)
    for canonical_etype, s in score.items():
        # We do not concatenate rankings into a single
        # ranking tensor to avoid unnecessary data copy.
        pos_score, neg_score = s
        if canonical_etype in ranking:
            ranking[canonical_etype].append(calc_ranking(pos_score, neg_score))
        else:
            ranking[canonical_etype] = [calc_ranking(pos_score, neg_score)]


def lp_mini_batch_predict(model, emb, loader, device):
    """ Perform mini-batch prediction.

        This function follows full-grain GNN embedding inference.
        After having the GNN embeddings, we need to perform mini-batch
        computation to make predictions on the GNN embeddings.

        Parameters
        ----------
        model : GSgnnModel
            The GraphStorm GNN model
        emb : dict of Tensor
            The GNN embeddings
        loader : GSgnnEdgeDataLoader
            The GraphStorm dataloader
        device: th.device
            Device used to compute test scores

        Returns
        -------
        rankings: dict of tensors
            Rankings of positive scores in format of {etype: ranking}
    """
    decoder = model.decoder
    with th.no_grad():
        ranking = {}
        for pos_neg_tuple, neg_sample_type in loader:
            node_list = {}
            canonical_etype = prev_canonical_etype = list(pos_neg_tuple.keys())[0]
            pos_src, neg_src, pos_dst, neg_dst = pos_neg_tuple[prev_canonical_etype]
            pos_src_size = pos_src.shape[0]
            neg_src_size = neg_src.shape[0]
            # To optimize network bandwidth usage, we concatenate node lists from multiple
            # batches and retrieve their embeddings with a single remote pull. Heuristically
            # we have found using a batch size of 100K results in efficient network utilization
            # and reduces the end-to-end inference pipeline from 45 to 21 minutes compared to
            # using a batch size of 1024 on ogbn-papers100M dataset. More details can be found
            # at PR#101 (https://github.com/awslabs/graphstorm/pull/101)
            num_batch_to_cat = int(100000/pos_src.shape[0])
            for _ in range(num_batch_to_cat):
                pos_neg_tuple_next, _ = next(loader, (None, None))
                if pos_neg_tuple_next is not None:
                    canonical_etype = list(pos_neg_tuple_next.keys())[0]
                    if canonical_etype != prev_canonical_etype:
                        node_list[prev_canonical_etype] = (pos_src, neg_src, pos_dst, neg_dst)
                        pos_src, neg_src, pos_dst, neg_dst = pos_neg_tuple_next[canonical_etype]
                    else:
                        pos_src_, neg_src_, pos_dst_, neg_dst_ = pos_neg_tuple_next[canonical_etype]
                        pos_src = th.cat((pos_src, pos_src_), dim=0)
                        neg_src = th.cat((neg_src, neg_src_), dim=0)
                        pos_dst = th.cat((pos_dst, pos_dst_), dim=0)
                        neg_dst = th.cat((neg_dst, neg_dst_), dim=0)
                    prev_canonical_etype = canonical_etype

            node_list[canonical_etype] = (pos_src, neg_src, pos_dst, neg_dst)

            for etype, item in node_list.items():
                batch_embs = get_embs(emb, item, neg_sample_type, etype)
                pos_src_emb, neg_src_emb, pos_dst_emb, neg_dst_emb = batch_embs
                p_st = ns_st = 0
                batch_emb = {}
                # Split the concatenated batch back into orginal batch size to avoid GPU OOM
                for _ in range(num_batch_to_cat):
                    batch_emb[etype] = (pos_src_emb[p_st: p_st + pos_src_size],
                        neg_src_emb[ns_st: ns_st + neg_src_size]
                            if neg_src_emb is not None else None,
                        pos_dst_emb[p_st: p_st + pos_src_size],
                        neg_dst_emb[ns_st: ns_st + neg_src_size]
                            if neg_dst_emb is not None else None)
                    compute_batch_score(decoder, ranking, batch_emb, neg_sample_type, device)
                    p_st += pos_src_size
                    ns_st += neg_src_size
                    if p_st >= pos_src_emb.shape[0]:
                        break
        rankings = {}
        for canonical_etype, rank in ranking.items():
            rankings[canonical_etype] = th.cat(rank, dim=0)
    return rankings
