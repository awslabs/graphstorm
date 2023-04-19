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

def get_embs(emb, pos_neg_tuple, neg_sample_type, loader, scale):
    """ Fetch node embeddings for mini-batch prediction.

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
        loader : GSgnnEdgeDataLoader
            The GraphStorm dataloader
        scale : int
            Number of batches to be fused whilw fetching embeddings

        Return
        ------
        tuple of tensors
            node embeddings stored in a tuple:
            tuple(positive source embeddings, negative source embeddings,
            postive destination embeddings, negatve destination embeddings).
    """
    assert isinstance(pos_neg_tuple, dict) and len(pos_neg_tuple) == 1, \
    "DotDecoder is only applicable to link prediction task with " \
    "single target training edge type"

    canonical_etype = list(pos_neg_tuple.keys())[0]
    pos_src, neg_src, pos_dst, neg_dst = pos_neg_tuple[canonical_etype]
    utype, _, vtype = canonical_etype

    for _ in range(scale):
        pos_neg_tuple, _ = next(loader, (None, None))
        if pos_neg_tuple is None: break
        pos_src_, neg_src_, pos_dst_, neg_dst_ = pos_neg_tuple[canonical_etype]
        pos_src = th.cat((pos_src, pos_src_), dim=0)
        neg_src = th.cat((neg_src, neg_src_), dim=0)
        pos_dst = th.cat((pos_dst, pos_dst_), dim=0)
        neg_dst = th.cat((neg_dst, neg_dst_), dim=0)

    pos_src_emb = emb[utype][pos_src]
    pos_dst_emb = emb[vtype][pos_dst]
    if neg_src is not None:
        neg_src_emb = emb[utype][neg_src.reshape(-1,)]
    if neg_dst is not None:
        if neg_sample_type == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
            neg_dst_emb = emb[vtype][neg_dst.reshape(-1,)]
        elif neg_sample_type == BUILTIN_LP_JOINT_NEG_SAMPLER:
            neg_dst_emb = emb[vtype][neg_dst]
        else:
            assert False, f"Unknow negative sample type {neg_sample_type}"

    return (pos_src_emb, pos_dst_emb, neg_src_emb, neg_dst_emb)


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
        dict of (list, list):
            Return a dictionary of edge type to
            (positive scores, negative scores)
    """
    decoder = model.decoder
    with th.no_grad():
        scores = {}
        for pos_neg_tuple, neg_sample_type in loader:
            # TODO(IN): Find a scaling factor based on CPU mem and network throughput
            scale = 5
            canonical_etype = list(pos_neg_tuple.keys())[0]
            pos_src, neg_src, _, _ = pos_neg_tuple[canonical_etype]
            eval_batch_size = pos_src.shape[0]
            neg_sample_size = neg_src.shape[0]
            batch_emb = get_embs(emb, pos_neg_tuple, neg_sample_type, loader, scale)
            pos_src_emb, pos_dst_emb, neg_src_emb, neg_dst_emb = batch_emb
            # TODO(IN): Check if neg src/dst is None
            for s in range(scale):
                b_st = s * eval_batch_size
                nb_st = s * neg_sample_size
                if b_st > eval_batch_size * scale: break
                batch_emb = (pos_src_emb[b_st: b_st + eval_batch_size],
                    pos_dst_emb[b_st: b_st + eval_batch_size],
                    neg_src_emb[nb_st: nb_st + neg_sample_size],
                    neg_dst_emb[nb_st: nb_st + neg_sample_size])
                score = \
                    decoder.calc_test_scores(
                        batch_emb, pos_neg_tuple, neg_sample_type, device)
                for canonical_etype, s in score.items():
                    # We do not concatenate pos scores/neg scores
                    # into a single pos score tensor/neg score tensor
                    # to avoid unnecessary data copy.
                    if canonical_etype in scores:
                        scores[canonical_etype].append(s)
                    else:
                        scores[canonical_etype] = [s]
    return scores
