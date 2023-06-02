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
from ..eval.utils import calc_ranking
from .edge_decoder import (LinkPredictWeightedDotDecoder,
                           LinkPredictWeightedDistMultDecoder)

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

    def prepare_pos_graph(self, pos_graph, data, device):
        """ Prepare edge features for pos_graph

        This method should be called before forward().
        This method is supposed to load any edge data from graph data into pos_graph

        Parameters
        ----------
        pos_graph: DGLGraph
            Positive graph containing positive edges.
        data: GSgnnData
            Graph data.
        device: torch device
            Device to store data.
        """

    def prepare_neg_graph(self, neg_graph, data, device):
        """ Prepare edge features for neg_graph

        This method should be called before forward().
        This method is supposed to load any edge data from graph data into neg_graph

        Parameters
        ----------
        neg_graph: DGLGraph
            Negative graph containing negative edges.
        data: GSgnnData
            Graph data.
        device: torch device
            Device to store data.
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

    def prepare_pos_graph(self, pos_graph, data, device):
        if isinstance(self.decoder, (LinkPredictWeightedDotDecoder,
                                     LinkPredictWeightedDistMultDecoder)):
            # We only extract edge feature (edge weight) for pos_graph if any
            # We do not support edge feature in message passing.
            input_edges = {etype: pos_graph.edges[etype].data[dgl.EID] \
                           for etype in pos_graph.canonical_etypes}
            input_edge_feats = data.get_edge_feats(input_edges, device)
            # store edge feature into pos_graph
            for etype, feat in input_edge_feats.items():
                # self.decoder.edge_weight_fields can be a string
                # or a dict of etype -> list of string, where the length of
                # the list is always 1. (Only one edge weight)
                # See graphstorm.config.GSConfig.lp_edge_weight_for_loss
                # for more details.
                weight_field = self.decoder.edge_weight_fields \
                    if isinstance(self.decoder.edge_weight_fields, str) \
                    else self.decoder.edge_weight_fields[etype][0]
                pos_graph.edges[etype].data[weight_field] = feat

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
            score = \
                decoder.calc_test_scores(
                    emb, pos_neg_tuple, neg_sample_type, device)
            for canonical_etype, s in score.items():
                # We do not concatenate rankings into a single
                # ranking tensor to avoid unnecessary data copy.
                pos_score, neg_score = s
                if canonical_etype in ranking:
                    ranking[canonical_etype].append(calc_ranking(pos_score, neg_score))
                else:
                    ranking[canonical_etype] = [calc_ranking(pos_score, neg_score)]

        rankings = {}
        for canonical_etype, rank in ranking.items():
            rankings[canonical_etype] = th.cat(rank, dim=0)
    return rankings
