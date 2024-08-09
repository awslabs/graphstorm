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
from .utils import normalize_node_embs
from ..eval.utils import calc_ranking

class GSgnnLinkPredictionModelInterface:
    """ The interface for GraphStorm link prediction model.

    This interface defines one method: ``forward()`` for training. Link prediction models
    should inherite this interface and implement this method.
    """
    @abc.abstractmethod
    def forward(self, blocks, pos_graph, neg_graph,
        node_feats, edge_feats, pos_edge_feats=None, neg_edge_feats=None, input_nodes=None):
        """ The forward function for link prediction.

        This method is used for training. It takes a list of DGL message flow graphs (MFGs),
        node features, and edge features of a mini-batch as inputs, and
        computes the loss of the model in the mini-batch as the return value. More
        detailed information about DGL MFG can be found in `DGL Neighbor Sampling
        Overview
        <https://docs.dgl.ai/stochastic_training/neighbor_sampling_overview.html>`_.

        Parameters
        ----------
        blocks: list of DGL MFGs
            Sampled subgraph in the list of DGL message flow graph (MFG) format. More
            detailed information about DGL MFG can be found in `DGL Neighbor Sampling
            Overview
            <https://docs.dgl.ai/stochastic_training/neighbor_sampling_overview.html>`_.
        pos_graph : a DGLGraph
            The graph that contains the positive edges.
        neg_graph : a DGLGraph
            The graph that contains the negative edges.
        node_feats : dict of Tensors
            The input node features of the message passing graph.
        edge_feats : dict of Tensors
            The input edge features of the message passing graph.
        input_nodes: dict of Tensors
            The input nodes of a mini-batch.

        Returns
        -------
        float: The loss of prediction of this mini-batch.
        """

# pylint: disable=abstract-method
class GSgnnLinkPredictionModelBase(GSgnnModelBase, GSgnnLinkPredictionModelInterface):
    """ GraphStorm GNN model base class for link-prediction tasks.

    This base class extends GraphStorm ``GSgnnModelBase`` and
    ``GSgnnLinkPredictionModelInterface``. When users want to define a customized link
    prediction GNN model and train the model in GraphStorm, the model class needs to
    inherit from this base class, and implement the required methods including ``forward()``,
    ``predict()``, ``save_model()``, ``restore_model()`` and ``create_optimizer()``.
    """

    def normalize_node_embs(self, embs):
        """ By default do nothing.

            One can implement his/her own node normalization method or call
            .utils.normalize_node_embs to leverage the builtin normalization
            methods.
        """
        return embs

class GSgnnLinkPredictionModel(GSgnnModel, GSgnnLinkPredictionModelInterface):
    """ GraphStorm GNN model for link prediction

        Parameters
        ----------
        alpha_l2norm : float
            The alpha for L2 normalization.
        embed_norm_method: str
            Node embedding normalization method
    """
    def __init__(self, alpha_l2norm, embed_norm_method=None):
        super(GSgnnLinkPredictionModel, self).__init__()
        self.alpha_l2norm = alpha_l2norm
        self.embed_norm_method = embed_norm_method

    def normalize_node_embs(self, embs):
        return normalize_node_embs(embs, self.embed_norm_method)

    # pylint: disable=unused-argument
    def forward(self, blocks, pos_graph,
        neg_graph, node_feats, edge_feats,
        pos_edge_feats=None, neg_edge_feats=None, input_nodes=None):
        """ The forward function for link prediction.

        This model doesn't support edge features for now.
        """
        alpha_l2norm = self.alpha_l2norm
        if blocks is None or len(blocks) == 0:
            # no GNN message passing
            encode_embs = self.comput_input_embed(input_nodes, node_feats)
        else:
            # GNN message passing
            encode_embs = self.compute_embed_step(blocks, node_feats, input_nodes)

        # Call emb normalization.
        encode_embs = self.normalize_node_embs(encode_embs)

        # TODO add w_relation in calculating the score. The current is only valid for
        # homogenous graph.
        pos_score = self.decoder(pos_graph, encode_embs, pos_edge_feats)
        neg_score = self.decoder(neg_graph, encode_embs, neg_edge_feats)
        assert pos_score.keys() == neg_score.keys(), \
            "Positive scores and Negative scores must have edges of same" \
            f"edge types, but get {pos_score.keys()} and {neg_score.keys()}"
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

        This function follows full-graph GNN embedding inference.
        After having the GNN embeddings, we need to perform mini-batch
        computation to make predictions on the GNN embeddings.

        Note: callers should call model.eval() before calling this function
        and call model.train() after when doing training.

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
    return run_lp_mini_batch_predict(decoder,
                                     emb,
                                     loader,
                                     device)

def run_lp_mini_batch_predict(decoder, emb, loader, device):
    """ Perform mini-batch link prediction with the given decoder.

        This function follows full-graph GNN embedding inference.
        After having the GNN embeddings, we need to perform mini-batch
        computation to make predictions on the GNN embeddings.

        Note: callers should call model.eval() before calling this function
        and call model.train() after when doing training.

        Parameters
        ----------
        decoder : LinkPredictNoParamDecoder or LinkPredictLearnableDecoder
            The GraphStorm link prediction decoder model
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
