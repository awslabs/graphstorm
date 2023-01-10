"""GNN model for link prediction in GraphStorm"""
import torch as th

from .gnn import GSgnnModel

class GSgnnLinkPredictionModel(GSgnnModel):
    """ GraphStorm GNN model for link prediction

    Parameters
    ----------
    alpha_l2norm : float
        The alpha for L2 normalization.
    """
    def __init__(self, alpha_l2norm):
        super(GSgnnLinkPredictionModel, self).__init__()
        self.alpha_l2norm = alpha_l2norm

    def forward(self, blocks, pos_graph, neg_graph, input_feats, input_nodes):
        """ The forward function for link prediction.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        pos_graph : a DGLGraph
            The graph that contains the positive edges.
        neg_graph : a DGLGraph
            The graph that contains the negative edges.
        input_feats : dict
            The input features of the message passing graphs.
        input_nodes : dict
            The input nodes of the message passing graphs.

        Returns
        -------
        The loss of prediction.
        """
        alpha_l2norm = self.alpha_l2norm
        gnn_embs = self.compute_embed_step(blocks, input_feats, input_nodes)

        # TODO add w_relation in calculating the score. The current is only valid for
        # homogenous graph.
        pos_score = self.decoder(pos_graph, gnn_embs)
        neg_score = self.decoder(neg_graph, gnn_embs)
        pred_loss = self.loss_func(pos_score, neg_score)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # weighted addition to the total loss
        return pred_loss + alpha_l2norm * reg_loss
