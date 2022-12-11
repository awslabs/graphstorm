"""GNN model for link prediction in GraphStorm"""
import torch as th

from .gnn import GSgnnModel, set_gnn_encoder
from .loss_func import LinkPredictLossFunc
from .edge_decoder import LinkPredictDotDecoder, LinkPredictDistMultDecoder

class GSgnnLinkPredictionModel(GSgnnModel):
    """ GraphStorm GNN model for link prediction

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    train_etype : tuple of str
        The canonical edge types for training.
    alpha_l2norm : float
        The alpha for L2 normalization.
    """
    def __init__(self, g, train_etype, alpha_l2norm):
        super(GSgnnLinkPredictionModel, self).__init__(g)
        self.alpha_l2norm = alpha_l2norm

        # train_etypes is used when loading decoder.
        # Inference script also needs train_etypes
        self.train_etypes = train_etype

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

    @property
    def task_name(self):
        """ The task name of the model.
        """
        return 'link_prediction'

def create_lp_gnn_model(g, config, train_task):
    """ Create a GNN model for link prediction.

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    config: GSConfig
        Configurations
    train_task : bool
        Whether this model is used for training.

    Returns
    -------
    GSgnnModel : The GNN model.
    """
    model = GSgnnLinkPredictionModel(g, config.train_etype, config.alpha_l2norm)
    set_gnn_encoder(model, g, config, train_task)
    num_train_etype = len(config.train_etype)
    # For backword compatibility, we add this check.
    # if train etype is 1, There is no need to use DistMult
    assert num_train_etype > 1 or config.use_dot_product, \
            "If number of train etype is 1, please use dot product"
    if config.use_dot_product:
        # if the training set only contains one edge type or it is specified in the arguments,
        # we use dot product as the score function.
        if g.rank() == 0:
            print('use dot product for single-etype task.')
            print("Using inner product objective for supervision")
        decoder = LinkPredictDotDecoder(model.gnn_encoder.out_dims)
    else:
        if g.rank() == 0:
            print("Using distmult objective for supervision")
        decoder = LinkPredictDistMultDecoder(g,
                                             model.gnn_encoder.out_dims,
                                             config.gamma)
    model.set_decoder(decoder)
    model.set_loss_func(LinkPredictLossFunc())
    return model
