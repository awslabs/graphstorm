"""GNN model for node prediction task in GraphStorm
"""
import torch as th

from .gnn import GSgnnModel, set_gnn_encoder
from .loss_func import ClassifyLossFunc, RegressionLossFunc
from .node_decoder import EntityClassifier, EntityRegression
from ..config import BUILTIN_TASK_NODE_CLASSIFICATION
from ..config import BUILTIN_TASK_NODE_REGRESSION

class GSgnnNodeModel(GSgnnModel):
    """ GraphStorm GNN model for node prediction tasks

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    predict_ntype: str
        The node type for prediction.
    alpha_l2norm : float
        The alpha for L2 normalization.
    task_type : str
        The task type
    """
    def __init__(self, g, predict_ntype, alpha_l2norm, task_type):
        super(GSgnnNodeModel, self).__init__(g)
        self.predict_ntype = predict_ntype
        self.alpha_l2norm = alpha_l2norm
        self._task_type = task_type

    def forward(self, blocks, input_feats, input_nodes, labels):
        """ The forward function for node prediction.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        input_feats : dict
            The input features of the message passing graphs.
        input_nodes : dict
            The input nodes of the message passing graphs.
        labels: Tensor
            The labels of the predicted nodes.

        Returns
        -------
        The loss of prediction.
        """
        alpha_l2norm = self.alpha_l2norm
        gnn_embs = self.compute_embed_step(blocks, input_feats, input_nodes)
        emb = gnn_embs[self.predict_ntype]
        logits = self.decoder(emb)
        pred_loss = self.loss_func(logits, labels)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # weighted addition to the total loss
        total_loss = pred_loss + alpha_l2norm * reg_loss

        return total_loss

    def predict(self, g, feat_name, target_nidx,
                fanout, batch_size, mini_batch_infer, task_tracker=None):
        """ Make predict on the target nodes of the input graph.

        Parameters
        ----------
        g : DGLGraph
            The input graph
        feat_name : str
            The node feature names
        target_nidx: dict of tensors
            The idices of nodes to generate embeddings.
        fanout : list of int
            The fanout for sampling neighbors.
        batch_size : int
            The mini-batch size
        mini_batch_infer : bool
            whether or not to use mini-batch inference.
        task_tracker : GSTaskTrackerAbc
            The task tracker

        Returns
        -------
        tensor
            The prediction results.
        """
        outputs = self.compute_embeddings(g, feat_name, target_nidx,
                                          fanout, batch_size, mini_batch_infer, task_tracker)
        output = outputs[self.predict_ntype]
        self.decoder.eval()
        with th.no_grad():
            # TODO(zhengda) we need to use mini-batch here.
            res = self.decoder.predict(output[0:len(output)].to(self.device))
        self.decoder.train()
        return res, outputs

    @property
    def task_name(self):
        """ The task name of the model.
        """
        return self._task_type

def create_node_gnn_model(g, config, train_task):
    """ Create a GNN model for node prediction.

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
    model = GSgnnNodeModel(g, config.predict_ntype, config.alpha_l2norm, config.task_type)
    set_gnn_encoder(model, g, config, train_task)
    if config.task_type == BUILTIN_TASK_NODE_CLASSIFICATION:
        model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims,
                                           config.num_classes,
                                           config.multilabel))
        model.set_loss_func(ClassifyLossFunc(config))
    elif config.task_type == BUILTIN_TASK_NODE_REGRESSION:
        model.set_decoder(EntityRegression(model.gnn_encoder.out_dims))
        model.set_loss_func(RegressionLossFunc())
    else:
        raise ValueError('unknown node task: {}'.format(config.task_type))
    return model
