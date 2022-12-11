"""GNN model for edge prediction s in GraphStorm
"""
import torch as th
from torch.utils.data import DataLoader

from .gnn import GSgnnModel, set_gnn_encoder
from .loss_func import ClassifyLossFunc, RegressionLossFunc
from .edge_decoder import DenseBiDecoder, MLPEdgeDecoder
from ..config import BUILTIN_TASK_EDGE_CLASSIFICATION
from ..config import BUILTIN_TASK_EDGE_REGRESSION

class GSgnnEdgeModel(GSgnnModel):
    """ GraphStorm GNN model for edge prediction tasks

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    target_etype : tuple of str
        The canonical edge type for prediction.
    alpha_l2norm : float
        The alpha for L2 normalization.
    task_type : str
        The task type
    """
    def __init__(self, g, target_etype, alpha_l2norm, task_type):
        super(GSgnnEdgeModel, self).__init__(g)
        self.target_etype = target_etype
        self.alpha_l2norm = alpha_l2norm
        self._task_type = task_type

    def forward(self, blocks, batch_graph, input_feats, input_nodes, labels):
        """ The forward function for edge prediction.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        batch_graph : a DGLGraph
            The graph where we run edge classification.
        input_feats : dict
            The input features of the message passing graphs.
        input_nodes : dict
            The input nodes of the message passing graphs.
        labels: Tensor
            The labels of the predicted edges.

        Returns
        -------
        The loss of prediction.
        """
        alpha_l2norm = self.alpha_l2norm
        gnn_embs = self.compute_embed_step(blocks, input_feats, input_nodes)

        logits = self.decoder(batch_graph, gnn_embs)
        pred_loss = self.loss_func(logits, labels)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # weighted addition to the total loss
        return pred_loss + alpha_l2norm * reg_loss

    def predict(self, g, feat_name, src_dst_pairs,
                fanout, batch_size, mini_batch_infer, task_tracker=None):
        """Make a prediction on the edges.

        Parameters
        ----------
        g : DistGraph
            The distributed graph.
        feat_name : str
            The node feature names
        src_dst_pairs: dict of node pairs
            The src and dst node IDs of the edges of different etype.
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
        # find the target src and dst ntypes
        # TODO(zhengda) support multiple edge types
        target_src_ntype, target_etype, target_dst_ntype = self.target_etype[0]
        assert target_etype in src_dst_pairs, "The input edges does not have the target edge type."
        src_dst_pairs = src_dst_pairs[target_etype]
        # ----------------------
        # 1. Collect all the node ids from the edge id to create the target nidxs
        # 2. Call the compute embeddings function
        # 3. Call the decoder on the results
        # ----------------------
        # Collect all node IDs from the edges for computing node embeddings.
        # TODO(zhengda) here we use full-graph inference. Ideally, we should
        # only compute the node embeddings of related nodes.
        node_embeddings = self.compute_embeddings(g, feat_name, None,
                                                  fanout, batch_size,
                                                  mini_batch_infer, task_tracker)

        decoder = self.decoder
        if decoder is not None:
            decoder.eval()
        with th.no_grad():
            dataloader = DataLoader(th.arange(len(src_dst_pairs[0])),
                                    batch_size=batch_size, shuffle=False)
            # save preds and labels together in order not to shuffle
            # the order when gather tensors from other trainers
            preds_list = []
            node_embeddings_src = node_embeddings[target_src_ntype][src_dst_pairs[0]]
            node_embeddings_dst = node_embeddings[target_dst_ntype][src_dst_pairs[1]]
            device = self.device
            for edge_ids in dataloader:
                # TODO(zhengda) how to deal with edge features?
                preds_list.append(decoder.predict(
                    node_embeddings_src[edge_ids].to(device),
                    node_embeddings_dst[edge_ids].to(device),
                    None))
            # can't use torch.stack here becasue the size of last tensor is different
            preds = th.cat(preds_list)
        th.distributed.barrier()
        if decoder is not None:
            decoder.train()

        return preds, node_embeddings

    @property
    def task_name(self):
        """ The task name of the model.
        """
        return self._task_type

def create_edge_gnn_model(g, config, train_task):
    """ Create a GNN model for edge prediction.

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
    model = GSgnnEdgeModel(g, config.target_etype, config.alpha_l2norm, config.task_type)
    set_gnn_encoder(model, g, config, train_task)
    if config.task_type == BUILTIN_TASK_EDGE_CLASSIFICATION:
        num_classes = config.num_classes
        decoder_type = config.decoder_type
        num_decoder_basis = config.num_decoder_basis
        dropout = config.dropout if train_task else 0
        # TODO(zhengda) we should support multiple target etypes
        target_etype = config.target_etype[0]
        if decoder_type == "DenseBiDecoder":
            decoder = DenseBiDecoder(in_units=model.gnn_encoder.out_dims,
                                     num_classes=num_classes,
                                     multilabel=config.multilabel,
                                     num_basis=num_decoder_basis,
                                     dropout_rate=dropout,
                                     regression=False,
                                     target_etype=target_etype)
        elif decoder_type == "MLPDecoder":
            decoder = MLPEdgeDecoder(2 * model.gnn_encoder.out_dims,
                                     num_classes,
                                     multilabel=config.multilabel,
                                     target_etype=target_etype)
        else:
            assert False, f"decoder {decoder_type} is not supported."
        model.set_decoder(decoder)
        model.set_loss_func(ClassifyLossFunc(config))
    elif config.task_type == BUILTIN_TASK_EDGE_REGRESSION:
        decoder_type = config.decoder_type
        num_decoder_basis = config.num_decoder_basis
        dropout = config.dropout if train_task else 0
        # TODO(zhengda) we should support multiple target etypes
        target_etype = config.target_etype[0]
        if decoder_type == "DenseBiDecoder":
            decoder = DenseBiDecoder(model.gnn_encoder.out_dims, 1,
                                     num_basis=num_decoder_basis,
                                     multilabel=False,
                                     target_etype=target_etype,
                                     dropout_rate=dropout,
                                     regression=True)
        elif decoder_type == "MLPDecoder":
            decoder = MLPEdgeDecoder(2 * model.gnn_encoder.out_dims, 1,
                                     multilabel=False,
                                     target_etype=target_etype,
                                     regression=True)
        else:
            assert False, "decoder not supported"
        model.set_decoder(decoder)
        model.set_loss_func(RegressionLossFunc())
    else:
        raise ValueError('unknown node task: {}'.format(config.task_type))
    return model
