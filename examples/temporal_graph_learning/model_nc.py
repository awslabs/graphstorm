import torch
import torch.nn as nn
import torch.nn.functional as F

import graphstorm as gs

from input_encoder import NodeEncoderInputLayer
from gnn_encoder import TemporalRelationalGraphEncoder
from graphstorm.model.node_decoder import EntityClassifier
from graphstorm.model.loss_func import ClassifyLossFunc

def create_rgcn_model_for_nc(g, config):
    """ Create a customized model for node prediction.

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    config: GSConfig
        Configurations

    Returns
    -------
    GSgnnNodeModel : The GNN model.
    """

    model = gs.model.GSgnnNodeModel(config.alpha_l2norm)

    # Set input layer (project input feats to hidden dims)
    feat_size = gs.get_node_feat_size(g, config.node_feat_name)
    encoder = NodeEncoderInputLayer(
        g,
        config.hidden_size,
        feat_size,
        dropout=config.dropout,
    )
    model.set_node_input_encoder(encoder)

    # Set customized GNN encoders & decoder
    model.set_gnn_encoder(
        TemporalRelationalGraphEncoder(
            g,
            config.hidden_size,
            config.hidden_size,
            num_hidden_layers=config.num_layers - 1,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
    )

    # Set node cls decoders
    model.set_decoder(
        EntityClassifier(model.gnn_encoder.out_dims,
                         config.num_classes,
                         config.multilabel)
    )

    # Set node cls loss
    model.set_loss_func(
        ClassifyLossFunc(config.multilabel,
                         config.multilabel_weights,
                         config.imbalance_class_weights)
    )

    # Set optimizer
    model.init_optimizer(
        lr=config.lr,
        sparse_optimizer_lr=config.sparse_optimizer_lr,
        weight_decay=config.wd_l2norm,
        lm_lr=config.lm_tune_lr,
    )
    return model



