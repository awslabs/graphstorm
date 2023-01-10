import dgl
import torch as th

from .utils import sys_tracker, get_rank
from .config import BUILTIN_TASK_NODE_CLASSIFICATION
from .config import BUILTIN_TASK_NODE_REGRESSION
from .config import BUILTIN_TASK_EDGE_CLASSIFICATION
from .config import BUILTIN_TASK_EDGE_REGRESSION
from .model.embed import GSNodeInputLayer
from .model.rgcn_encoder import RelationalGCNEncoder
from .model.rgat_encoder import RelationalGATEncoder
from .model.node_gnn import GSgnnNodeModel
from .model.edge_gnn import GSgnnEdgeModel
from .model.lp_gnn import GSgnnLinkPredictionModel
from .model.loss_func import ClassifyLossFunc, RegressionLossFunc
from .model.loss_func import LinkPredictLossFunc
from .model.node_decoder import EntityClassifier, EntityRegression
from .model.edge_decoder import DenseBiDecoder, MLPEdgeDecoder
from .model.edge_decoder import LinkPredictDotDecoder, LinkPredictDistMultDecoder
from .model.utils import get_feat_size
from .tracker import get_task_tracker_class

def initialize(ip_config, backend):
    """ Initialize distributed inference context

    Parameters
    ----------
    ip_config: str
        File path of ip_config file
    backend: str
        Torch distributed backend
    """
    # We need to use socket for communication in DGL 0.8. The tensorpipe backend has a bug.
    # This problem will be fixed in the future.
    dgl.distributed.initialize(ip_config, net_type='socket')
    th.distributed.init_process_group(backend=backend)
    sys_tracker.check("load DistDGL")

def create_builtin_node_gnn_model(g, config, train_task):
    """ Create a built-in GNN model for node prediction.

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
    model = GSgnnNodeModel(config.alpha_l2norm)
    set_gnn_encoder(model, g, config, train_task)
    if config.task_type == BUILTIN_TASK_NODE_CLASSIFICATION:
        model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims,
                                           config.num_classes,
                                           config.multilabel))
        model.set_loss_func(ClassifyLossFunc(config.multilabel,
                                             config.multilabel_weights,
                                             config.imbalance_class_weights))
    elif config.task_type == BUILTIN_TASK_NODE_REGRESSION:
        model.set_decoder(EntityRegression(model.gnn_encoder.out_dims))
        model.set_loss_func(RegressionLossFunc())
    else:
        raise ValueError('unknown node task: {}'.format(config.task_type))
    if train_task:
        model.init_optimizer(lr=config.lr, sparse_lr=config.sparse_lr,
                             weight_decay=config.wd_l2norm)
    return model

def create_builtin_edge_gnn_model(g, config, train_task):
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
    model = GSgnnEdgeModel(config.alpha_l2norm)
    set_gnn_encoder(model, g, config, train_task)
    if config.task_type == BUILTIN_TASK_EDGE_CLASSIFICATION:
        num_classes = config.num_classes
        decoder_type = config.decoder_type
        dropout = config.dropout if train_task else 0
        # TODO(zhengda) we should support multiple target etypes
        target_etype = config.target_etype[0]
        if decoder_type == "DenseBiDecoder":
            num_decoder_basis = config.num_decoder_basis
            decoder = DenseBiDecoder(in_units=model.gnn_encoder.out_dims,
                                     num_classes=num_classes,
                                     multilabel=config.multilabel,
                                     num_basis=num_decoder_basis,
                                     dropout_rate=dropout,
                                     regression=False,
                                     target_etype=target_etype)
        elif decoder_type == "MLPDecoder":
            decoder = MLPEdgeDecoder(model.gnn_encoder.out_dims,
                                     num_classes,
                                     multilabel=config.multilabel,
                                     target_etype=target_etype)
        else:
            assert False, f"decoder {decoder_type} is not supported."
        model.set_decoder(decoder)
        model.set_loss_func(ClassifyLossFunc(config.multilabel,
                                             config.multilabel_weights,
                                             config.imbalance_class_weights))
    elif config.task_type == BUILTIN_TASK_EDGE_REGRESSION:
        decoder_type = config.decoder_type
        dropout = config.dropout if train_task else 0
        # TODO(zhengda) we should support multiple target etypes
        target_etype = config.target_etype[0]
        if decoder_type == "DenseBiDecoder":
            num_decoder_basis = config.num_decoder_basis
            decoder = DenseBiDecoder(model.gnn_encoder.out_dims, 1,
                                     num_basis=num_decoder_basis,
                                     multilabel=False,
                                     target_etype=target_etype,
                                     dropout_rate=dropout,
                                     regression=True)
        elif decoder_type == "MLPDecoder":
            decoder = MLPEdgeDecoder(model.gnn_encoder.out_dims, 1,
                                     multilabel=False,
                                     target_etype=target_etype,
                                     regression=True)
        else:
            assert False, "decoder not supported"
        model.set_decoder(decoder)
        model.set_loss_func(RegressionLossFunc())
    else:
        raise ValueError('unknown node task: {}'.format(config.task_type))
    if train_task:
        model.init_optimizer(lr=config.lr, sparse_lr=config.sparse_lr,
                             weight_decay=config.wd_l2norm)
    return model

def create_builtin_lp_gnn_model(g, config, train_task):
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
    model = GSgnnLinkPredictionModel(config.alpha_l2norm)
    set_gnn_encoder(model, g, config, train_task)
    num_train_etype = len(config.train_etype)
    # For backword compatibility, we add this check.
    # if train etype is 1, There is no need to use DistMult
    assert num_train_etype > 1 or config.use_dot_product, \
            "If number of train etype is 1, please use dot product"
    if config.use_dot_product:
        # if the training set only contains one edge type or it is specified in the arguments,
        # we use dot product as the score function.
        if get_rank() == 0:
            print('use dot product for single-etype task.')
            print("Using inner product objective for supervision")
        decoder = LinkPredictDotDecoder(model.gnn_encoder.out_dims)
    else:
        if get_rank() == 0:
            print("Using distmult objective for supervision")
        decoder = LinkPredictDistMultDecoder(g.canonical_etypes,
                                             model.gnn_encoder.out_dims,
                                             config.gamma)
    model.set_decoder(decoder)
    model.set_loss_func(LinkPredictLossFunc())
    if train_task:
        model.init_optimizer(lr=config.lr, sparse_lr=config.sparse_lr,
                             weight_decay=config.wd_l2norm)
    return model

def set_gnn_encoder(model, g, config, train_task):
    """ Create GNN encoder.

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    config: GSConfig
        Configurations
    train_task : bool
        Whether this model is used for training.
    """
    # Set input layer
    feat_size = get_feat_size(g, config.feat_name)
    encoder = GSNodeInputLayer(g, feat_size, config.n_hidden,
                               dropout=config.dropout,
                               use_node_embeddings=config.use_node_embeddings)
    model.set_node_input_encoder(encoder)

    # Set GNN encoders
    model_encoder_type = config.model_encoder_type
    dropout = config.dropout if train_task else 0
    if model_encoder_type == "rgcn":
        n_bases = config.n_bases
        # we need to set the n_layers -1 because there is an output layer
        # that is hard coded.
        gnn_encoder = RelationalGCNEncoder(g,
                                           config.n_hidden, config.n_hidden,
                                           num_bases=n_bases,
                                           num_hidden_layers=config.n_layers -1,
                                           dropout=dropout,
                                           use_self_loop=config.use_self_loop)
    elif model_encoder_type == "rgat":
        # we need to set the n_layers -1 because there is an output layer that is hard coded.
        gnn_encoder = RelationalGATEncoder(g,
                                           config.n_hidden,
                                           config.n_hidden,
                                           config.n_heads,
                                           num_hidden_layers=config.n_layers -1,
                                           dropout=dropout,
                                           use_self_loop=config.use_self_loop)
    else:
        assert False, "Unknown gnn model type {}".format(model_encoder_type)
    model.set_gnn_encoder(gnn_encoder)

def create_builtin_task_tracker(config, rank):
    tracker_class = get_task_tracker_class(config.task_tracker)
    return tracker_class(config, rank)
