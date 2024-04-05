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

    GSF utility functions.
"""

import os
import logging
import numpy as np
import dgl
import torch as th
import torch.nn.functional as F
from dgl.distributed import role
from dgl.distributed.constants import DEFAULT_NTYPE
from dgl.distributed.constants import DEFAULT_ETYPE

from .utils import sys_tracker, get_rank
from .utils import setup_device
from .config import BUILTIN_TASK_NODE_CLASSIFICATION
from .config import BUILTIN_TASK_NODE_REGRESSION
from .config import BUILTIN_TASK_EDGE_CLASSIFICATION
from .config import BUILTIN_TASK_EDGE_REGRESSION
from .config import BUILTIN_LP_DOT_DECODER
from .config import BUILTIN_LP_DISTMULT_DECODER
from .config import (BUILTIN_LP_LOSS_CROSS_ENTROPY,
                     BUILTIN_LP_LOSS_CONTRASTIVELOSS)
from .model.embed import GSNodeEncoderInputLayer
from .model.lm_embed import GSLMNodeEncoderInputLayer, GSPureLMNodeInputLayer
from .model.rgcn_encoder import RelationalGCNEncoder, RelGraphConvLayer
from .model.rgat_encoder import RelationalGATEncoder
from .model.hgt_encoder import HGTEncoder
from .model.gnn_with_reconstruct import GNNEncoderWithReconstructedEmbed
from .model.sage_encoder import SAGEEncoder
from .model.gat_encoder import GATEncoder
from .model.gatv2_encoder import GATv2Encoder
from .model.node_gnn import GSgnnNodeModel
from .model.node_glem import GLEM
from .model.edge_gnn import GSgnnEdgeModel
from .model.lp_gnn import GSgnnLinkPredictionModel
from .model.loss_func import (ClassifyLossFunc,
                              RegressionLossFunc,
                              LinkPredictBCELossFunc,
                              WeightedLinkPredictBCELossFunc,
                              LinkPredictContrastiveLossFunc)
from .model.node_decoder import EntityClassifier, EntityRegression
from .model.edge_decoder import (DenseBiDecoder,
                                 MLPEdgeDecoder,
                                 MLPEFeatEdgeDecoder)
from .model.edge_decoder import (LinkPredictDotDecoder,
                                 LinkPredictDistMultDecoder,
                                 LinkPredictContrastiveDotDecoder,
                                 LinkPredictContrastiveDistMultDecoder,
                                 LinkPredictWeightedDotDecoder,
                                 LinkPredictWeightedDistMultDecoder)
from .tracker import get_task_tracker_class

def initialize(ip_config=None, backend='gloo', local_rank=0, use_wholegraph=False):
    """ Initialize distributed training and inference context.

    .. code::

        # Standalone mode
        import graphstorm as gs
        gs.initialize()

    .. code::

        # distributed mode
        import graphstorm as gs
        gs.initialize(ip_config="/tmp/ip_list.txt", backend="gloo")

    Parameters
    ----------
    ip_config: str
        File path of ip_config file, e.g., `/tmp/ip_list.txt`
        Default: None
    backend: str
        Torch distributed backend, e.g., ``gloo`` or ``nccl``.
        Default: 'gloo'
    local_rank: int
        The local rank of the current process.
        Default: 0
    use_wholegraph: bool
        Whether to use wholegraph for feature transfer.
        Default: False
    """
    # We need to use socket for communication in DGL 0.8. The tensorpipe backend has a bug.
    # This problem will be fixed in the future.
    dgl.distributed.initialize(ip_config, net_type='socket')
    assert th.cuda.is_available() or backend == "gloo", "Gloo backend required for a CPU setting."
    if ip_config is not None:
        th.distributed.init_process_group(backend=backend)
        # Use wholegraph for feature and label fetching
        if use_wholegraph:
            from .wholegraph import init_wholegraph
            init_wholegraph()

    sys_tracker.check("load DistDGL")
    device = setup_device(local_rank)
    sys_tracker.check(f"setup device on {device}")

def get_node_feat_size(g, node_feat_names):
    """ Get the feature's size on each node type in the input graph.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.
    node_feat_names : str or dict of str
        The node feature names.

    Returns
    -------
    dict of int : the feature size for each node type.

    """
    node_feat_size = {}
    for ntype in g.ntypes:
        # user can specify the name of the field
        if node_feat_names is None:
            feat_name = None
        elif isinstance(node_feat_names, dict) and ntype in node_feat_names:
            feat_name = node_feat_names[ntype]
        elif isinstance(node_feat_names, str):
            feat_name = node_feat_names
        else:
            feat_name = None

        if feat_name is None:
            node_feat_size[ntype] = 0
        elif isinstance(feat_name, str): # global feat_name
            # We force users to know which node type has node feature
            # This helps avoid unexpected training behavior.
            assert feat_name in g.nodes[ntype].data, \
                    f"Warning. The feature \"{feat_name}\" " \
                    f"does not exists for the node type \"{ntype}\"."
            node_feat_size[ntype] = np.prod(g.nodes[ntype].data[feat_name].shape[1:])
        else:
            node_feat_size[ntype] = 0
            for fname in feat_name:
                # We force users to know which node type has node feature
                # This helps avoid unexpected training behavior.
                assert fname in g.nodes[ntype].data, \
                        f"Warning. The feature \"{fname}\" " \
                        f"does not exists for the node type \"{ntype}\"."
                # TODO: we only allow an input node feature as a 2D tensor
                # Support 1D or nD when required.
                assert len(g.nodes[ntype].data[fname].shape) == 2, \
                    "Input node features should be 2D tensors"
                fsize = np.prod(g.nodes[ntype].data[fname].shape[1:])
                node_feat_size[ntype] += fsize
    return node_feat_size

def get_rel_names_for_reconstruct(g, reconstructed_embed_ntype, feat_size):
    """ Get the relation types for reconstructing node features.

    Parameters
    ----------
    g : DistGraph
        The input graph.
    reconstructed_embed_ntype : list of str
        The node type that requires to reconstruct node features.
    feat_size : dict of int
        The feature size on each node type.

    Returns
    -------
    list of tuples : the relation types for reconstructing node features.
    """
    etypes = g.canonical_etypes
    reconstruct_etypes = []
    for dst_ntype in reconstructed_embed_ntype:
        if feat_size[dst_ntype] > 0:
            logging.warning("Node %s already have features. " \
                    + "No need to reconstruct their features.", dst_ntype)
        for etype in etypes:
            src_type = etype[0]
            if etype[2] == dst_ntype and feat_size[src_type] > 0:
                reconstruct_etypes.append(etype)
    return reconstruct_etypes

def create_builtin_node_gnn_model(g, config, train_task):
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
    return create_builtin_node_model(g, config, train_task)

def create_builtin_node_model(g, config, train_task):
    """ Create a built-in model for node prediction.

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
    if config.training_method["name"] == "glem":
        model = GLEM(config.alpha_l2norm, config.target_ntype, **config.training_method["kwargs"])
    elif config.training_method["name"] == "default":
        model = GSgnnNodeModel(config.alpha_l2norm)
    set_encoder(model, g, config, train_task)

    if config.task_type == BUILTIN_TASK_NODE_CLASSIFICATION:
        if not isinstance(config.num_classes, dict):
            model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims \
                                                if model.gnn_encoder is not None \
                                                else model.node_input_encoder.out_dims,
                                               config.num_classes,
                                               config.multilabel))
            model.set_loss_func(ClassifyLossFunc(config.multilabel,
                                             config.multilabel_weights,
                                             config.imbalance_class_weights))
        else:
            decoder = {}
            loss_func = {}
            for ntype in config.target_ntype:
                decoder[ntype] = EntityClassifier(model.gnn_encoder.out_dims \
                                                if model.gnn_encoder is not None \
                                                else model.node_input_encoder.out_dims,
                                               config.num_classes[ntype],
                                               config.multilabel[ntype])
                loss_func[ntype] = ClassifyLossFunc(config.multilabel[ntype],
                                                config.multilabel_weights[ntype],
                                                config.imbalance_class_weights[ntype])

            model.set_decoder(decoder)
            model.set_loss_func(loss_func)

    elif config.task_type == BUILTIN_TASK_NODE_REGRESSION:
        model.set_decoder(EntityRegression(model.gnn_encoder.out_dims \
                                            if model.gnn_encoder is not None \
                                            else model.node_input_encoder.out_dims))
        model.set_loss_func(RegressionLossFunc())
    else:
        raise ValueError('unknown node task: {}'.format(config.task_type))
    if train_task:
        model.init_optimizer(lr=config.lr, sparse_optimizer_lr=config.sparse_optimizer_lr,
                             weight_decay=config.wd_l2norm,
                             lm_lr=config.lm_tune_lr)
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
    return create_builtin_edge_model(g, config, train_task)

def create_builtin_edge_model(g, config, train_task):
    """ Create a model for edge prediction.

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
    set_encoder(model, g, config, train_task)
    if config.task_type == BUILTIN_TASK_EDGE_CLASSIFICATION:
        num_classes = config.num_classes
        decoder_type = config.decoder_type
        dropout = config.dropout if train_task else 0
        # TODO(zhengda) we should support multiple target etypes
        target_etype = config.target_etype[0]
        if decoder_type == "DenseBiDecoder":
            num_decoder_basis = config.num_decoder_basis
            assert config.num_ffn_layers_in_decoder == 0, \
                "DenseBiDecoder does not support adding extra feedforward neural network layers" \
                "You can increases num_basis to increase the parameter size."
            decoder = DenseBiDecoder(in_units=model.gnn_encoder.out_dims \
                                        if model.gnn_encoder is not None \
                                        else model.node_input_encoder.out_dims,
                                     num_classes=num_classes,
                                     multilabel=config.multilabel,
                                     num_basis=num_decoder_basis,
                                     dropout_rate=dropout,
                                     regression=False,
                                     target_etype=target_etype)
        elif decoder_type == "MLPDecoder":
            decoder = MLPEdgeDecoder(model.gnn_encoder.out_dims \
                                        if model.gnn_encoder is not None \
                                        else model.node_input_encoder.out_dims,
                                     num_classes,
                                     multilabel=config.multilabel,
                                     target_etype=target_etype,
                                     num_ffn_layers=config.num_ffn_layers_in_decoder)
        elif decoder_type == "MLPEFeatEdgeDecoder":
            decoder_edge_feat = config.decoder_edge_feat
            assert decoder_edge_feat is not None, \
                "decoder-edge-feat must be provided when " \
                "decoder_type == MLPEFeatEdgeDecoder"
            # We need to get the edge_feat input dim.
            if isinstance(decoder_edge_feat, str):
                assert decoder_edge_feat in g.edges[target_etype].data
                feat_dim = g.edges[target_etype].data[decoder_edge_feat].shape[-1]
            else:
                feat_dim = sum([g.edges[target_etype].data[fname].shape[-1] \
                    for fname in decoder_edge_feat[target_etype]])

            decoder = MLPEFeatEdgeDecoder(
                h_dim=model.gnn_encoder.out_dims \
                    if model.gnn_encoder is not None \
                    else model.node_input_encoder.out_dims,
                feat_dim=feat_dim,
                out_dim=num_classes,
                multilabel=config.multilabel,
                target_etype=target_etype,
                dropout=config.dropout,
                num_ffn_layers=config.num_ffn_layers_in_decoder)
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
            decoder = DenseBiDecoder(model.gnn_encoder.out_dims \
                                        if model.gnn_encoder is not None \
                                        else model.node_input_encoder.out_dims,
                                     1,
                                     num_basis=num_decoder_basis,
                                     multilabel=False,
                                     target_etype=target_etype,
                                     dropout_rate=dropout,
                                     regression=True)
        elif decoder_type == "MLPDecoder":
            decoder = MLPEdgeDecoder(model.gnn_encoder.out_dims \
                                        if model.gnn_encoder is not None \
                                        else model.node_input_encoder.out_dims,
                                     1,
                                     multilabel=False,
                                     target_etype=target_etype,
                                     regression=True)
        elif decoder_type == "MLPEFeatEdgeDecoder":
            decoder_edge_feat = config.decoder_edge_feat
            assert decoder_edge_feat is not None, \
                "decoder-edge-feat must be provided when " \
                "decoder_type == MLPEFeatEdgeDecoder"
            # We need to get the edge_feat input dim.
            if isinstance(decoder_edge_feat, str):
                assert decoder_edge_feat in g.edges[target_etype].data
                feat_dim = g.edges[target_etype].data[decoder_edge_feat].shape[-1]
            else:
                feat_dim = sum([g.edges[target_etype].data[fname].shape[-1] \
                    for fname in decoder_edge_feat[target_etype]])

            decoder = MLPEFeatEdgeDecoder(
                h_dim=model.gnn_encoder.out_dims \
                    if model.gnn_encoder is not None \
                    else model.node_input_encoder.out_dims,
                feat_dim=feat_dim,
                out_dim=1,
                multilabel=False,
                target_etype=target_etype,
                dropout=config.dropout,
                regression=True)
        else:
            assert False, "decoder not supported"
        model.set_decoder(decoder)
        model.set_loss_func(RegressionLossFunc())
    else:
        raise ValueError('unknown node task: {}'.format(config.task_type))
    if train_task:
        model.init_optimizer(lr=config.lr, sparse_optimizer_lr=config.sparse_optimizer_lr,
                             weight_decay=config.wd_l2norm,
                             lm_lr=config.lm_tune_lr)
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
    return create_builtin_lp_model(g, config, train_task)

def create_builtin_lp_model(g, config, train_task):
    """ Create a model for link prediction.

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
    GSgnnModel : The model.
    """
    model = GSgnnLinkPredictionModel(config.alpha_l2norm,
                                     config.lp_embed_normalizer)
    set_encoder(model, g, config, train_task)
    num_train_etype = len(config.train_etype) \
        if config.train_etype is not None \
        else len(g.canonical_etypes) # train_etype is None, every etype is used for training
    # For backword compatibility, we add this check.
    # if train etype is 1, There is no need to use DistMult
    assert num_train_etype > 1 or config.lp_decoder_type == BUILTIN_LP_DOT_DECODER, \
            "If number of train etype is 1, please use dot product"
    out_dims = model.gnn_encoder.out_dims \
                    if model.gnn_encoder is not None \
                    else model.node_input_encoder.out_dims
    if config.lp_decoder_type == BUILTIN_LP_DOT_DECODER:
        # if the training set only contains one edge type or it is specified in the arguments,
        # we use dot product as the score function.
        if get_rank() == 0:
            logging.debug('use dot product for single-etype task.')
            logging.debug("Using inner product objective for supervision")
        if config.lp_edge_weight_for_loss is None:
            decoder = LinkPredictContrastiveDotDecoder(out_dims) \
                if config.lp_loss_func == BUILTIN_LP_LOSS_CONTRASTIVELOSS else \
                LinkPredictDotDecoder(out_dims)
        else:
            decoder = LinkPredictWeightedDotDecoder(out_dims,
                                                    config.lp_edge_weight_for_loss)
    elif config.lp_decoder_type == BUILTIN_LP_DISTMULT_DECODER:
        if get_rank() == 0:
            logging.debug("Using distmult objective for supervision")
        if config.lp_edge_weight_for_loss is None:
            decoder = LinkPredictContrastiveDistMultDecoder(g.canonical_etypes,
                                                            out_dims,
                                                            config.gamma) \
                if config.lp_loss_func == BUILTIN_LP_LOSS_CONTRASTIVELOSS else \
                LinkPredictDistMultDecoder(g.canonical_etypes,
                                           out_dims,
                                           config.gamma)
        else:
            decoder = LinkPredictWeightedDistMultDecoder(g.canonical_etypes,
                                                         out_dims,
                                                         config.gamma,
                                                         config.lp_edge_weight_for_loss)
    else:
        raise Exception(f"Unknow link prediction decoder type {config.lp_decoder_type}")
    model.set_decoder(decoder)
    if config.lp_loss_func == BUILTIN_LP_LOSS_CONTRASTIVELOSS:
        model.set_loss_func(LinkPredictContrastiveLossFunc(config.contrastive_loss_temperature))
    elif config.lp_loss_func == BUILTIN_LP_LOSS_CROSS_ENTROPY:
        if config.lp_edge_weight_for_loss is None:
            model.set_loss_func(LinkPredictBCELossFunc())
        else:
            model.set_loss_func(WeightedLinkPredictBCELossFunc())
    else:
        raise TypeError(f"Unknown link prediction loss function {config.lp_loss_func}")
    if train_task:
        model.init_optimizer(lr=config.lr, sparse_optimizer_lr=config.sparse_optimizer_lr,
                             weight_decay=config.wd_l2norm,
                             lm_lr=config.lm_tune_lr)
    return model

def set_encoder(model, g, config, train_task):
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
    feat_size = get_node_feat_size(g, config.node_feat_name)
    reconstruct_feats = len(config.construct_feat_ntype) > 0
    model_encoder_type = config.model_encoder_type
    if config.node_lm_configs is not None:
        emb_path = os.path.join(os.path.dirname(config.part_config),
                "cached_embs") if config.cache_lm_embed else None
        if model_encoder_type == "lm":
            # only use language model(s) as input layer encoder(s)
            encoder = GSPureLMNodeInputLayer(g, config.node_lm_configs,
                                            num_train=config.lm_train_nodes,
                                            lm_infer_batch_size=config.lm_infer_batch_size,
                                            cached_embed_path=emb_path,
                                            wg_cached_embed=config.use_wholegraph_embed)
        else:
            encoder = GSLMNodeEncoderInputLayer(g, config.node_lm_configs,
                                                feat_size, config.hidden_size,
                                                num_train=config.lm_train_nodes,
                                                lm_infer_batch_size=config.lm_infer_batch_size,
                                                dropout=config.dropout,
                                                use_node_embeddings=config.use_node_embeddings,
                                                cached_embed_path=emb_path,
                                                wg_cached_embed=config.use_wholegraph_embed,
                                                force_no_embeddings=config.construct_feat_ntype)
    else:
        encoder = GSNodeEncoderInputLayer(g, feat_size, config.hidden_size,
                                          dropout=config.dropout,
                                          activation=config.input_activate,
                                          use_node_embeddings=config.use_node_embeddings,
                                          force_no_embeddings=config.construct_feat_ntype,
                                          num_ffn_layers_in_input=config.num_ffn_layers_in_input,
                                          use_wholegraph_sparse_emb=config.use_wholegraph_embed)
    # The number of feature dimensions can change. For example, the feature dimensions
    # of BERT embeddings are determined when the input encoder is created.
    feat_size = encoder.in_dims
    model.set_node_input_encoder(encoder)

    # Set GNN encoders
    dropout = config.dropout if train_task else 0
    if model_encoder_type == "mlp" or model_encoder_type == "lm":
        # Only input encoder is used
        assert config.num_layers == 0, "No GNN layers"
        gnn_encoder = None
    elif model_encoder_type == "rgcn":
        num_bases = config.num_bases
        # we need to set the num_layers -1 because there is an output layer
        # that is hard coded.
        gnn_encoder = RelationalGCNEncoder(g,
                                           config.hidden_size, config.hidden_size,
                                           num_bases=num_bases,
                                           num_hidden_layers=config.num_layers -1,
                                           dropout=dropout,
                                           use_self_loop=config.use_self_loop,
                                           num_ffn_layers_in_gnn=config.num_ffn_layers_in_gnn,
                                           norm=config.gnn_norm)
    elif model_encoder_type == "rgat":
        # we need to set the num_layers -1 because there is an output layer that is hard coded.
        gnn_encoder = RelationalGATEncoder(g,
                                           config.hidden_size,
                                           config.hidden_size,
                                           config.num_heads,
                                           num_hidden_layers=config.num_layers -1,
                                           dropout=dropout,
                                           use_self_loop=config.use_self_loop,
                                           num_ffn_layers_in_gnn=config.num_ffn_layers_in_gnn,
                                           norm=config.gnn_norm)
    elif model_encoder_type == "hgt":
        # we need to set the num_layers -1 because there is an output layer that is hard coded.
        gnn_encoder = HGTEncoder(g,
                                 config.hidden_size,
                                 config.hidden_size,
                                 num_hidden_layers=config.num_layers -1,
                                 num_heads=config.num_heads,
                                 dropout=dropout,
                                 norm=config.gnn_norm,
                                 num_ffn_layers_in_gnn=config.num_ffn_layers_in_gnn)
    elif model_encoder_type == "sage":
        # we need to check if the graph is homogeneous
        assert check_homo(g) == True, \
            'The graph is not a homogeneous graph, can not use sage model encoder'
        # we need to set the num_layers -1 because there is an output layer that is hard coded.
        gnn_encoder = SAGEEncoder(h_dim=config.hidden_size,
                                  out_dim=config.hidden_size,
                                  num_hidden_layers=config.num_layers - 1,
                                  dropout=dropout,
                                  aggregator_type='pool',
                                  num_ffn_layers_in_gnn=config.num_ffn_layers_in_gnn,
                                  norm=config.gnn_norm)
    elif model_encoder_type == "gat":
        # we need to check if the graph is homogeneous
        assert check_homo(g) == True, \
            'The graph is not a homogeneous graph, can not use gat model encoder'
        gnn_encoder = GATEncoder(h_dim=config.hidden_size,
                                 out_dim=config.hidden_size,
                                 num_heads=config.num_heads,
                                 num_hidden_layers=config.num_layers -1,
                                 dropout=dropout,
                                 num_ffn_layers_in_gnn=config.num_ffn_layers_in_gnn)
    elif model_encoder_type == "gatv2":
        # we need to check if the graph is homogeneous
        assert check_homo(g) == True, \
            'The graph is not a homogeneous graph, can not use gatv2 model encoder'
        gnn_encoder = GATv2Encoder(h_dim=config.hidden_size,
                                   out_dim=config.hidden_size,
                                   num_heads=config.num_heads,
                                   num_hidden_layers=config.num_layers -1,
                                   dropout=dropout,
                                   num_ffn_layers_in_gnn=config.num_ffn_layers_in_gnn)
    else:
        assert False, "Unknown gnn model type {}".format(model_encoder_type)

    if reconstruct_feats:
        rel_names = get_rel_names_for_reconstruct(g, config.construct_feat_ntype, feat_size)
        dst_types = set([rel_name[2] for rel_name in rel_names])
        for ntype in config.construct_feat_ntype:
            assert ntype in dst_types, \
                    f"We cannot reconstruct features of node {ntype} " \
                    + "probably because their neighbors don't have features."
        assert config.construct_feat_encoder == "rgcn", \
                "We only support RGCN for reconstructing node features."
        input_gnn = RelGraphConvLayer(config.hidden_size, config.hidden_size,
                                      rel_names, len(rel_names),
                                      self_loop=False, # We should disable self loop so that
                                                       # the encoder doesn't use dest node
                                                       # features.
                                      activation=F.relu,
                                      num_ffn_layers_in_gnn=config.num_ffn_layers_in_input)
        gnn_encoder = GNNEncoderWithReconstructedEmbed(gnn_encoder, input_gnn, rel_names)
    model.set_gnn_encoder(gnn_encoder)

def check_homo(g):
    """ Check if it is a valid homogeneous graph

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    """
    if g.ntypes == [DEFAULT_NTYPE] and g.etypes == [DEFAULT_ETYPE[1]]:
        return True
    return False


def create_builtin_task_tracker(config):
    tracker_class = get_task_tracker_class(config.task_tracker)
    return tracker_class(config.eval_frequency)
