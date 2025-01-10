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
import importlib.metadata

import numpy as np
import dgl
import torch as th
import torch.nn.functional as F

from dgl.distributed.constants import DEFAULT_NTYPE
from dgl.distributed.constants import DEFAULT_ETYPE
from packaging import version

from .utils import sys_tracker, get_rank
from .utils import setup_device
from .config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                     BUILTIN_TASK_NODE_REGRESSION,
                     BUILTIN_TASK_EDGE_CLASSIFICATION,
                     BUILTIN_TASK_EDGE_REGRESSION,
                     BUILTIN_TASK_LINK_PREDICTION,
                     BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                     BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT)
from .config import (BUILTIN_LP_DOT_DECODER,
                     BUILTIN_LP_DISTMULT_DECODER,
                     BUILTIN_LP_ROTATE_DECODER,
                     BUILTIN_LP_TRANSE_L1_DECODER,
                     BUILTIN_LP_TRANSE_L2_DECODER)
from .config import (BUILTIN_LP_LOSS_CROSS_ENTROPY,
                     BUILTIN_LP_LOSS_CONTRASTIVELOSS,
                     BUILTIN_CLASS_LOSS_CROSS_ENTROPY,
                     BUILTIN_CLASS_LOSS_FOCAL)
from .eval.eval_func import (
    SUPPORTED_HIT_AT_METRICS,
    SUPPORTED_LINK_PREDICTION_METRICS)
from .model.embed import GSNodeEncoderInputLayer, GSEdgeEncoderInputLayer
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
                              LinkPredictAdvBCELossFunc,
                              WeightedLinkPredictAdvBCELossFunc,
                              LinkPredictContrastiveLossFunc,
                              FocalLossFunc)

from .model.node_decoder import EntityClassifier, EntityRegression
from .model.edge_decoder import (DenseBiDecoder,
                                 MLPEdgeDecoder,
                                 MLPEFeatEdgeDecoder,
                                 EdgeRegression)
from .model.edge_decoder import (LinkPredictDotDecoder,
                                 LinkPredictDistMultDecoder,
                                 LinkPredictContrastiveDotDecoder,
                                 LinkPredictContrastiveDistMultDecoder,
                                 LinkPredictWeightedDotDecoder,
                                 LinkPredictWeightedDistMultDecoder,
                                 LinkPredictRotatEDecoder,
                                 LinkPredictContrastiveRotatEDecoder,
                                 LinkPredictWeightedRotatEDecoder,
                                 LinkPredictTransEDecoder,
                                 LinkPredictContrastiveTransEDecoder,
                                 LinkPredictWeightedTransEDecoder)
from .dataloading import (BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                          BUILTIN_LP_JOINT_NEG_SAMPLER,
                          BUILTIN_LP_INBATCH_JOINT_NEG_SAMPLER,
                          BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER,
                          BUILTIN_LP_LOCALJOINT_NEG_SAMPLER,
                          BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER,
                          BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER,
                          BUILTIN_FAST_LP_UNIFORM_NEG_SAMPLER,
                          BUILTIN_FAST_LP_JOINT_NEG_SAMPLER,
                          BUILTIN_FAST_LP_LOCALUNIFORM_NEG_SAMPLER,
                          BUILTIN_FAST_LP_LOCALJOINT_NEG_SAMPLER)
from .dataloading import (FastGSgnnLinkPredictionDataLoader,
                          FastGSgnnLPJointNegDataLoader,
                          FastGSgnnLPLocalUniformNegDataLoader,
                          FastGSgnnLPLocalJointNegDataLoader,
                          GSgnnLinkPredictionDataLoader,
                          GSgnnLPJointNegDataLoader,
                          GSgnnLPLocalUniformNegDataLoader,
                          GSgnnLPLocalJointNegDataLoader,
                          GSgnnLPInBatchJointNegDataLoader,
                          GSgnnAllEtypeLPJointNegDataLoader,
                          GSgnnAllEtypeLinkPredictionDataLoader)
from .dataloading import (GSgnnLinkPredictionTestDataLoader,
                          GSgnnLinkPredictionJointTestDataLoader,
                          GSgnnLinkPredictionPredefinedTestDataLoader)

from .eval import (GSgnnClassificationEvaluator,
                   GSgnnRegressionEvaluator,
                   GSgnnRconstructFeatRegScoreEvaluator,
                   GSgnnPerEtypeLPEvaluator,
                   GSgnnLPEvaluator)

from .tracker import get_task_tracker_class

def initialize(
        ip_config=None,
        backend='gloo',
        local_rank=0,
        use_wholegraph=False,
        use_graphbolt=False,
    ):
    """ Initialize distributed training and inference context. For GraphStorm Standalone mode,
    no argument is needed. For Distributed mode, users need to provide an IP address list file.

    .. code::

        # Standalone mode
        import graphstorm as gs
        gs.initialize()

    .. code::

        # distributed mode
        import graphstorm as gs
        gs.initialize(ip_config="/tmp/ip_list.txt")

    Parameters
    ----------
    ip_config: str
        File path of the IP address file, e.g., `/tmp/ip_list.txt`
        Default: None.
    backend: str
        Torch distributed backend, e.g., ``gloo`` or ``nccl``.
        Default: ``gloo``.
    local_rank: int
        The local rank of the current process.
        Default: 0.
    use_wholegraph: bool
        Whether to use wholegraph for feature transfer.
        Default: False.
    use_graphbolt: bool
        Whether to use GraphBolt graph representation.
        Requires installed DGL version to be at least ``2.1.0``.
        Default: False.

        .. versionadded:: 0.4.0
    """
    dgl_version = importlib.metadata.version('dgl')
    if version.parse(dgl_version) >= version.parse("2.1.0"):
        dgl.distributed.initialize( # pylint: disable=unexpected-keyword-arg
            ip_config,
            net_type='socket',
            use_graphbolt=use_graphbolt,
        )
    else:
        if use_graphbolt:
            raise ValueError(
                f"use_graphbolt was 'true' but but DGL version was {dgl_version}. "
                "GraphBolt DGL initialization requires DGL version >= 2.1.0"
            )
        dgl.distributed.initialize(
            ip_config,
            net_type='socket',
        )
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
    """ Get the overall feature size of each node type with feature names specified in the
    ``node_feat_names``. If a node type has multiple features, the returned feature size
    will be the sum of the sizes of these features for that node type.

    Parameters
    ----------
    g : DistGraph
        A DGL distributed graph.
    node_feat_names : str, or dict of list of str
        The node feature names. A string indicates that all nodes share the same feature name,
        while a dictionary with a list of strings indicates that each
        node type has different node feature names.

    Returns
    -------
    node_feat_size: dict of int
        The feature size for the node types and feature names specified in the
        ``node_feat_names``. If feature name is not specified, the feature size
        will be 0.
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

def get_edge_feat_size(g, edge_feat_names):
    """ Get the overall feature size of each edge type with feature names specified in the
    ``edge_feat_names``. If an edge type has multiple features, the returned feature size
    will be the sum of the sizes of these features for that edge type.

    .. versionadded:: 0.4.0
        Add the `get_edge_feat_size` in v0.4.0 to support edge features.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.
    edge_feat_names : str, or dict of list of str
        The edge feature names.

    Returns
    -------
    edge_feat_size: dict of int
        The feature size for each edge type. If feature name is not specified, the feature size
        will be 0.
    """
    # check if edge types in edge_feat_names are in graph
    if edge_feat_names and isinstance(edge_feat_names, dict):
        for etype in edge_feat_names.keys():
            assert etype in list(g.canonical_etypes), \
                f"Graph data does not contain the specified edge type {etype}!, " + \
                "Please check the values of \'edge_feat_names\' variable."

    edge_feat_size = {}
    for canonical_etype in g.canonical_etypes:
        # user can specify the name of the field or do nothing
        if edge_feat_names is None:
            feat_name = None
        elif isinstance(edge_feat_names, dict) and canonical_etype in edge_feat_names:
            feat_name = edge_feat_names[canonical_etype]
        elif isinstance(edge_feat_names, str):
            feat_name = edge_feat_names
        else:
            feat_name = None

        if feat_name is None:
            edge_feat_size[canonical_etype] = 0
        elif isinstance(feat_name, str): # global feat_name for all edge types
            # We force users to know which edge type has edge feature
            # This helps avoid unexpected training behavior.
            assert feat_name in g.edges[canonical_etype].data, \
                    f"Warning. The feature \"{feat_name}\" " \
                    f"does not exists for the edge type \"{canonical_etype}\"."
            edge_feat_size[canonical_etype] = \
                np.prod(g.edges[canonical_etype].data[feat_name].shape[1:])
        else:
            edge_feat_size[canonical_etype] = 0
            for fname in feat_name:
                # We force users to know which node type has node feature
                # This helps avoid unexpected training behavior.
                assert fname in g.edges[canonical_etype].data, \
                        f"Warning. The feature \"{fname}\" " \
                        f"does not exist for the edge type \"{canonical_etype}\"."
                # TODO: we only allow an input node feature as a 2D tensor
                # Support 1D or nD when required.
                assert len(g.edges[canonical_etype].data[fname].shape) == 2, \
                    "Input edge features should be 2D tensors"
                fsize = np.prod(g.edges[canonical_etype].data[fname].shape[1:])
                edge_feat_size[canonical_etype] += fsize

    return edge_feat_size

def get_rel_names_for_reconstruct(g, reconstructed_embed_ntype, feat_size):
    """ Get the edge type list for reconstructing node features.

    Parameters
    ----------
    g : DistGraph
        A DGL distributed graph.
    reconstructed_embed_ntype : list of str
        The node types for which node features need to be reconstructed.
    feat_size : dict of int
        The feature size on each node type in the format of {"ntype": size}.

    Returns
    -------
    reconstruct_etypes: list of tuples
        The edge types whose destination nodes required for reconstructing node features.
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

# pylint: disable=unused-argument
def create_builtin_reconstruct_nfeat_decoder(g, decoder_input_dim, config, train_task):
    """ create builtin node feature reconstruction decoder
        according to task config.

    Parameters
    ----------
    g: DGLGraph
        The graph data.
        Note(xiang): Make it consistent with create_builtin_edge_decoder.
        Reserved for future.
    decoder_input_dim: int
        Input dimension size of the decoder.
    config: GSConfig
        Configurations.
    train_task : bool
        Whether this model is used for training.

    Returns
    -------
    decoder: The node task decoder(s).
    loss_func: The loss function(s).
    """
    dropout = config.dropout if train_task else 0
    target_ntype = config.target_ntype
    reconstruct_feat = config.reconstruct_nfeat_name
    feat_dim = g.nodes[target_ntype].data[reconstruct_feat].shape[1]

    decoder = EntityRegression(decoder_input_dim,
                               dropout=dropout,
                               out_dim=feat_dim,
                               use_bias=config.decoder_bias)

    loss_func = RegressionLossFunc()
    return decoder, loss_func

# pylint: disable=unused-argument
def create_builtin_reconstruct_efeat_decoder(g, decoder_input_dim, config, train_task):
    """ create builtin edge feature reconstruction decoder
        according to task config

    Parameters
    ----------
    g: DGLGraph
        The graph data.
    decoder_input_dim: int
        Input dimension size of the decoder.
    config: GSConfig
        Configurations.
    train_task : bool
        Whether this model is used for training.

    Returns
    -------
    decoder: The edge feature reconstruction decoder(s).
    loss_func: The loss function(s).

    .. versionadded:: 0.4.0
    """
    dropout = config.dropout if train_task else 0
    # Only support one edge type per edge feature
    # reconstruction task.
    target_etype = config.target_etype[0]
    reconstruct_feat = config.reconstruct_efeat_name
    assert len(g.edges[target_etype].data[reconstruct_feat].shape) == 2, \
        "The edge feature {reconstruct_feat} of {target_etype} edges " \
        f"Must be a 2D tensor, but got {g.edges[target_etype].data[reconstruct_feat].shape}."
    feat_dim = g.edges[target_etype].data[reconstruct_feat].shape[1]

    decoder = EdgeRegression(decoder_input_dim,
                             target_etype=target_etype,
                             out_dim=feat_dim,
                             dropout=dropout,
                             use_bias=config.decoder_bias)

    loss_func = RegressionLossFunc()
    return decoder, loss_func

# pylint: disable=unused-argument
def create_builtin_node_decoder(g, decoder_input_dim, config, train_task):
    """ create builtin node decoder according to task config

    Parameters
    ----------
    g: DGLGraph
        The graph data.
        Note(xiang): Make it consistent with create_builtin_edge_decoder.
        Reserved for future.
    decoder_input_dim: int
        Input dimension size of the decoder
    config: GSConfig
        Configurations
    train_task : bool
        Whether this model is used for training.

    Returns
    -------
    decoder: The node task decoder(s)
    loss_func: The loss function(s)
    """
    dropout = config.dropout if train_task else 0
    if config.task_type == BUILTIN_TASK_NODE_CLASSIFICATION:
        if not isinstance(config.num_classes, dict):
            decoder = EntityClassifier(decoder_input_dim,
                                       config.num_classes,
                                       config.multilabel,
                                       dropout=dropout,
                                       norm=config.decoder_norm,
                                       use_bias=config.decoder_bias)
            if config.class_loss_func == BUILTIN_CLASS_LOSS_CROSS_ENTROPY:
                loss_func = ClassifyLossFunc(config.multilabel,
                                             config.multilabel_weights,
                                             config.imbalance_class_weights)
            elif config.class_loss_func == BUILTIN_CLASS_LOSS_FOCAL:
                assert config.num_classes == 1, \
                    "Focal loss only works with binary classification." \
                    "num_classes should be set to 1."
                # set default value of alpha to 0.25 for focal loss
                # set default value of gamma to 2. for focal loss
                alpha = config.alpha if config.alpha is not None else 0.25
                gamma = config.gamma if config.gamma is not None else 2.
                loss_func = FocalLossFunc(alpha, gamma)
            else:
                raise RuntimeError(
                    f"Unknown classification loss {config.class_loss_func}")
        else:
            decoder = {}
            loss_func = {}
            for ntype in config.target_ntype:
                decoder[ntype] = EntityClassifier(decoder_input_dim,
                                                  config.num_classes[ntype],
                                                  config.multilabel[ntype],
                                                  dropout=dropout,
                                                  norm=config.decoder_norm,
                                                  use_bias=config.decoder_bias)

                if config.class_loss_func == BUILTIN_CLASS_LOSS_CROSS_ENTROPY:
                    loss_func[ntype] = ClassifyLossFunc(config.multilabel[ntype],
                                                        config.multilabel_weights[ntype],
                                                        config.imbalance_class_weights[ntype])
                elif config.class_loss_func == BUILTIN_CLASS_LOSS_FOCAL:
                    # set default value of alpha to 0.25 for focal loss
                    # set default value of gamma to 2. for focal loss
                    alpha = config.alpha if config.alpha is not None else 0.25
                    gamma = config.gamma if config.gamma is not None else 2.
                    loss_func[ntype] =  FocalLossFunc(alpha, gamma)
                else:
                    raise RuntimeError(
                        f"Unknown classification loss {config.class_loss_func}")
    elif config.task_type == BUILTIN_TASK_NODE_REGRESSION:
        decoder  = EntityRegression(decoder_input_dim,
                                    dropout=dropout,
                                    norm=config.decoder_norm,
                                    use_bias=config.decoder_bias)
        loss_func = RegressionLossFunc()
    else:
        raise ValueError('unknown node task: {}'.format(config.task_type))

    return decoder, loss_func


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

    encoder_out_dims = model.gnn_encoder.out_dims \
        if model.gnn_encoder is not None \
            else model.node_input_encoder.out_dims
    decoder, loss_func = create_builtin_node_decoder(g, encoder_out_dims, config, train_task)
    model.set_decoder(decoder)
    model.set_loss_func(loss_func)

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

def create_builtin_edge_decoder(g, decoder_input_dim, config, train_task):
    """ create builtin edge decoder according to task config

    Parameters
    ----------
    g: DGLGraph
        The graph data.
    decoder_input_dim: int
        Input dimension size of the decoder.
    config: GSConfig
        Configurations.
    train_task : bool
        Whether this model is used for training.

    Returns
    -------
    decoder: The node task decoder(s)
    loss_func: The loss function(s)
    """
    dropout = config.dropout if train_task else 0
    if config.task_type == BUILTIN_TASK_EDGE_CLASSIFICATION:
        num_classes = config.num_classes
        decoder_type = config.decoder_type
        # TODO(zhengda) we should support multiple target etypes
        target_etype = config.target_etype[0]
        if decoder_type == "DenseBiDecoder":
            num_decoder_basis = config.num_decoder_basis
            assert config.num_ffn_layers_in_decoder == 0, \
                "DenseBiDecoder does not support adding extra feedforward neural network layers" \
                "You can increases num_basis to increase the parameter size."
            decoder = DenseBiDecoder(in_units=decoder_input_dim,
                                     num_classes=num_classes,
                                     multilabel=config.multilabel,
                                     num_basis=num_decoder_basis,
                                     dropout_rate=dropout,
                                     regression=False,
                                     target_etype=target_etype,
                                     norm=config.decoder_norm,
                                     use_bias=config.decoder_bias)
        elif decoder_type == "MLPDecoder":
            decoder = MLPEdgeDecoder(decoder_input_dim,
                                     num_classes,
                                     multilabel=config.multilabel,
                                     target_etype=target_etype,
                                     num_ffn_layers=config.num_ffn_layers_in_decoder,
                                     norm=config.decoder_norm,
                                     use_bias=config.decoder_bias)
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
                feat_dim = sum(
                    g.edges[target_etype].data[fname].shape[-1] \
                    for fname in decoder_edge_feat[target_etype]
                )

            decoder = MLPEFeatEdgeDecoder(
                h_dim=decoder_input_dim,
                feat_dim=feat_dim,
                out_dim=num_classes,
                multilabel=config.multilabel,
                target_etype=target_etype,
                dropout=config.dropout,
                num_ffn_layers=config.num_ffn_layers_in_decoder,
                norm=config.decoder_norm,
                use_bias=config.decoder_bias)
        else:
            assert False, f"decoder {decoder_type} is not supported."

        if config.class_loss_func == BUILTIN_CLASS_LOSS_CROSS_ENTROPY:
            loss_func = ClassifyLossFunc(config.multilabel,
                                         config.multilabel_weights,
                                         config.imbalance_class_weights)
        elif config.class_loss_func == BUILTIN_CLASS_LOSS_FOCAL:
            # set default value of alpha to 0.25 for focal loss
            # set default value of gamma to 2. for focal loss
            alpha = config.alpha if config.alpha is not None else 0.25
            gamma = config.gamma if config.gamma is not None else 2.
            loss_func = FocalLossFunc(alpha, gamma)
        else:
            raise RuntimeError(
                f"Unknown classification loss {config.class_loss_func}")

    elif config.task_type == BUILTIN_TASK_EDGE_REGRESSION:
        decoder_type = config.decoder_type
        dropout = config.dropout if train_task else 0
        # TODO(zhengda) we should support multiple target etypes
        target_etype = config.target_etype[0]
        if decoder_type == "DenseBiDecoder":
            num_decoder_basis = config.num_decoder_basis
            decoder = DenseBiDecoder(decoder_input_dim,
                                     1,
                                     num_basis=num_decoder_basis,
                                     multilabel=False,
                                     target_etype=target_etype,
                                     dropout_rate=dropout,
                                     regression=True,
                                     norm=config.decoder_norm,
                                     use_bias=config.decoder_bias)
        elif decoder_type == "MLPDecoder":
            decoder = MLPEdgeDecoder(decoder_input_dim,
                                     1,
                                     multilabel=False,
                                     target_etype=target_etype,
                                     regression=True,
                                     num_ffn_layers=config.num_ffn_layers_in_decoder,
                                     norm=config.decoder_norm,
                                     use_bias=config.decoder_bias)
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
                feat_dim = sum(
                    g.edges[target_etype].data[fname].shape[-1] \
                    for fname in decoder_edge_feat[target_etype]
                )

            decoder = MLPEFeatEdgeDecoder(
                h_dim=decoder_input_dim,
                feat_dim=feat_dim,
                out_dim=1,
                multilabel=False,
                target_etype=target_etype,
                dropout=config.dropout,
                regression=True,
                num_ffn_layers=config.num_ffn_layers_in_decoder,
                norm=config.decoder_norm,
                use_bias=config.decoder_bias)
        else:
            assert False, "decoder not supported"
        loss_func = RegressionLossFunc()
    else:
        raise ValueError('unknown node task: {}'.format(config.task_type))
    return decoder, loss_func

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
    encoder_out_dims = model.gnn_encoder.out_dims \
        if model.gnn_encoder is not None \
            else model.node_input_encoder.out_dims
    decoder, loss_func = create_builtin_edge_decoder(g, encoder_out_dims, config, train_task)
    model.set_decoder(decoder)
    model.set_loss_func(loss_func)

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

# pylint: disable=unused-argument
def create_builtin_lp_decoder(g, decoder_input_dim, config, train_task):
    """ create builtin link prediction decoder according to task config

    Parameters
    ----------
    g: DGLGraph
        The graph data.
    decoder_input_dim: int
        Input dimension size of the decoder.
    config: GSConfig
        Configurations.
    train_task : bool
        Whether this model is used for training.

    Returns
    -------
    decoder: The node task decoder(s)
    loss_func: The loss function(s)
    """
    if config.decoder_norm is not None:
        # TODO: Support decoder norm for lp decoders when required.
        logging.warning("Decoder norm (batch norm or layer norm) is not supported"
                        "for link prediction decoders.")
    if config.lp_decoder_type == BUILTIN_LP_DOT_DECODER:
        # if the training set only contains one edge type or it is specified in the arguments,
        # we use dot product as the score function.
        if get_rank() == 0:
            logging.debug('use dot product for single-etype task.')
            logging.debug("Using inner product objective for supervision")
        if config.lp_edge_weight_for_loss is None:
            decoder = LinkPredictContrastiveDotDecoder(decoder_input_dim) \
                if config.lp_loss_func == BUILTIN_LP_LOSS_CONTRASTIVELOSS else \
                LinkPredictDotDecoder(decoder_input_dim)
        else:
            decoder = LinkPredictWeightedDotDecoder(decoder_input_dim,
                                                    config.lp_edge_weight_for_loss)
    elif config.lp_decoder_type == BUILTIN_LP_DISTMULT_DECODER:
        if get_rank() == 0:
            logging.debug("Using distmult objective for supervision")

        # default gamma for distmult is 12.
        gamma = config.gamma if config.gamma is not None else 12.
        if config.lp_edge_weight_for_loss is None:
            decoder = LinkPredictContrastiveDistMultDecoder(g.canonical_etypes,
                                                            decoder_input_dim,
                                                            gamma) \
                if config.lp_loss_func == BUILTIN_LP_LOSS_CONTRASTIVELOSS else \
                LinkPredictDistMultDecoder(g.canonical_etypes,
                                           decoder_input_dim,
                                           gamma)
        else:
            decoder = LinkPredictWeightedDistMultDecoder(g.canonical_etypes,
                                                         decoder_input_dim,
                                                         gamma,
                                                         config.lp_edge_weight_for_loss)
    elif config.lp_decoder_type == BUILTIN_LP_ROTATE_DECODER:
        if get_rank() == 0:
            logging.debug("Using RotatE objective for supervision")

        # default gamma for RotatE is 12.
        gamma = config.gamma if config.gamma is not None else 12.
        if config.lp_edge_weight_for_loss is None:
            decoder = LinkPredictContrastiveRotatEDecoder(g.canonical_etypes,
                                                          decoder_input_dim,
                                                          gamma) \
                if config.lp_loss_func == BUILTIN_LP_LOSS_CONTRASTIVELOSS else \
                LinkPredictRotatEDecoder(g.canonical_etypes,
                                         decoder_input_dim,
                                         gamma)
        else:
            decoder = LinkPredictWeightedRotatEDecoder(g.canonical_etypes,
                                                       decoder_input_dim,
                                                       gamma,
                                                       config.lp_edge_weight_for_loss)
    elif config.lp_decoder_type in [BUILTIN_LP_TRANSE_L1_DECODER, BUILTIN_LP_TRANSE_L2_DECODER]:
        if get_rank() == 0:
            logging.debug("Using TransE objective for supervision")

        # default gamma for TransE is 12.
        gamma = config.gamma if config.gamma is not None else 12.

        score_norm = 'l1' if config.lp_decoder_type == BUILTIN_LP_TRANSE_L1_DECODER else 'l2'
        if config.lp_edge_weight_for_loss is None:
            decoder = LinkPredictContrastiveTransEDecoder(g.canonical_etypes,
                                                          decoder_input_dim,
                                                          gamma,
                                                          score_norm) \
                if config.lp_loss_func == BUILTIN_LP_LOSS_CONTRASTIVELOSS else \
                LinkPredictTransEDecoder(g.canonical_etypes,
                                         decoder_input_dim,
                                         gamma,
                                         score_norm)
        else:
            decoder = LinkPredictWeightedTransEDecoder(g.canonical_etypes,
                                                       decoder_input_dim,
                                                       gamma,
                                                       score_norm,
                                                       config.lp_edge_weight_for_loss)
    else:
        raise RuntimeError(
            f"Unknown link prediction decoder type {config.lp_decoder_type}")

    if config.lp_loss_func == BUILTIN_LP_LOSS_CONTRASTIVELOSS:
        loss_func = LinkPredictContrastiveLossFunc(config.contrastive_loss_temperature)
    elif config.lp_loss_func == BUILTIN_LP_LOSS_CROSS_ENTROPY:
        if config.lp_edge_weight_for_loss is None:
            if config.adversarial_temperature is None:
                loss_func = LinkPredictBCELossFunc()
            else:
                loss_func = LinkPredictAdvBCELossFunc(config.adversarial_temperature)
        else:
            if config.adversarial_temperature is None:
                loss_func = WeightedLinkPredictBCELossFunc()
            else:
                loss_func = WeightedLinkPredictAdvBCELossFunc(config.adversarial_temperature)
    else:
        raise TypeError(f"Unknown link prediction loss function {config.lp_loss_func}")

    return decoder, loss_func

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
    decoder, loss_func = create_builtin_lp_decoder(g, out_dims, config, train_task)

    model.set_decoder(decoder)
    model.set_loss_func(loss_func)

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
    node_feat_size = get_node_feat_size(g, config.node_feat_name)
    reconstruct_feats = len(config.construct_feat_ntype) > 0
    model_encoder_type = config.model_encoder_type
    if config.node_lm_configs is not None:
        emb_path = os.path.join(os.path.dirname(config.part_config),
                "cached_embs") if config.cache_lm_embed else None
        if model_encoder_type == "lm":
            # only use language model(s) as input layer encoder(s)
            node_encoder = GSPureLMNodeInputLayer(g, config.node_lm_configs,
                                                  num_train=config.lm_train_nodes,
                                                  lm_infer_batch_size=config.lm_infer_batch_size,
                                                  cached_embed_path=emb_path,
                                                  wg_cached_embed=config.use_wholegraph_embed)
        else:
            node_encoder = GSLMNodeEncoderInputLayer(g, config.node_lm_configs,
                                                    node_feat_size, config.hidden_size,
                                                    num_train=config.lm_train_nodes,
                                                    lm_infer_batch_size=config.lm_infer_batch_size,
                                                    dropout=config.dropout,
                                                    use_node_embeddings=config.use_node_embeddings,
                                                    cached_embed_path=emb_path,
                                                    wg_cached_embed=config.use_wholegraph_embed,
                                                    force_no_embeddings=config.construct_feat_ntype
                                                    )
    else:
        node_encoder = GSNodeEncoderInputLayer(g, node_feat_size, config.hidden_size,
                                            dropout=config.dropout,
                                            activation=config.input_activate,
                                            use_node_embeddings=config.use_node_embeddings,
                                            force_no_embeddings=config.construct_feat_ntype,
                                            num_ffn_layers_in_input=config.num_ffn_layers_in_input,
                                            use_wholegraph_sparse_emb=config.use_wholegraph_embed)
        # set edge encoder input layer no matter if having edge feature names or not
        # TODO: add support of languange models and GLEM
        edge_feat_size = get_edge_feat_size(g, config.edge_feat_name)
        edge_encoder = GSEdgeEncoderInputLayer(g, edge_feat_size, config.hidden_size,
                                        dropout=config.dropout,
                                        activation=config.input_activate,
                                        num_ffn_layers_in_input=config.num_ffn_layers_in_input)
        model.set_edge_input_encoder(edge_encoder)

    # The number of feature dimensions can change. For example, the feature dimensions
    # of BERT embeddings are determined when the input encoder is created.
    node_feat_size = node_encoder.in_dims
    model.set_node_input_encoder(node_encoder)

    # Set GNN encoders
    dropout = config.dropout if train_task else 0
    out_emb_size = config.out_emb_size if config.out_emb_size else config.hidden_size

    if model_encoder_type in ("mlp", "lm"):
        # Only input encoder is used
        assert config.num_layers == 0, "No GNN layers"
        gnn_encoder = None
    elif model_encoder_type == "rgcn":
        num_bases = config.num_bases
        # we need to set the num_layers -1 because there is an output layer
        # that is hard coded.
        gnn_encoder = RelationalGCNEncoder(g,
                                           config.hidden_size, out_emb_size,
                                           num_bases=num_bases,
                                           num_hidden_layers=config.num_layers -1,
                                           edge_feat_name=config.edge_feat_name,
                                           edge_feat_mp_op=config.edge_feat_mp_op,
                                           dropout=dropout,
                                           use_self_loop=config.use_self_loop,
                                           num_ffn_layers_in_gnn=config.num_ffn_layers_in_gnn,
                                           norm=config.gnn_norm)
    elif model_encoder_type == "rgat":
        # we need to set the num_layers -1 because there is an output layer that is hard coded.
        gnn_encoder = RelationalGATEncoder(g,
                                           config.hidden_size,
                                           out_emb_size,
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
                                 out_emb_size,
                                 num_hidden_layers=config.num_layers -1,
                                 num_heads=config.num_heads,
                                 dropout=dropout,
                                 norm=config.gnn_norm,
                                 num_ffn_layers_in_gnn=config.num_ffn_layers_in_gnn)
    elif model_encoder_type == "sage":
        # we need to check if the graph is homogeneous
        assert check_homo(g), \
            'The graph is not a homogeneous graph, can not use sage model encoder'
        # we need to set the num_layers -1 because there is an output layer that is hard coded.
        gnn_encoder = SAGEEncoder(h_dim=config.hidden_size,
                                  out_dim=out_emb_size,
                                  num_hidden_layers=config.num_layers - 1,
                                  dropout=dropout,
                                  aggregator_type='pool',
                                  num_ffn_layers_in_gnn=config.num_ffn_layers_in_gnn,
                                  norm=config.gnn_norm)
    elif model_encoder_type == "gat":
        # we need to check if the graph is homogeneous
        assert check_homo(g), \
            'The graph is not a homogeneous graph, can not use gat model encoder'
        gnn_encoder = GATEncoder(h_dim=config.hidden_size,
                                 out_dim=out_emb_size,
                                 num_heads=config.num_heads,
                                 num_hidden_layers=config.num_layers -1,
                                 dropout=dropout,
                                 num_ffn_layers_in_gnn=config.num_ffn_layers_in_gnn)
    elif model_encoder_type == "gatv2":
        # we need to check if the graph is homogeneous
        assert check_homo(g), \
            'The graph is not a homogeneous graph, can not use gatv2 model encoder'
        gnn_encoder = GATv2Encoder(h_dim=config.hidden_size,
                                   out_dim=out_emb_size,
                                   num_heads=config.num_heads,
                                   num_hidden_layers=config.num_layers -1,
                                   dropout=dropout,
                                   num_ffn_layers_in_gnn=config.num_ffn_layers_in_gnn)
    else:
        assert False, "Unknown gnn model type {}".format(model_encoder_type)

    # Check use edge feature and GNN encoder capacity
    assert (config.edge_feat_name is None) or \
                (config.edge_feat_name is not None and gnn_encoder.is_support_edge_feat()), \
                    f'The \"{gnn_encoder.__class__.__name__}\" encoder dose not support ' + \
                     'edge feature in this version. Please check GraphStorm documentations ' + \
                     'to find gnn encoders that support edge features, e.g., \"rgcn\".'

    if reconstruct_feats:
        rel_names = get_rel_names_for_reconstruct(g, config.construct_feat_ntype, node_feat_size)
        dst_types = {rel_name[2] for rel_name in rel_names}
        for ntype in config.construct_feat_ntype:
            assert ntype in dst_types, \
                    f"We cannot reconstruct features of node {ntype} " \
                    + "probably because their neighbors don't have features."
        assert config.construct_feat_encoder == "rgcn", \
                "We only support RGCN for reconstructing node features."
        input_gnn = RelGraphConvLayer(config.hidden_size, out_emb_size,
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
    """ Create a builtin task tracker

    Parameters
    ----------
    config: GSConfig
        Configurations
    """
    tracker_class = get_task_tracker_class(config.task_tracker)
    return tracker_class(config.eval_frequency)

def get_builtin_lp_eval_dataloader_class(config):
    """ Return a builtin link prediction evaluation dataloader
        based on input config

    Parameters
    ----------
    config: GSConfig
        Configurations
    """
    test_dataloader_cls = None
    if config.eval_etypes_negative_dstnode is not None:
        test_dataloader_cls = GSgnnLinkPredictionPredefinedTestDataLoader
    elif config.eval_negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
        test_dataloader_cls = GSgnnLinkPredictionTestDataLoader
    elif config.eval_negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER:
        test_dataloader_cls = GSgnnLinkPredictionJointTestDataLoader
    else:
        raise ValueError('Unknown test negative sampler.'
            'Supported test negative samplers include '
            f'[{BUILTIN_LP_UNIFORM_NEG_SAMPLER}, {BUILTIN_LP_JOINT_NEG_SAMPLER}]')
    return test_dataloader_cls

def get_builtin_lp_train_dataloader_class(config):
    """ Return a builtin link prediction training dataloader
        based on input config

    Parameters
    ----------
    config: GSConfig
        Configurations
    """
    dataloader_cls = None
    if config.train_negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
        dataloader_cls = GSgnnLinkPredictionDataLoader
    elif config.train_negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER:
        dataloader_cls = GSgnnLPJointNegDataLoader
    elif config.train_negative_sampler == BUILTIN_LP_INBATCH_JOINT_NEG_SAMPLER:
        dataloader_cls = GSgnnLPInBatchJointNegDataLoader
    elif config.train_negative_sampler == BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER:
        dataloader_cls = GSgnnLPLocalUniformNegDataLoader
    elif config.train_negative_sampler == BUILTIN_LP_LOCALJOINT_NEG_SAMPLER:
        dataloader_cls = GSgnnLPLocalJointNegDataLoader
    elif config.train_negative_sampler == BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER:
        dataloader_cls = GSgnnAllEtypeLinkPredictionDataLoader
    elif config.train_negative_sampler == BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER:
        dataloader_cls = GSgnnAllEtypeLPJointNegDataLoader
    elif config.train_negative_sampler == BUILTIN_FAST_LP_UNIFORM_NEG_SAMPLER:
        dataloader_cls = FastGSgnnLinkPredictionDataLoader
    elif config.train_negative_sampler == BUILTIN_FAST_LP_JOINT_NEG_SAMPLER:
        dataloader_cls = FastGSgnnLPJointNegDataLoader
    elif config.train_negative_sampler == BUILTIN_FAST_LP_LOCALUNIFORM_NEG_SAMPLER:
        dataloader_cls = FastGSgnnLPLocalUniformNegDataLoader
    elif config.train_negative_sampler == BUILTIN_FAST_LP_LOCALJOINT_NEG_SAMPLER:
        dataloader_cls = FastGSgnnLPLocalJointNegDataLoader
    else:
        raise ValueError('Unknown negative sampler')

    return dataloader_cls

def create_task_decoder(task_info, g, decoder_input_dim, train_task):
    """ Create a task decoder according to task_info.

    Parameters
    ----------
    task_info: TaskInfo
        Task info.
    g: Dist DGLGraph
        Graph
    decoder_input_dim: int
        The dimension of the input embedding of the decoder
    train_task: bool
        Whether the task is a training task

    Return
    ------
    decoder: The task decoder
    loss_func: The loss function
    """
    if task_info.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        return create_builtin_node_decoder(g, decoder_input_dim, task_info.task_config, train_task)
    elif task_info.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
        return create_builtin_edge_decoder(g, decoder_input_dim, task_info.task_config, train_task)
    elif task_info.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
        return create_builtin_lp_decoder(g, decoder_input_dim, task_info.task_config, train_task)
    elif task_info.task_type in [BUILTIN_TASK_RECONSTRUCT_NODE_FEAT]:
        return create_builtin_reconstruct_nfeat_decoder(
            g, decoder_input_dim, task_info.task_config, train_task)
    elif task_info.task_type in [BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT]:
        return create_builtin_reconstruct_efeat_decoder(
            g, decoder_input_dim, task_info.task_config, train_task)
    else:
        raise TypeError(f"Unknown task type {task_info.task_type}")

def create_evaluator(task_info):
    """ Create task specific evaluator according to task_info for multi-task learning.

    Parameters
    ----------
    task_info: TaskInfo
        Task info.

    Return
    ------
    Evaluator
    """
    config = task_info.task_config
    if task_info.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION]:
        assert isinstance(config.multilabel, bool), \
            "In multi-task learning, we define one task at one time." \
            "But here, the task config is expecting to define multiple " \
            "tasks as config.multilabel is not a single boolean " \
            f'but {config.multilabel}.'
        return GSgnnClassificationEvaluator(config.eval_frequency,
                                            config.eval_metric,
                                            config.multilabel,
                                            config.use_early_stop,
                                            config.early_stop_burnin_rounds,
                                            config.early_stop_rounds,
                                            config.early_stop_strategy)
    elif task_info.task_type in [BUILTIN_TASK_NODE_REGRESSION]:
        return GSgnnRegressionEvaluator(config.eval_frequency,
                                        config.eval_metric,
                                        config.use_early_stop,
                                        config.early_stop_burnin_rounds,
                                        config.early_stop_rounds,
                                        config.early_stop_strategy)
    elif task_info.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION]:
        return GSgnnClassificationEvaluator(config.eval_frequency,
                                            config.eval_metric,
                                            config.multilabel,
                                            config.use_early_stop,
                                            config.early_stop_burnin_rounds,
                                            config.early_stop_rounds,
                                            config.early_stop_strategy)
    elif task_info.task_type in [BUILTIN_TASK_EDGE_REGRESSION]:
        return GSgnnRegressionEvaluator(config.eval_frequency,
                                        config.eval_metric,
                                        config.use_early_stop,
                                        config.early_stop_burnin_rounds,
                                        config.early_stop_rounds,
                                        config.early_stop_strategy)
    elif task_info.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
        if config.report_eval_per_type:
            return GSgnnPerEtypeLPEvaluator(eval_frequency=config.eval_frequency,
                                    eval_metric_list=config.eval_metric,
                                    major_etype=config.model_select_etype,
                                    use_early_stop=config.use_early_stop,
                                    early_stop_burnin_rounds=config.early_stop_burnin_rounds,
                                    early_stop_rounds=config.early_stop_rounds,
                                    early_stop_strategy=config.early_stop_strategy)
        else:
            return GSgnnLPEvaluator(eval_frequency=config.eval_frequency,
                                    eval_metric_list=config.eval_metric,
                                    use_early_stop=config.use_early_stop,
                                    early_stop_burnin_rounds=config.early_stop_burnin_rounds,
                                    early_stop_rounds=config.early_stop_rounds,
                                    early_stop_strategy=config.early_stop_strategy)
    elif task_info.task_type in [BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                                 BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT]:
        return GSgnnRconstructFeatRegScoreEvaluator(
            config.eval_frequency,
            config.eval_metric,
            config.use_early_stop,
            config.early_stop_burnin_rounds,
            config.early_stop_rounds,
            config.early_stop_strategy)
    return None

def create_lp_evaluator(config):
    """ Create LP specific evaluator.

        Parameters
        ----------
        config: GSConfig
            Configuration.

        Return
        ------
        Evaluator: A link prediction evaluator
    """
    assert all(
        (x.startswith(SUPPORTED_HIT_AT_METRICS) or x in SUPPORTED_LINK_PREDICTION_METRICS)
            for x in config.eval_metric), (
        "Invalid LP evaluation metrics. "
        f"GraphStorm only supports {SUPPORTED_LINK_PREDICTION_METRICS} as metrics "
        f"for link prediction, got {config.eval_metric}")

    if config.report_eval_per_type:
        return GSgnnPerEtypeLPEvaluator(eval_frequency=config.eval_frequency,
                                eval_metric_list=config.eval_metric,
                                major_etype=config.model_select_etype,
                                use_early_stop=config.use_early_stop,
                                early_stop_burnin_rounds=config.early_stop_burnin_rounds,
                                early_stop_rounds=config.early_stop_rounds,
                                early_stop_strategy=config.early_stop_strategy)
    else:
        return GSgnnLPEvaluator(eval_frequency=config.eval_frequency,
                                eval_metric_list=config.eval_metric,
                                use_early_stop=config.use_early_stop,
                                early_stop_burnin_rounds=config.early_stop_burnin_rounds,
                                early_stop_rounds=config.early_stop_rounds,
                                early_stop_strategy=config.early_stop_strategy)
