"""
    Copyright 2024 Contributors
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Unit tests for gsf.py
"""

import os
import json
import yaml
import pytest
import tempfile
import copy
from argparse import Namespace
from pathlib import Path
from transformers import AutoModel
import numpy as np
import torch as th
from torch.nn.parallel import DistributedDataParallel as DDP
from graphstorm.config import GSConfig
from graphstorm.gsf import (get_edge_feat_size,
                            create_builtin_node_decoder,
                            create_builtin_edge_decoder,
                            create_builtin_lp_decoder,
                            create_builtin_node_gnn_model,
                            create_builtin_edge_gnn_model,
                            create_builtin_lp_gnn_model,
                            create_builtin_node_model,
                            restore_builtin_model_from_artifacts)
from graphstorm.utils import check_graph_name
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                               BUILTIN_TASK_NODE_REGRESSION,
                               BUILTIN_TASK_EDGE_CLASSIFICATION,
                               BUILTIN_TASK_EDGE_REGRESSION,
                               BUILTIN_TASK_LINK_PREDICTION,
                               BUILTIN_LP_DOT_DECODER,
                               BUILTIN_LP_DISTMULT_DECODER,
                               BUILTIN_LP_ROTATE_DECODER,
                               BUILTIN_LP_TRANSE_L1_DECODER,
                               BUILTIN_CLASS_LOSS_CROSS_ENTROPY,
                               BUILTIN_CLASS_LOSS_FOCAL,
                               BUILTIN_LP_LOSS_CROSS_ENTROPY,
                               BUILTIN_LP_LOSS_CONTRASTIVELOSS,
                               BUILTIN_LP_LOSS_BPR,
                               BUILTIN_REGRESSION_LOSS_MSE,
                               BUILTIN_REGRESSION_LOSS_SHRINKAGE)
from graphstorm.model.rgcn_encoder import RelationalGCNEncoder
from graphstorm.model.node_decoder import (EntityClassifier,
                                           EntityRegression)
from graphstorm.model.edge_decoder import (DenseBiDecoder,
                                           MLPEdgeDecoder,
                                           LinkPredictDotDecoder,
                                           LinkPredictDistMultDecoder,
                                           LinkPredictContrastiveDotDecoder,
                                           LinkPredictContrastiveDistMultDecoder,
                                           LinkPredictWeightedDotDecoder,
                                           LinkPredictWeightedDistMultDecoder,
                                           LinkPredictRotatEDecoder,
                                           LinkPredictContrastiveRotatEDecoder,
                                           LinkPredictWeightedRotatEDecoder,LinkPredictTransEDecoder,
                                           LinkPredictWeightedTransEDecoder,
                                           LinkPredictContrastiveTransEDecoder)

from graphstorm.model.loss_func import (ClassifyLossFunc,
                                        RegressionLossFunc,
                                        LinkPredictBCELossFunc,
                                        WeightedLinkPredictBCELossFunc,
                                        LinkPredictContrastiveLossFunc,
                                        LinkPredictAdvBCELossFunc,
                                        WeightedLinkPredictAdvBCELossFunc,
                                        LinkPredictBPRLossFunc,
                                        WeightedLinkPredictBPRLossFunc,
                                        FocalLossFunc,
                                        ShrinkageLossFunc)

from data_utils import (generate_dummy_dist_graph,
                        generate_dummy_hetero_graph,
                        generate_dummy_dist_graph_hete_rev_edge)
from config_utils import create_dummy_config_obj


class GSTestConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def test_create_builtin_node_decoder():
    g = None
    decoder_input_dim = 64
    train_task=False

    # node classification + cross entropy loss
    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_NODE_CLASSIFICATION,
            "num_classes": 2,
            "multilabel": False,
            "class_loss_func": BUILTIN_CLASS_LOSS_CROSS_ENTROPY,
            "multilabel_weights": None,
            "decoder_norm": None,
            "imbalance_class_weights": None,
            "decoder_bias": False,
        }
    )

    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, EntityClassifier)
    assert isinstance(loss_func, ClassifyLossFunc)

    # node classification + focal loss with default params
    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_NODE_CLASSIFICATION,
            "num_classes": 2,
            "multilabel": False,
            "class_loss_func": BUILTIN_CLASS_LOSS_FOCAL,
            "multilabel_weights": None,
            "decoder_norm": None,
            "imbalance_class_weights": None,
            "alpha": None,
            "gamma": None,
            "decoder_bias": False
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, EntityClassifier)
    assert isinstance(loss_func, FocalLossFunc)
    assert loss_func.alpha == 0.25
    assert loss_func.gamma == 2.

    # node classification + focal loss with num class = 1
    # Note: make sure the current code is backward compatible
    # May remove this test in the future
    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_NODE_CLASSIFICATION,
            "num_classes": 1,
            "multilabel": False,
            "class_loss_func": BUILTIN_CLASS_LOSS_FOCAL,
            "multilabel_weights": None,
            "decoder_norm": None,
            "imbalance_class_weights": None,
            "alpha": None,
            "gamma": None,
            "decoder_bias": False
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, EntityClassifier)
    assert isinstance(loss_func, FocalLossFunc)
    assert loss_func.alpha == 0.25
    assert loss_func.gamma == 2.

    # node classification + cross entropy with num class = 1
    # Will cause failure
    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_NODE_CLASSIFICATION,
            "num_classes": 1,
            "multilabel": False,
            "class_loss_func": BUILTIN_CLASS_LOSS_CROSS_ENTROPY,
            "multilabel_weights": None,
            "decoder_norm": None,
            "imbalance_class_weights": None,
            "alpha": None,
            "gamma": None,
            "decoder_bias": False
        }
    )
    with pytest.raises(AssertionError):
        decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)

    # node classification + cross entropy loss for multiple node types
    config = GSTestConfig(
        {
            "target_ntype": ["n0", "n1"],
            "task_type": BUILTIN_TASK_NODE_CLASSIFICATION,
            "class_loss_func": BUILTIN_CLASS_LOSS_CROSS_ENTROPY,
            "num_classes": {
                "n0": 2,
                "n1": 4
            },
            "multilabel":  {
                "n0": True,
                "n1": True
            },
            "multilabel_weights": {
                "n0": None,
                "n1": None
            },

            "imbalance_class_weights": {
                "n0": None,
                "n1": None
            },
            "decoder_norm": None,
            "decoder_bias": False,
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, dict)
    assert isinstance(decoder["n0"], EntityClassifier)
    assert isinstance(decoder["n1"], EntityClassifier)
    assert isinstance(loss_func, dict)
    assert isinstance(loss_func["n0"], ClassifyLossFunc)
    assert isinstance(loss_func["n1"], ClassifyLossFunc)

    # node classification + focal loss with default params for multiple node types
    config = GSTestConfig(
        {
            "target_ntype": ["n0", "n1"],
            "task_type": BUILTIN_TASK_NODE_CLASSIFICATION,
            "num_classes": {
                "n0": 2,
                "n1": 2
            },
            "multilabel":  {
                "n0": False,
                "n1": False
            },
            "multilabel_weights": {
                "n0": None,
                "n1": None
            },
            "class_loss_func": BUILTIN_CLASS_LOSS_FOCAL,
            "decoder_norm": None,
            "alpha": 0.3,
            "gamma": 3.,
            "decoder_bias": False,
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, dict)
    assert isinstance(decoder["n0"], EntityClassifier)
    assert isinstance(decoder["n1"], EntityClassifier)
    assert isinstance(loss_func, dict)
    assert isinstance(loss_func["n0"], FocalLossFunc)
    assert isinstance(loss_func["n1"], FocalLossFunc)
    assert loss_func["n0"].alpha == 0.3
    assert loss_func["n0"].gamma == 3.

    # node classification + focal loss with num class = 1
    # for multiple node types
    # Note: make sure the current code is backward compatible
    # May remove this test in the future
    config = GSTestConfig(
        {
            "target_ntype": ["n0", "n1"],
            "task_type": BUILTIN_TASK_NODE_CLASSIFICATION,
            "num_classes": {
                "n0": 1,
                "n1": 2
            },
            "multilabel":  {
                "n0": False,
                "n1": False
            },
            "multilabel_weights": {
                "n0": None,
                "n1": None
            },
            "class_loss_func": BUILTIN_CLASS_LOSS_FOCAL,
            "decoder_norm": None,
            "alpha": 0.3,
            "gamma": 3.,
            "decoder_bias": False,
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, dict)
    assert isinstance(decoder["n0"], EntityClassifier)
    assert isinstance(decoder["n1"], EntityClassifier)
    assert isinstance(loss_func, dict)
    assert isinstance(loss_func["n0"], FocalLossFunc)
    assert isinstance(loss_func["n1"], FocalLossFunc)
    assert loss_func["n0"].alpha == 0.3
    assert loss_func["n0"].gamma == 3.

    # node classification + cross entropy with num class = 1
    # Will cause failure for multiple node types
    config = GSTestConfig(
        {
            "target_ntype": ["n0", "n1"],
            "task_type": BUILTIN_TASK_NODE_CLASSIFICATION,
            "class_loss_func": BUILTIN_CLASS_LOSS_CROSS_ENTROPY,
            "num_classes": {
                "n0": 1,
                "n1": 4
            },
            "multilabel":  {
                "n0": True,
                "n1": True
            },
            "multilabel_weights": {
                "n0": None,
                "n1": None
            },

            "imbalance_class_weights": {
                "n0": None,
                "n1": None
            },
            "decoder_norm": None,
            "decoder_bias": False,
        }
    )
    with pytest.raises(AssertionError):
        decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)

    # node regression
    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_NODE_REGRESSION,
            "decoder_norm": None,
            "decoder_bias": False,
            "regression_loss_func": BUILTIN_REGRESSION_LOSS_MSE
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, EntityRegression)
    assert isinstance(loss_func, RegressionLossFunc)

    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_NODE_REGRESSION,
            "decoder_norm": None,
            "decoder_bias": False,
            "regression_loss_func": BUILTIN_REGRESSION_LOSS_SHRINKAGE,
            "alpha": None,
            "gamma": None
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, EntityRegression)
    assert isinstance(loss_func, ShrinkageLossFunc)
    assert loss_func.alpha == 10
    assert loss_func.gamma == 0.2

def test_create_builtin_edge_decoder():
    g = None
    decoder_input_dim = 64
    train_task=False

    # edge classification + cross entropy loss + DenseBiDecoder
    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_EDGE_CLASSIFICATION,
            "target_etype": [("n0", "r0", "n1")],
            "num_classes": 2,
            "decoder_type": "DenseBiDecoder",
            "num_decoder_basis": 2,
            "multilabel": False,
            "class_loss_func": BUILTIN_CLASS_LOSS_CROSS_ENTROPY,
            "decoder_norm": None,
            "num_ffn_layers_in_decoder": 0,
            "multilabel_weights": None,
            "imbalance_class_weights": None,
            "decoder_bias": True,
        }
    )
    decoder, loss_func = create_builtin_edge_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, DenseBiDecoder)
    assert isinstance(loss_func, ClassifyLossFunc)

    # edge classification + focal loss with default param + DenseBiDecoder
    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_EDGE_CLASSIFICATION,
            "target_etype": [("n0", "r0", "n1")],
            "num_classes": 2,
            "decoder_type": "DenseBiDecoder",
            "num_decoder_basis": 2,
            "multilabel": False,
            "class_loss_func": BUILTIN_CLASS_LOSS_FOCAL,
            "decoder_norm": None,
            "num_ffn_layers_in_decoder": 0,
            "alpha": None,
            "gamma": None,
            "decoder_bias": True,
        }
    )
    decoder, loss_func = create_builtin_edge_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, DenseBiDecoder)
    assert isinstance(loss_func, FocalLossFunc)
    assert loss_func.alpha == 0.25
    assert loss_func.gamma == 2.

    # edge classification + focal loss + MLPDecoder
    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_EDGE_CLASSIFICATION,
            "target_etype": [("n0", "r0", "n1")],
            "num_classes": 2,
            "decoder_type": "MLPDecoder",
            "multilabel": False,
            "class_loss_func": BUILTIN_CLASS_LOSS_FOCAL,
            "decoder_norm": None,
            "num_ffn_layers_in_decoder": 0,
            "alpha": 0.3,
            "gamma": 3.,
            "decoder_bias": True,
        }
    )
    decoder, loss_func = create_builtin_edge_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, MLPEdgeDecoder)
    assert isinstance(loss_func, FocalLossFunc)
    assert loss_func.alpha == 0.3
    assert loss_func.gamma == 3.

    # edge regression + DenseBiDecoder
    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_EDGE_REGRESSION,
            "target_etype": [("n0", "r0", "n1")],
            "num_classes": 2,
            "decoder_type": "DenseBiDecoder",
            "num_decoder_basis": 2,
            "decoder_norm": None,
            "decoder_bias": False,
            "regression_loss_func": BUILTIN_REGRESSION_LOSS_MSE
        }
    )
    decoder, loss_func = create_builtin_edge_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, DenseBiDecoder)
    assert isinstance(loss_func, RegressionLossFunc)

     # edge regression + MLPDecoder
    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_EDGE_REGRESSION,
            "target_etype": [("n0", "r0", "n1")],
            "num_classes": 2,
            "decoder_type": "MLPDecoder",
            "num_ffn_layers_in_decoder": 0,
            "decoder_norm": None,
            "decoder_bias": False,
            "regression_loss_func": BUILTIN_REGRESSION_LOSS_SHRINKAGE,
            "alpha": 0.3,
            "gamma": 3.,
        }
    )
    decoder, loss_func = create_builtin_edge_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, MLPEdgeDecoder)
    assert isinstance(loss_func, ShrinkageLossFunc)
    assert loss_func.alpha == 0.3
    assert loss_func.gamma == 3.

def test_create_builtin_lp_decoder():
    g = generate_dummy_hetero_graph()
    decoder_input_dim = 64
    train_task=False

    # dot-product + cross entropy
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CROSS_ENTROPY,
            "lp_edge_weight_for_loss": None,
            "decoder_norm": None,
            "adversarial_temperature": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictDotDecoder)
    assert isinstance(loss_func, LinkPredictBCELossFunc)

    # dot-product + cross entropy + adversarial
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CROSS_ENTROPY,
            "lp_edge_weight_for_loss": None,
            "decoder_norm": None,
            "adversarial_temperature": 1.0,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictDotDecoder)
    assert isinstance(loss_func, LinkPredictAdvBCELossFunc)

    # dot-product + cross entropy + edge weight
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CROSS_ENTROPY,
            "decoder_norm": None,
            "lp_edge_weight_for_loss": "weight",
            "adversarial_temperature": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictWeightedDotDecoder)
    assert isinstance(loss_func, WeightedLinkPredictBCELossFunc)

    # dot-product + cross entropy + edge weight + adversarial
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CROSS_ENTROPY,
            "decoder_norm": None,
            "lp_edge_weight_for_loss": "weight",
            "adversarial_temperature": 1.0,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictWeightedDotDecoder)
    assert isinstance(loss_func, WeightedLinkPredictAdvBCELossFunc)

    # dot-product + bayesian personalized ranking loss
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_BPR,
            "lp_edge_weight_for_loss": None,
            "decoder_norm": "l2norm",
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictDotDecoder)
    assert isinstance(loss_func, LinkPredictBPRLossFunc)

    # dot-product + bayesian personalized ranking loss
    # + edge weight
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_BPR,
            "lp_edge_weight_for_loss": "weight",
            "decoder_norm": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictWeightedDotDecoder)
    assert isinstance(loss_func, WeightedLinkPredictBPRLossFunc)

    # dot-product + contrastive loss
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CONTRASTIVELOSS,
            "lp_edge_weight_for_loss": None,
            "decoder_norm": "l2norm",
            "contrastive_loss_temperature": 1.0,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictContrastiveDotDecoder)
    assert isinstance(loss_func, LinkPredictContrastiveLossFunc)

    # dist mult + cross entropy
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_DISTMULT_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CROSS_ENTROPY,
            "lp_edge_weight_for_loss": None,
            "decoder_norm": None,
            "gamma": None,
            "adversarial_temperature": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictDistMultDecoder)
    assert isinstance(loss_func, LinkPredictBCELossFunc)
    assert decoder.gamma == 12.

    # dist mult + cross entropy + edge weight
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_DISTMULT_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CROSS_ENTROPY,
            "decoder_norm": None,
            "lp_edge_weight_for_loss": "weight",
            "gamma": None,
            "adversarial_temperature": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictWeightedDistMultDecoder)
    assert isinstance(loss_func, WeightedLinkPredictBCELossFunc)
    assert decoder.gamma == 12.

    # dist mult + bayesian personalized ranking loss
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_DISTMULT_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_BPR,
            "lp_edge_weight_for_loss": None,
            "decoder_norm": None,
            "gamma": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictDistMultDecoder)
    assert isinstance(loss_func, LinkPredictBPRLossFunc)
    assert decoder.gamma == 12.

    # dist mult + bayesian personalized ranking loss + edge weight
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_DISTMULT_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_BPR,
            "lp_edge_weight_for_loss": "weight",
            "decoder_norm": None,
            "gamma": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictWeightedDistMultDecoder)
    assert isinstance(loss_func, WeightedLinkPredictBPRLossFunc)
    assert decoder.gamma == 12.

    # dist mult + contrastive loss
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_DISTMULT_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CONTRASTIVELOSS,
            "lp_edge_weight_for_loss": None,
            "decoder_norm": "l2norm",
            "contrastive_loss_temperature": 1.0,
            "gamma": 6.
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictContrastiveDistMultDecoder)
    assert isinstance(loss_func, LinkPredictContrastiveLossFunc)
    assert decoder.gamma == 6.

    # rotate + cross entropy
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_ROTATE_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CROSS_ENTROPY,
            "lp_edge_weight_for_loss": None,
            "decoder_norm": None,
            "gamma": None,
            "adversarial_temperature": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictRotatEDecoder)
    assert isinstance(loss_func, LinkPredictBCELossFunc)
    assert decoder.gamma == 12.

    # rotate + cross entropy + edge weight
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_ROTATE_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CROSS_ENTROPY,
            "decoder_norm": None,
            "lp_edge_weight_for_loss": "weight",
            "gamma": None,
            "adversarial_temperature": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictWeightedRotatEDecoder)
    assert isinstance(loss_func, WeightedLinkPredictBCELossFunc)
    assert decoder.gamma == 12.

    # rotate + bayesian personalized ranking loss
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_ROTATE_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_BPR,
            "lp_edge_weight_for_loss": None,
            "decoder_norm": None,
            "gamma": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictRotatEDecoder)
    assert isinstance(loss_func, LinkPredictBPRLossFunc)
    assert decoder.gamma == 12.

    # rotate + bayesian personalized ranking loss  + edge weight
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_ROTATE_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_BPR,
            "decoder_norm": None,
            "lp_edge_weight_for_loss": "weight",
            "gamma": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictWeightedRotatEDecoder)
    assert isinstance(loss_func, WeightedLinkPredictBPRLossFunc)
    assert decoder.gamma == 12.

    # rotate + contrastive loss
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_ROTATE_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CONTRASTIVELOSS,
            "lp_edge_weight_for_loss": None,
            "decoder_norm": "l2norm",
            "contrastive_loss_temperature": 1.0,
            "gamma": 6.
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictContrastiveRotatEDecoder)
    assert isinstance(loss_func, LinkPredictContrastiveLossFunc)
    assert decoder.gamma == 6.

    # transe + cross entropy
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_TRANSE_L1_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CROSS_ENTROPY,
            "lp_edge_weight_for_loss": None,
            "adversarial_temperature": None,
            "decoder_norm": None,
            "gamma": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictTransEDecoder)
    assert isinstance(loss_func, LinkPredictBCELossFunc)
    assert decoder.gamma == 12.

    # transe + cross entropy + edge weight
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_TRANSE_L1_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CROSS_ENTROPY,
            "decoder_norm": None,
            "adversarial_temperature": None,
            "lp_edge_weight_for_loss": "weight",
            "gamma": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictWeightedTransEDecoder)
    assert isinstance(loss_func, WeightedLinkPredictBCELossFunc)
    assert decoder.gamma == 12.

    # transe + bayesian personalized ranking loss
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_TRANSE_L1_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_BPR,
            "lp_edge_weight_for_loss": None,
            "decoder_norm": None,
            "gamma": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictTransEDecoder)
    assert isinstance(loss_func, LinkPredictBPRLossFunc)
    assert decoder.gamma == 12.

    # transe + bayesian personalized ranking loss  + edge weight
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_TRANSE_L1_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_BPR,
            "decoder_norm": None,
            "lp_edge_weight_for_loss": "weight",
            "gamma": None,
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictWeightedTransEDecoder)
    assert isinstance(loss_func, WeightedLinkPredictBPRLossFunc)
    assert decoder.gamma == 12.

    # transe + contrastive loss
    config = GSTestConfig(
        {
            "lp_decoder_type": BUILTIN_LP_TRANSE_L1_DECODER,
            "lp_loss_func": BUILTIN_LP_LOSS_CONTRASTIVELOSS,
            "lp_edge_weight_for_loss": None,
            "decoder_norm": "l2norm",
            "contrastive_loss_temperature": 1.0,
            "gamma": 6.
        }
    )
    decoder, loss_func = create_builtin_lp_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, LinkPredictContrastiveTransEDecoder)
    assert isinstance(loss_func, LinkPredictContrastiveLossFunc)
    assert decoder.gamma == 6.

def test_check_graph_name():
    graph_name = "a"
    check_graph_name(graph_name)
    graph_name = "graph_name"
    check_graph_name(graph_name)
    graph_name = "graph-name"
    check_graph_name(graph_name)
    graph_name = "123-graph-name"
    check_graph_name(graph_name)
    graph_name = "_Graph-name"
    check_graph_name(graph_name)

    # test with invalid graph name
    graph_name = "/graph_name"
    with pytest.raises(AssertionError):
        check_graph_name(graph_name)

    graph_name = "|graph_name"
    with pytest.raises(AssertionError):
        check_graph_name(graph_name)

    graph_name = "\graph_name"
    with pytest.raises(AssertionError):
        check_graph_name(graph_name)

def test_get_edge_feat_size():
    g = generate_dummy_hetero_graph()

    # Test case 0: normal edge feature names
    edge_feat_names1 = 'feat'
    edge_feat_size = get_edge_feat_size(g, edge_feat_names1)

    assert edge_feat_size[("n0", "r0", "n1")] == 2
    assert edge_feat_size[("n0", "r1", "n1")] == 2

    edge_feat_names1 = {
        ("n0", "r0", "n1"): ['feat'],
        ("n0", "r1", "n1"): ['feat']
    }
    edge_feat_size = get_edge_feat_size(g, edge_feat_names1)

    assert edge_feat_size[("n0", "r0", "n1")] == 2
    assert edge_feat_size[("n0", "r1", "n1")] == 2

    # Test case 1: None edge feature names
    edge_feat_size = get_edge_feat_size(g, None)
    assert edge_feat_size[("n0", "r0", "n1")] == 0
    assert edge_feat_size[("n0", "r1", "n1")] == 0

    # Test case 2: Partial edge feature names
    edge_feat_names2 = {
        ("n0", "r0", "n1"): ['feat']
    }
    edge_feat_size = get_edge_feat_size(g, edge_feat_names2)
    assert edge_feat_size[("n0", "r0", "n1")] == 2
    assert edge_feat_size[("n0", "r1", "n1")] == 0

    # Test case 3: non 2D edge feature error
    edge_feat_names3 = {
        ("n0", "r1", "n1"): ['feat', 'label']
    }
    try:
        edge_feat_size = get_edge_feat_size(g, edge_feat_names3)
    except:
        edge_feat_size = None
    assert edge_feat_size is None

    # Test case 4: non-existing edge feature names, should raise assertion errors.
    edge_feat_names4 = {
        ("n0", "r0", "n1"): ['none']
    }
    try:
        edge_feat_size = get_edge_feat_size(g, edge_feat_names4)
    except:
        edge_feat_size = {}
    assert edge_feat_size == {}

    # Test case 5: non-existing edge types, should raise assertion errors.
    edge_feat_names5 = {
        ("n0", "r2", "n1"): ['feat']
    }
    try:
        edge_feat_size = get_edge_feat_size(g, edge_feat_names5)
    except:
        edge_feat_size = {}
    assert edge_feat_size == {}

@pytest.mark.parametrize("add_reverse_edges", [False, True])
def test_restore_builtin_model_from_artifacts(add_reverse_edges):
    """ Test restore a builtin mode from artifacts
    
    The test needs three inputs: 1/ model dir path, 2/ json file name, 3/ yaml file name.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # create a dummy gdsg dist graph
        if not add_reverse_edges:
            g, _, graph_config = generate_dummy_dist_graph(tmpdirname,
                                                       graph_name='test',
                                                       return_graph_config=True)
        else:
            g, _, graph_config = generate_dummy_dist_graph_hete_rev_edge(tmpdirname,
                                                                         graph_name='test',
                                                                         return_graph_config=True)
        # create dummy YAML config 
        yaml_object = create_dummy_config_obj()
        yaml_object["gsf"]["basic"] = {
            "backend": "gloo",
            "ip_config": os.path.join(tmpdirname, "ip.txt"),
            "part_config": os.path.join(tmpdirname, "part.json"),
            "model_encoder_type": "rgcn",
            "eval_frequency": 100,
            "no_validation": True,
        }
        yaml_object["gsf"]["gnn"]["hidden_size"] = 128
        yaml_object["gsf"]["input"] = {
            "node_feat_name": ["n0:feat,feat1", "n1:feat,feat1"]
        }
        # create dummpy ip.txt
        with open(os.path.join(tmpdirname, "ip.txt"), "w") as f:
            f.write("127.0.0.1\n")
        # create dummpy part.json
        with open(os.path.join(tmpdirname, "part.json"), "w") as f:
            json.dump({
                "graph_name": "test"
            }, f)

    # Case 1: test node classification model restoration
        # set to be a NC task
        yaml_object["gsf"]["node_classification"] ={
            "target_ntype": "n1",
            "label_field": "label",
            "multilabel": False,
            "num_classes": 10
        }
        with open(os.path.join(tmpdirname, "nc_basic.yaml"), "w") as f:
            yaml.dump(yaml_object, f)

        # use dummy dist hetero graph and yaml file to build a NC model and save
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'nc_basic.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        nc_model = create_builtin_node_gnn_model(g, config, train_task=False)
        nc_model.save_model(tmpdirname)

        # restore the model from the current temp file
        graph_config_file = os.path.basename(graph_config)

        nc_model, _, _ = restore_builtin_model_from_artifacts(tmpdirname, graph_config_file, 'nc_basic.yaml')

        assert nc_model.gnn_encoder.num_layers == yaml_object["gsf"]["gnn"]["num_layers"]
        assert nc_model.gnn_encoder.h_dims == yaml_object["gsf"]["gnn"]["hidden_size"]
        assert nc_model.gnn_encoder.out_dims == yaml_object["gsf"]["gnn"]["hidden_size"]
        assert isinstance(nc_model.gnn_encoder, RelationalGCNEncoder)
        assert isinstance(nc_model.decoder, EntityClassifier)
        assert nc_model.decoder.decoder.shape[1] == yaml_object["gsf"]["node_classification"]["num_classes"]

    # Case 2: test node regression model restoration   
        # set to be a NR task
        yaml_object["gsf"].pop('node_classification')
        yaml_object["gsf"]["node_regression"] ={
            "target_ntype": "n1",
            "label_field": "label",
            "eval_metric": "mse"
        }
        with open(os.path.join(tmpdirname, "nr_basic.yaml"), "w") as f:
            yaml.dump(yaml_object, f)

        # use dummy dist hetero graph and yaml file to build a NR model and save
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'nr_basic.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        nr_model = create_builtin_node_gnn_model(g, config, train_task=False)
        nr_model.save_model(tmpdirname)

        # restore the model from the current temp file
        nr_model, _, _ = restore_builtin_model_from_artifacts(tmpdirname, graph_config_file, 'nr_basic.yaml')

        # only check decoder shape as other model configurations are the same as nc model,
        # and regression decoder 
        assert nr_model.decoder.decoder.shape[1] == 1

    # Case 3: test edge classification model restoration   
        # set to be a EC task
        yaml_object["gsf"].pop('node_regression')
        yaml_object["gsf"]["edge_classification"] ={
            "target_ntype": "n0,r1,n1",
            "label_field": "label",
            "multilabel": False,
            "num_classes": 2
        }
        with open(os.path.join(tmpdirname, "ec_basic.yaml"), "w") as f:
            yaml.dump(yaml_object, f)

        # use dummy dist hetero graph and yaml file to build a EC model and save
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'ec_basic.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        ec_model = create_builtin_edge_gnn_model(g, config, train_task=False)
        ec_model.save_model(tmpdirname)

        # restore the model from the current temp file
        ec_model, _, _ = restore_builtin_model_from_artifacts(tmpdirname, graph_config_file, 'ec_basic.yaml')

        # only check decoder output features as other model configurations are the same as nc model,
        assert ec_model.decoder.combine_basis.out_features == yaml_object["gsf"]["edge_classification"]["num_classes"]

    # Case 4: test edge regression model restoration   
        # set to be a ER task
        yaml_object["gsf"].pop('edge_classification')
        yaml_object["gsf"]["edge_regression"] ={
            "target_ntype": "n0,r1,n1",
            "label_field": "label",
            "multilabel": False,
            "eval_metric": "mse"
        }
        with open(os.path.join(tmpdirname, "er_basic.yaml"), "w") as f:
            yaml.dump(yaml_object, f)

        # use dummy dist hetero graph and yaml file to build a ER model and save
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'er_basic.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        er_model = create_builtin_edge_gnn_model(g, config, train_task=False)
        er_model.save_model(tmpdirname)

        # restore the model from the current temp file
        er_model, _, _ = restore_builtin_model_from_artifacts(tmpdirname, graph_config_file, 'er_basic.yaml')

        # only check decoder output features as other model configurations are the same as nc model,
        assert er_model.decoder.regression_head.out_features == 1

    # Case 5: test link prediction model restoration   
        # set to be a LP task
        yaml_object["gsf"].pop('edge_regression')
        yaml_object["gsf"]["link_prediction"] = {
            "num_negative_edges": 4,
            "num_negative_edges_eval": 10,
            "train_negative_sampler": "joint",
            "train_etype": ["n0,r1,n1", "n0,r2,n1"]
        }
        with open(os.path.join(tmpdirname, "lp_basic.yaml"), "w") as f:
            yaml.dump(yaml_object, f)

        # use dummy dist hetero graph and yaml file to build a LP model and save
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_basic.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        lp_model = create_builtin_lp_gnn_model(g, config, train_task=False)
        lp_model.save_model(tmpdirname)

        # restore the model from the current temp file
        lp_model, _, _ = restore_builtin_model_from_artifacts(tmpdirname, graph_config_file, 'lp_basic.yaml')

        # only check decoder output features as other model configurations are the same as nc model,
        assert lp_model.decoder._w_relation.embedding_dim == yaml_object['gsf']['gnn']['hidden_size']
    
    # Case 6: test lm model layer model restoration
        yaml_object["gsf"].pop('link_prediction')
        yaml_object["gsf"]["node_classification"] ={
            "target_ntype": "n1",
            "label_field": "label",
            "multilabel": False,
            "num_classes": 10
        }
        yaml_object["lm_model"] = {}
        yaml_object["lm_model"]["node_lm_models"] = [{
            "lm_type": "bert",
            "model_name": "bert-base-uncased",
            "gradient_checkpoint": "true",
            "node_types": ["n1"]
        }]
        with open(os.path.join(tmpdirname, "nc_lm_basic.yaml"), "w") as f:
            yaml.dump(yaml_object, f)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'nc_lm_basic.yaml'),
                        local_rank=0)
        config = GSConfig(args)
        nc_lm_model = create_builtin_node_gnn_model(g, config, train_task=False)
        nc_lm_model.save_model(tmpdirname)

        nc_lm_model, _, _ = restore_builtin_model_from_artifacts(tmpdirname, graph_config_file, 'nc_lm_basic.yaml')
        assert nc_lm_model.node_input_encoder.node_feat_size == {'n0': 6, 'n1': 774}
        # RT Layer
        layer = nc_lm_model.node_input_encoder
        prefix = "_lm_models.n1."
        # Simple HuggingFace keys validation
        hf_model = AutoModel.from_pretrained("bert-base-uncased")
        hf_keys = set(hf_model.state_dict().keys())
        layer_keys = {name for name, param in layer.named_parameters()}
        # Huggingface position ID
        skip_buffers = {'embeddings.position_ids', 'embeddings.token_type_ids'}
        for hf_key in hf_keys:
            if hf_key in skip_buffers:
                continue
            prefix_hf_key = prefix + hf_key
            assert prefix_hf_key in layer_keys, \
                "Huggingface Model Keys are not in the restored model"

def test_save_load_builtin_models():
    """ Test save and load built-in GS models

    Built-in models contain: embed_layer (node encoder), edge_embed_layer, gnn_layers, and decoders.
    The saved models could include all or some of the four modules. And restored models too.

    This function only test the normal cases by following the built-in pipelines of GraphStorm.
    Because the differences among different tasks are the decoders only. This script just use a
    NC task.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # create a dummy gdsg dist graph
        g, _, graph_config = generate_dummy_dist_graph(tmpdirname,
                                                       graph_name='test',
                                                       return_graph_config=True)

        # create dummy YAML config 
        yaml_object = create_dummy_config_obj()
        yaml_object["gsf"]["basic"] = {
            "backend": "gloo",
            "ip_config": os.path.join(tmpdirname, "ip.txt"),
            "part_config": os.path.join(tmpdirname, "part.json"),
            "model_encoder_type": "rgcn",
            "eval_frequency": 100,
            "no_validation": True,
        }
        yaml_object["gsf"]["gnn"]["hidden_size"] = 128
        yaml_object["gsf"]["input"] = {
            "node_feat_name": ["n0:feat,feat1", "n1:feat,feat1"],
            "edge_feat_name": ["n0,r0,n1:feat", "n0,r1,n1:feat"]
        }
        # create dummpy ip.txt
        with open(os.path.join(tmpdirname, "ip.txt"), "w") as f:
            f.write("127.0.0.1\n")
        # create dummpy part.json
        with open(os.path.join(tmpdirname, "part.json"), "w") as f:
            json.dump({
                "graph_name": "test"
            }, f)

        # 1. node model
        yaml_object["gsf"]["node_classification"] ={
            "target_ntype": "n1",
            "label_field": "label",
            "multilabel": False,
            "num_classes": 10
        }
        with open(os.path.join(tmpdirname, "nc_basic.yaml"), "w") as f:
            yaml.dump(yaml_object, f)

        # use dummy dist hetero graph and yaml file to build a NC model and save
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'nc_basic.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        node_model = create_builtin_node_model(g, config, True)

        # save the model
        model_path = os.path.join(tmpdirname, 'models')
        node_model.save_model(model_path)
        
        assert os.path.exists(os.path.join(model_path, 'model.bin'))
        
        # load model dict and check contents
        checkpoint = th.load(os.path.join(model_path, 'model.bin'),
                             map_location='cpu',
                             weights_only=True)

        assert "node_embed" in checkpoint
        assert "edge_embed" in checkpoint
        assert "gnn" in checkpoint
        assert "decoder" in checkpoint

        # load model back
        node_model_copy = copy.deepcopy(node_model)
        if isinstance(node_model_copy.node_input_encoder, DDP):
            ori_node_input_encoder = node_model_copy.node_input_encoder.module
        else:
            ori_node_input_encoder = node_model_copy.node_input_encoder
        if isinstance(node_model_copy.edge_input_encoder, DDP):
            ori_edge_input_encoder = node_model_copy.edge_input_encoder.module
        else:
            ori_edge_input_encoder = node_model_copy.edge_input_encoder
        if isinstance(node_model_copy.gnn_encoder, DDP):
            ori_gnn_encoder = node_model_copy.gnn_encoder.module
        else:
            ori_gnn_encoder = node_model_copy.gnn_encoder
        if isinstance(node_model_copy.decoder, DDP):
            ori_decoder = node_model_copy.decoder.module
        else:
            ori_decoder = node_model_copy.decoder

        # recreate a new node model and change its parameters' values
        node_model = create_builtin_node_model(g, config, True)
        for param in node_model.parameters():
            param.data[:] += 1
        node_model.restore_model(model_path)

        if isinstance(node_model.node_input_encoder, DDP):
            res_node_input_encoder = node_model.node_input_encoder.module
        else:
            res_node_input_encoder = node_model.node_input_encoder
        if isinstance(node_model.edge_input_encoder, DDP):
            res_edge_input_encoder = node_model.edge_input_encoder.module
        else:
            res_edge_input_encoder = node_model.edge_input_encoder
        if isinstance(node_model.gnn_encoder, DDP):
            res_gnn_encoder = node_model.gnn_encoder.module
        else:
            res_gnn_encoder = node_model.gnn_encoder
        if isinstance(node_model.decoder, DDP):
            res_decoder = node_model.decoder.module
        else:
            res_decoder = node_model.decoder

        # compare the copied model against the restored model
        assert len(dict(ori_node_input_encoder.named_parameters())) == \
            len(dict(res_node_input_encoder.named_parameters()))
        assert len(dict(ori_edge_input_encoder.named_parameters())) == \
            len(dict(res_edge_input_encoder.named_parameters()))

        for p_name, param in res_node_input_encoder.named_parameters():
            assert np.all(ori_node_input_encoder.get_parameter(p_name).data.numpy() == param.data.numpy())
        for p_name, param in res_edge_input_encoder.named_parameters():
            assert np.all(ori_edge_input_encoder.get_parameter(p_name).data.numpy() == param.data.numpy())
        for p_name, param in res_gnn_encoder.named_parameters():
            assert np.all(ori_gnn_encoder.get_parameter(p_name).data.numpy() == param.data.numpy())
        for p_name, param in res_decoder.named_parameters():
            assert np.all(ori_decoder.get_parameter(p_name).data.numpy() == param.data.numpy())
