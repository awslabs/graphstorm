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
import pytest

from graphstorm.gsf import (get_edge_feat_size,
                            create_builtin_node_decoder,
                            create_builtin_edge_decoder,
                            create_builtin_lp_decoder)
from graphstorm.utils import check_graph_name
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                               BUILTIN_TASK_NODE_REGRESSION,
                               BUILTIN_TASK_EDGE_CLASSIFICATION,
                               BUILTIN_TASK_EDGE_REGRESSION,
                               BUILTIN_LP_DOT_DECODER,
                               BUILTIN_LP_DISTMULT_DECODER,
                               BUILTIN_LP_ROTATE_DECODER,
                               BUILTIN_CLASS_LOSS_CROSS_ENTROPY,
                               BUILTIN_CLASS_LOSS_FOCAL,
                               BUILTIN_LP_LOSS_CROSS_ENTROPY,
                               BUILTIN_LP_LOSS_CONTRASTIVELOSS)

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
                                           LinkPredictWeightedRotatEDecoder)

from graphstorm.model.loss_func import (ClassifyLossFunc,
                                        RegressionLossFunc,
                                        LinkPredictBCELossFunc,
                                        WeightedLinkPredictBCELossFunc,
                                        LinkPredictContrastiveLossFunc,
                                        LinkPredictAdvBCELossFunc,
                                        WeightedLinkPredictAdvBCELossFunc,
                                        FocalLossFunc)

from data_utils import generate_dummy_hetero_graph

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
            "num_classes": 1,
            "multilabel": False,
            "class_loss_func": BUILTIN_CLASS_LOSS_FOCAL,
            "multilabel_weights": None,
            "decoder_norm": None,
            "imbalance_class_weights": None,
            "alpha": None,
            "gamma": None,
            "decoder_bias": False,
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, EntityClassifier)
    assert isinstance(loss_func, FocalLossFunc)
    assert loss_func.alpha == 0.25
    assert loss_func.gamma == 2.

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
                "n0": 1,
                "n1": 1
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

    # node regression
    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_NODE_REGRESSION,
            "decoder_norm": None,
            "decoder_bias": False,
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, EntityRegression)
    assert isinstance(loss_func, RegressionLossFunc)

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
            "num_classes": 1,
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
            "num_classes": 1,
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
        }
    )
    decoder, loss_func = create_builtin_edge_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, MLPEdgeDecoder)
    assert isinstance(loss_func, RegressionLossFunc)

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


if __name__ == '__main__':
    test_check_graph_name()
    test_create_builtin_node_decoder()
    test_create_builtin_edge_decoder()
    test_create_builtin_lp_decoder()
    test_get_edge_feat_size()
