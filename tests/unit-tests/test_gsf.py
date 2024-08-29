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

import torch as th

from graphstorm.gsf import create_builtin_node_decoder
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                               BUILTIN_TASK_NODE_REGRESSION,
                               BUILTIN_CLASS_LOSS_CROSS_ENTROPY,
                               BUILTIN_CLASS_LOSS_FOCAL)

from graphstorm.model.node_decoder import (EntityClassifier,
                                           EntityRegression)

from graphstorm.model.loss_func import (ClassifyLossFunc,
                                        RegressionLossFunc,
                                        LinkPredictBCELossFunc,
                                        WeightedLinkPredictBCELossFunc,
                                        LinkPredictContrastiveLossFunc,
                                        FocalLossFunc)

class GSTestConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def test_create_builtin_node_decoder():
    g = None
    decoder_input_dim = 64
    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_NODE_CLASSIFICATION,
            "num_classes": 2,
            "multilabel": False,
            "class_loss_func": BUILTIN_CLASS_LOSS_CROSS_ENTROPY,
            "multilabel_weights": None,
            "decoder_norm": None,
            "imbalance_class_weights": None
        }
    )
    train_task=False

    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, EntityClassifier)
    assert isinstance(loss_func, ClassifyLossFunc)

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
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, EntityClassifier)
    assert isinstance(loss_func, FocalLossFunc)
    assert loss_func.alpha == 0.25
    assert loss_func.gamma == 2.

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
            "decoder_norm": None
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, dict)
    assert isinstance(decoder["n0"], EntityClassifier)
    assert isinstance(decoder["n1"], EntityClassifier)
    assert isinstance(loss_func, dict)
    assert isinstance(loss_func["n0"], ClassifyLossFunc)
    assert isinstance(loss_func["n1"], ClassifyLossFunc)

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
            "alpha": None,
            "gamma": None,
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, dict)
    assert isinstance(decoder["n0"], EntityClassifier)
    assert isinstance(decoder["n1"], EntityClassifier)
    assert isinstance(loss_func, dict)
    assert isinstance(loss_func["n0"], FocalLossFunc)
    assert isinstance(loss_func["n1"], FocalLossFunc)

    config = GSTestConfig(
        {
            "task_type": BUILTIN_TASK_NODE_REGRESSION,
            "decoder_norm": None
        }
    )
    decoder, loss_func = create_builtin_node_decoder(g, decoder_input_dim, config, train_task)
    assert isinstance(decoder, EntityRegression)
    assert isinstance(loss_func, RegressionLossFunc)


if __name__ == '__main__':
    test_create_builtin_node_decoder()