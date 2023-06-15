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

    Builtin configs
"""

BUILTIN_GNN_ENCODER = ["rgat", "rgcn"]
BUILTIN_ENCODER = ["lm", "mlp"] + ["rgat", "rgcn"]
SUPPORTED_BACKEND = ["gloo", "nccl"]

GRAPHSTORM_MODEL_EMBED_LAYER = "embed"
GRAPHSTORM_MODEL_GNN_LAYER = "gnn"
GRAPHSTORM_MODEL_DECODER_LAYER = "decoder"
GRAPHSTORM_MODEL_ALL_LAYERS = [GRAPHSTORM_MODEL_EMBED_LAYER,
                               GRAPHSTORM_MODEL_GNN_LAYER,
                               GRAPHSTORM_MODEL_DECODER_LAYER]

BUILTIN_LP_LOSS_CROSS_ENTROPY = "cross_entropy"
BUILTIN_LP_LOSS_LOGSIGMOID_RANKING = "logsigmoid"
BUILTIN_LP_LOSS_FUNCTION = [BUILTIN_LP_LOSS_CROSS_ENTROPY, \
    BUILTIN_LP_LOSS_LOGSIGMOID_RANKING]

BUILTIN_TASK_NODE_CLASSIFICATION = "node_classification"
BUILTIN_TASK_NODE_REGRESSION = "node_regression"
BUILTIN_TASK_EDGE_CLASSIFICATION = "edge_classification"
BUILTIN_TASK_EDGE_REGRESSION = "edge_regression"
BUILTIN_TASK_LINK_PREDICTION = "link_prediction"

SUPPORTED_TASKS  = [BUILTIN_TASK_NODE_CLASSIFICATION, \
    BUILTIN_TASK_NODE_REGRESSION, \
    BUILTIN_TASK_EDGE_CLASSIFICATION, \
    BUILTIN_TASK_LINK_PREDICTION, \
    BUILTIN_TASK_EDGE_REGRESSION]

EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY = "consecutive_increase"
EARLY_STOP_AVERAGE_INCREASE_STRATEGY = "average_increase"

# Task tracker
GRAPHSTORM_SAGEMAKER_TASK_TRACKER = "sagemaker_task_tracker"

SUPPORTED_TASK_TRACKER = [GRAPHSTORM_SAGEMAKER_TASK_TRACKER]

# Link prediction decoder
BUILTIN_LP_DOT_DECODER = "dot_product"
BUILTIN_LP_DISTMULT_DECODER = "distmult"

SUPPORTED_LP_DECODER = [BUILTIN_LP_DOT_DECODER, BUILTIN_LP_DISTMULT_DECODER]
