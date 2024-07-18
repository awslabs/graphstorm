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
import dataclasses
import typing

BUILTIN_GNN_ENCODER = ["gat", "rgat", "rgcn", "sage", "hgt", "gatv2"]
BUILTIN_ENCODER = ["lm", "mlp"] + BUILTIN_GNN_ENCODER
SUPPORTED_BACKEND = ["gloo", "nccl"]

GRAPHSTORM_MODEL_EMBED_LAYER = "embed"
GRAPHSTORM_MODEL_DENSE_EMBED_LAYER = "dense_embed"
GRAPHSTORM_MODEL_SPARSE_EMBED_LAYER = "sparse_embed"
GRAPHSTORM_MODEL_GNN_LAYER = "gnn"
GRAPHSTORM_MODEL_DECODER_LAYER = "decoder"
GRAPHSTORM_MODEL_ALL_LAYERS = [GRAPHSTORM_MODEL_EMBED_LAYER,
                               GRAPHSTORM_MODEL_GNN_LAYER,
                               GRAPHSTORM_MODEL_DECODER_LAYER]
GRAPHSTORM_MODEL_LAYER_OPTIONS = GRAPHSTORM_MODEL_ALL_LAYERS + \
        [GRAPHSTORM_MODEL_DENSE_EMBED_LAYER,
         GRAPHSTORM_MODEL_SPARSE_EMBED_LAYER]

BUILTIN_LP_LOSS_CROSS_ENTROPY = "cross_entropy"
BUILTIN_LP_LOSS_LOGSIGMOID_RANKING = "logsigmoid"
BUILTIN_LP_LOSS_CONTRASTIVELOSS = "contrastive"
BUILTIN_LP_LOSS_FUNCTION = [BUILTIN_LP_LOSS_CROSS_ENTROPY, \
    BUILTIN_LP_LOSS_LOGSIGMOID_RANKING, BUILTIN_LP_LOSS_CONTRASTIVELOSS]

GRAPHSTORM_LP_EMB_L2_NORMALIZATION = "l2_norm"
GRAPHSTORM_LP_EMB_NORMALIZATION_METHODS = [GRAPHSTORM_LP_EMB_L2_NORMALIZATION]

BUILDIN_GNN_BATCH_NORM = 'batch'
BUILDIN_GNN_LAYER_NORM = 'layer'
BUILTIN_GNN_NORM = [BUILDIN_GNN_BATCH_NORM, BUILDIN_GNN_LAYER_NORM]

BUILTIN_TASK_NODE_CLASSIFICATION = "node_classification"
BUILTIN_TASK_NODE_REGRESSION = "node_regression"
BUILTIN_TASK_EDGE_CLASSIFICATION = "edge_classification"
BUILTIN_TASK_EDGE_REGRESSION = "edge_regression"
BUILTIN_TASK_LINK_PREDICTION = "link_prediction"
BUILTIN_TASK_COMPUTE_EMB = "compute_emb"
BUILTIN_TASK_RECONSTRUCT_NODE_FEAT = "reconstruct_node_feat"
BUILTIN_TASK_MULTI_TASK = "multi_task"

LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL = "ALL"

SUPPORTED_TASKS  = [BUILTIN_TASK_NODE_CLASSIFICATION, \
    BUILTIN_TASK_NODE_REGRESSION, \
    BUILTIN_TASK_EDGE_CLASSIFICATION, \
    BUILTIN_TASK_LINK_PREDICTION, \
    BUILTIN_TASK_EDGE_REGRESSION, \
    BUILTIN_TASK_RECONSTRUCT_NODE_FEAT]

EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY = "consecutive_increase"
EARLY_STOP_AVERAGE_INCREASE_STRATEGY = "average_increase"

# Task tracker
GRAPHSTORM_SAGEMAKER_TASK_TRACKER = "sagemaker_task_tracker"

SUPPORTED_TASK_TRACKER = [GRAPHSTORM_SAGEMAKER_TASK_TRACKER]

# Link prediction decoder
BUILTIN_LP_DOT_DECODER = "dot_product"
BUILTIN_LP_DISTMULT_DECODER = "distmult"

SUPPORTED_LP_DECODER = [BUILTIN_LP_DOT_DECODER, BUILTIN_LP_DISTMULT_DECODER]

################ Task info data classes ############################
def get_mttask_id(task_type, ntype=None, etype=None, label=None):
    """ Generate task ID for multi-task learning tasks.
        The ID is composed of the task type, the node type
        or the edge type(s) and the label of a task.

    Parameters
    ----------
    task_type: str
        Task type.
    ntype: str
        Node type.
    etype: str, tuple or list of tuple
        Edge type. It can be "ALL_ETYPE" meaning all the etypes.
        It can be a tuple representing an edge type.
        It can be a list of tuples representing a list of edge types.
    label: str
        Label name.

    Return
    ------
    str: Task ID.
    """
    task_id = [task_type]
    if ntype is not None:
        task_id.append(ntype) # node task
    if etype is not None:
        if isinstance(etype, str):
            task_id.append(etype)
        elif isinstance(etype, tuple):
            task_id.append("_".join(etype))
        elif isinstance(etype, list): # a list of etypes
            task_id.append("__".join(["_".join(et) for et in etype]))
        else:
            raise TypeError(f"Unknown etype format: {etype}. Must be a string " \
                            "or a tuple of strings or a list of tuples of strings.")
    if label is not None:
        task_id.append(label)

    return "-".join(task_id)

@dataclasses.dataclass
class TaskInfo:
    """Information of a training task in multi-task learning

    Parameters
    ----------
    task_type: str
        Task type.
    task_id: str
        Task id. Unique id for each task.
    batch_size: int
        Batch size of the current task.
    mask_fields: list
        Train/validation/test mask fields.
    dataloader:
        Task dataloader.
    eval_metric: list
        Evaluation metrics
    task_weight: float
        Weight of the task in final loss.
    """
    task_type : str
    task_id : str
    task_config : typing.Any = None
    dataloader : typing.Any = None # dataloder
