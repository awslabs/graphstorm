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

    Config parser
"""

from .argument import GSConfig
from .argument import get_argument_parser

from .config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                     BUILTIN_TASK_NODE_REGRESSION,
                     BUILTIN_TASK_EDGE_CLASSIFICATION,
                     BUILTIN_TASK_EDGE_REGRESSION,
                     BUILTIN_TASK_LINK_PREDICTION,
                     BUILTIN_TASK_COMPUTE_EMB,
                     BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                     BUILTIN_TASK_MULTI_TASK)
from .config import SUPPORTED_TASKS

from .config import (BUILTIN_LP_DOT_DECODER,
                     BUILTIN_LP_DISTMULT_DECODER,
                     BUILTIN_LP_ROTATE_DECODER)
from .config import SUPPORTED_LP_DECODER

from .config import (GRAPHSTORM_MODEL_EMBED_LAYER,
                     GRAPHSTORM_MODEL_GNN_LAYER,
                     GRAPHSTORM_MODEL_DECODER_LAYER,
                     GRAPHSTORM_MODEL_ALL_LAYERS,
                     GRAPHSTORM_MODEL_LAYER_OPTIONS,
                     GRAPHSTORM_MODEL_DENSE_EMBED_LAYER,
                     GRAPHSTORM_MODEL_SPARSE_EMBED_LAYER)
from .config import (BUILTIN_GNN_NORM,
                     BUILDIN_GNN_LAYER_NORM,
                     BUILDIN_GNN_BATCH_NORM)

from .config import (BUILTIN_LP_LOSS_CROSS_ENTROPY,
                     BUILTIN_LP_LOSS_CONTRASTIVELOSS)
from .config import (GRAPHSTORM_LP_EMB_L2_NORMALIZATION,
                     GRAPHSTORM_LP_EMB_NORMALIZATION_METHODS)

from .config import TaskInfo
