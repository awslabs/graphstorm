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

from .config import BUILTIN_TASK_NODE_CLASSIFICATION
from .config import BUILTIN_TASK_NODE_REGRESSION
from .config import BUILTIN_TASK_EDGE_CLASSIFICATION
from .config import BUILTIN_TASK_EDGE_REGRESSION
from .config import BUILTIN_TASK_LINK_PREDICTION
from .config import SUPPORTED_TASKS

from .config import BUILTIN_LP_DOT_DECODER
from .config import BUILTIN_LP_DISTMULT_DECODER
from .config import SUPPORTED_LP_DECODER

from .config import (GRAPHSTORM_MODEL_EMBED_LAYER,
                     GRAPHSTORM_MODEL_GNN_LAYER,
                     GRAPHSTORM_MODEL_DECODER_LAYER,
                     GRAPHSTORM_MODEL_ALL_LAYERS)
