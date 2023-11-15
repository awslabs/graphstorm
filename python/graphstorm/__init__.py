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

    Graphstorm package.
"""
__version__ = "0.2.1"

from . import gsf
from . import utils
from .utils import get_rank, get_world_size
from .gsf import initialize, get_feat_size
from .gsf import create_builtin_node_gnn_model
from .gsf import create_builtin_edge_gnn_model
from .gsf import create_builtin_task_tracker
from .gsf import create_builtin_lp_gnn_model
from .gsf import create_builtin_lp_model
from .gsf import create_builtin_edge_model
from .gsf import create_builtin_node_model
