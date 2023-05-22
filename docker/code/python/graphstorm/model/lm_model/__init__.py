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

    Package initialization. Import necessary classes for Language model support
"""
from .utils import init_lm_model, get_lm_node_feats
from .utils import BUILTIN_HF_BERT

from .lm_model import TOKEN_IDX, VALID_LEN, ATT_MASK_IDX, TOKEN_TID_IDX
