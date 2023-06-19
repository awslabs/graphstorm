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

    Graphstorm trainer
"""
from .lp_trainer import GSgnnLinkPredictionTrainer
from .np_trainer import GSgnnNodePredictionTrainer
from .ep_trainer import GSgnnEdgePredictionTrainer
from .gsgnn_trainer import GSgnnTrainer
from .glem_np_trainer import GLEMNodePredictionTrainer
