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

    GSgnn HAT masked language model pre-training
"""

import os
import torch as th
import graphstorm as gs

from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig

from graphstorm.dataloading import GSgnnNodeTrainData
from graphstorm.trainer import GSgnnHATMasedLMTrainer
from graphstorm.utils import rt_profiler, sys_tracker, setup_device

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(True)

    gs.initialize(ip_config=config.ip_config, backend=config.backend)

    rt_profiler.init(config.profile_path, rank=gs.get_rank())
    sys_tracker.init(config.verbose, rank=gs.get_rank())
    device = setup_device(config.local_rank)

    train_data = GSgnnNodeTrainData(config.graph_name,
                                    config.part_config,
                                    train_ntypes=config.target_ntype,
                                    node_feat_field=config.node_feat_name,
                                    label_field=config.label_field)

    model = gs.create_builtin_hat_model(train_data.g, config, train_task=True)

    trainer = GSgnnHATMasedLMTrainer(model, gs.get_rank(),
        topk_model_to_save=config.topk_model_to_save)

    trainer.fit(train_data, )
