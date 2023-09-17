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

    Inference and training script for distillation tasks with GNN
"""

import random
import os
import torch as th
import numpy as np
import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.distiller import GSdistiller
from graphstorm.utils import setup_device


def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(False)
    gs.initialize(ip_config=config.ip_config, backend=config.backend)
    device = setup_device(config.local_rank)


    distiller = GSdistiller(gs.get_rank())
    distiller.setup_device(device=device)
    th.distributed.barrier()

    distiller.distill(
        config.lm_name,
        config.pre_trained_name,
        config.textual_data_path,
        config.batch_size,
        config.max_seq_len,
        config.lm_tune_lr,
        saved_path=config.save_model_path,
        save_model_frequency=config.save_model_frequency,
        eval_frequency=config.eval_frequency,
        max_global_step=config.max_global_step,
    )

def generate_parser():
    """ Generate an argument parser
    """
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    arg_parser=generate_parser()

    args = arg_parser.parse_args()
    print(args)
    main(args)
