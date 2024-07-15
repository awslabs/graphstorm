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

    Training script for distillation tasks with GNN
"""

import os
import logging
import torch as th
import graphstorm as gs
from graphstorm.config import GSConfig, get_argument_parser
from graphstorm.distiller import GSdistiller
from graphstorm.utils import get_device, barrier
from graphstorm.model.gnn_distill import GSDistilledModel
from graphstorm.dataloading import DistillDataloaderGenerator, DistillDataManager


def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(True)
    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  local_rank=config.local_rank)

    # initiate model
    student_model = GSDistilledModel(lm_type=config.distill_lm_configs[0]["lm_type"],
        pre_trained_name=config.distill_lm_configs[0]["model_name"])

    # initiate DataloaderGenerator and DataManager
    dataloader_generator = DistillDataloaderGenerator(tokenizer=student_model.tokenizer,
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
    )
    train_data_mgr = DistillDataManager(
        dataloader_generator,
        dataset_path=os.path.join(config.textual_data_path, 'train'),
        local_rank=config.local_rank,
        world_size=th.distributed.get_world_size(),
        is_train=True,
    )
    eval_data_mgr = DistillDataManager(
        dataloader_generator,
        dataset_path=os.path.join(config.textual_data_path, 'val'),
        local_rank=config.local_rank,
        world_size=th.distributed.get_world_size(),
        is_train=False,
    )

    # get GNN embed dim
    dataset_iterator = eval_data_mgr.get_iterator()
    eval_data_mgr.refresh_manager()
    if not dataset_iterator:
        raise RuntimeError("No validation data")
    batch = next(iter(dataset_iterator))
    gnn_embed_dim = batch["labels"].shape[1]
    student_model.init_proj_layer(gnn_embed_dim=gnn_embed_dim)

    # initiate distiller
    distiller = GSdistiller(model=student_model)
    distiller.setup_device(device=get_device())
    barrier()

    distiller.fit(
        train_data_mgr=train_data_mgr,
        eval_data_mgr=eval_data_mgr,
        distill_lr=config.lm_tune_lr,
        saved_path=config.save_model_path,
        save_model_frequency=config.save_model_frequency,
        eval_frequency=config.eval_frequency,
        max_distill_step=config.max_distill_step,
    )

def generate_parser():
    """ Generate an argument parser
    """
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    arg_parser=generate_parser()

    # Ignore unknown args to make script more robust to input arguments
    gs_args, unknown_args = arg_parser.parse_known_args()
    logging.warning("Unknown arguments for command "
                    "graphstorm.run.gs_gnn_distillation: %s",
                    unknown_args)
    main(gs_args)
