"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GSgnn multi-task learning
"""
import os

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.dataloading import GSgnnData
from graphstorm.trainer import GSgnnMultiTaskLearningTrainer

from graphstorm.utils import rt_profiler, sys_tracker, get_device, use_wholegraph
from graphstorm.utils import get_lm_ntypes

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(True)

    use_wg_feats = use_wholegraph(config.part_config)
    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  local_rank=config.local_rank,
                  use_wholegraph=config.use_wholegraph_embed or use_wg_feats)
    rt_profiler.init(config.profile_path, rank=gs.get_rank())
    sys_tracker.init(config.verbose, rank=gs.get_rank())
    train_data = GSgnnData(config.part_config,
                           node_feat_field=config.node_feat_name,
                           edge_feat_field=config.edge_feat_name,
                           lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))

    tasks = config.multi_tasks
    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []
    decoders = []
    for task in tasks:
        train_loader = create_task_train_dataloader(task, config, train_task=True)
        decoder = create_task_decoder(task, config)
        val_loader = create_task_val_dataloader(task, config, train_task=False)
        test_loader = create_task_test_dataloader(task, config, train_task=False)

    train_dataloader = GSgnnMultiTaskDataLoader(train_dataloaders)
    val_dataloader = GSgnnMultiTaskDataLoader(val_dataloaders)
    test_dataloader = GSgnnMultiTaskDataLoader(test_dataloaders)

    trainer = GSgnnMultiTaskLearningTrainer(model, topk_model_to_save=config.topk_model_to_save)

    # Preparing input layer for training or inference.
    # The input layer can pre-compute node features in the preparing step if needed.
    # For example pre-compute all BERT embeddings
    model.prepare_input_encoder(train_data)
    if config.save_model_path is not None:
        save_model_path = config.save_model_path
    elif config.save_embed_path is not None:
        # If we need to save embeddings, we need to save the model somewhere.
        save_model_path = os.path.join(config.save_embed_path, "model")
    else:
        save_model_path = None

    trainer.fit(train_loader=train_dataloader,
                val_loader=val_dataloader,
                test_loader=test_dataloader,
                num_epochs=config.num_epochs,
                save_model_path=save_model_path,
                use_mini_batch_infer=config.use_mini_batch_infer,
                save_model_frequency=config.save_model_frequency,
                save_perf_results_path=config.save_perf_results_path,
                freeze_input_layer_epochs=config.freeze_lm_encoder_epochs,
                max_grad_norm=config.max_grad_norm,
                grad_norm_type=config.grad_norm_type)

    if config.save_embed_path is not None:
        # Save node embeddings
