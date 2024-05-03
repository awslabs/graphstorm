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
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                               BUILTIN_TASK_NODE_REGRESSION,
                               BUILTIN_TASK_EDGE_CLASSIFICATION,
                               BUILTIN_TASK_EDGE_REGRESSION,
                               BUILTIN_TASK_LINK_PREDICTION)
from graphstorm.dataloading import GSgnnData
from graphstorm.dataloading import (GSgnnNodeDataLoader,
                                    GSgnnEdgeDataLoader,
                                    GSgnnMultiTaskDataLoader)
from graphstorm.trainer import GSgnnMultiTaskLearningTrainer

from graphstorm.utils import rt_profiler, sys_tracker, get_device, use_wholegraph
from graphstorm.utils import get_lm_ntypes

def create_task_train_dataloader(task, config, train_data):
    if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        train_idxs = train_data.get_node_train_set(config.target_ntype)
        return GSgnnNodeDataLoader(train_data,
                                   train_idxs,
                                   fanout=config.fanout,
                                   batch_size=config.batch_size,
                                   train_task=True,
                                   node_feats=config.node_feat_name,
                                   label_field=config.label_field)
    elif task.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
        train_idxs = train_data.get_edge_train_set(config.target_etype)
        return GSgnnEdgeDataLoader(train_data,
                                   train_idxs,
                                   fanout=config.fanout,
                                   batch_size=config.batch_size,
                                   node_feats=config.node_feat_name,
                                   label_field=config.label_field,
                                   decoder_edge_feats=config.decoder_edge_feat,
                                   train_task=True,
                                   reverse_edge_types_map=config.reverse_edge_types_map,
                                   remove_target_edge_type=config.remove_target_edge_type,
                                   exclude_training_targets=config.exclude_training_targets)
    elif task.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
        train_idxs = train_data.get_edge_train_set(config.train_etype)
        dataloader_cls = gs.get_lp_train_sampler(config)
        return dataloader_cls(train_data,
                              train_idxs,
                              config.fanout,
                              config.batch_size,
                              config.num_negative_edges,
                              node_feats=config.node_feat_name,
                              pos_graph_edge_feats=config.lp_edge_weight_for_loss,
                              train_task=True,
                              reverse_edge_types_map=config.reverse_edge_types_map,
                              exclude_training_targets=config.exclude_training_targets,
                              edge_dst_negative_field=config.train_etypes_negative_dstnode,
                              num_hard_negs=config.num_train_hard_negatives)

    return None

def create_task_val_dataloader(task, config, train_data):
    fanout = config.eval_fanout if config.use_mini_batch_infer else []
    if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        eval_ntype = config.eval_target_ntype \
            if config.eval_target_ntype is not None else config.target_ntype
        val_idxs = train_data.get_node_val_set(eval_ntype)

        if len(val_idxs) > 0:
            return GSgnnNodeDataLoader(train_data,
                                       val_idxs,
                                       fanout=fanout,
                                       batch_size=config.eval_batch_size,
                                       train_task=False,
                                       node_feats=config.node_feat_name,
                                       label_field=config.label_field,
                                       construct_feat_ntype=config.construct_feat_ntype,
                                       construct_feat_fanout=config.construct_feat_fanout)
    elif task.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
        val_idxs = train_data.get_edge_val_set(config.target_etype)
        if len(val_idxs) > 0:
            return GSgnnEdgeDataLoader(train_data,
                                       val_idxs,
                                       fanout=fanout,
                                       batch_size=config.eval_batch_size,
                                       node_feats=config.node_feat_name,
                                       label_field=config.label_field,
                                       decoder_edge_feats=config.decoder_edge_feat,
                                       train_task=False,
                                       reverse_edge_types_map=config.reverse_edge_types_map,
                                       remove_target_edge_type=config.remove_target_edge_type)
    elif task.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
        val_idxs = train_data.get_edge_val_set(config.eval_etype)
        dataloader_cls = gs.get_lp_eval_sampler(config)
        if len(val_idxs) > 0:
            if config.eval_etypes_negative_dstnode is not None:
                return dataloader_cls(train_data, val_idxs,
                    config.eval_batch_size,
                    fixed_edge_dst_negative_field=config.eval_etypes_negative_dstnode,
                    fanout=config.eval_fanout,
                    fixed_test_size=config.fixed_test_size,
                    node_feats=config.node_feat_name,
                    pos_graph_edge_feats=config.lp_edge_weight_for_loss)
            else:
                return dataloader_cls(train_data, val_idxs,
                    config.eval_batch_size,
                    config.num_negative_edges_eval, config.eval_fanout,
                    fixed_test_size=config.fixed_test_size,
                    node_feats=config.node_feat_name,
                    pos_graph_edge_feats=config.lp_edge_weight_for_loss)

    return None

def create_task_test_dataloader(task, config, train_data):
    if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        eval_ntype = config.eval_target_ntype \
            if config.eval_target_ntype is not None else config.target_ntype
        test_idxs = train_data.get_node_test_set(eval_ntype)
        fanout = config.eval_fanout if config.use_mini_batch_infer else []
        if len(test_idxs) > 0:
            return GSgnnNodeDataLoader(train_data,
                                       test_idxs,
                                       fanout=fanout,
                                       batch_size=config.eval_batch_size,
                                       train_task=False,
                                       node_feats=config.node_feat_name,
                                       label_field=config.label_field,
                                       construct_feat_ntype=config.construct_feat_ntype,
                                       construct_feat_fanout=config.construct_feat_fanout)

    elif task.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
        test_idxs = train_data.get_edge_test_set(config.target_etype)
        if len(test_idxs) > 0:
            return GSgnnEdgeDataLoader(train_data,
                                       test_idxs,
                                       fanout=fanout,
                                       batch_size=config.eval_batch_size,
                                       node_feats=config.node_feat_name,
                                       label_field=config.label_field,
                                       decoder_edge_feats=config.decoder_edge_feat,
                                       train_task=False,
                                       reverse_edge_types_map=config.reverse_edge_types_map,
                                       remove_target_edge_type=config.remove_target_edge_type)
    elif task.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
        test_idxs = train_data.get_edge_test_set(config.eval_etype)
        dataloader_cls = gs.get_lp_eval_sampler(config)
        if len(test_idxs) > 0:
            if config.eval_etypes_negative_dstnode is not None:
                return dataloader_cls(train_data, test_idxs,
                    config.eval_batch_size,
                    fixed_edge_dst_negative_field=config.eval_etypes_negative_dstnode,
                    fanout=config.eval_fanout,
                    fixed_test_size=config.fixed_test_size,
                    node_feats=config.node_feat_name,
                    pos_graph_edge_feats=config.lp_edge_weight_for_loss)
            else:
                return dataloader_cls(train_data, test_idxs,
                    config.eval_batch_size, config.num_negative_edges_eval, config.eval_fanout,
                    fixed_test_size=config.fixed_test_size,
                    node_feats=config.node_feat_name,
                    pos_graph_edge_feats=config.lp_edge_weight_for_loss)
    return None

def create_task_decoder(task, config):

def create_evaluator(task, config):

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
        train_loader = create_task_train_dataloader(task, config, train_data)
        val_loader = create_task_val_dataloader(task, config)
        test_loader = create_task_test_dataloader(task, config)
        decoder = create_task_decoder(task, config)
        evaluator = create_evaluator(task, config)

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
