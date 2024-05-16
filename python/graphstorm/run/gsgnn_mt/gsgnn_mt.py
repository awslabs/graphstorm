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
import logging

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

from graphstorm.eval import (GSgnnClassificationEvaluator,
                             GSgnnRegressionEvaluator,
                             GSgnnPerEtypeMrrLPEvaluator,
                             GSgnnMrrLPEvaluator,
                             GSgnnMultiTaskEvaluator)
from graphstorm.model.multitask_gnn import GSgnnMultiTaskSharedEncoderModel
from graphstorm.trainer import GSgnnMultiTaskLearningTrainer
from graphstorm.model.utils import save_full_node_embeddings
from graphstorm.model import do_full_graph_inference

from graphstorm.utils import rt_profiler, sys_tracker, get_device, use_wholegraph
from graphstorm.utils import get_lm_ntypes

def create_task_train_dataloader(task, config, task_config, train_data):
    """
    """
    fanout = config.fanout
    # All tasks share the same input encoder, so the node feats must be same.
    node_feats = config.node_feat_name
    if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        train_idxs = train_data.get_node_train_set(config.target_ntype)
        # TODO(xiangsx): Support construct feat
        return GSgnnNodeDataLoader(train_data,
                                   train_idxs,
                                   fanout=fanout,
                                   batch_size=task_config.batch_size,
                                   train_task=True,
                                   node_feats=node_feats,
                                   label_field=task_config.label_field)
    elif task.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
        train_idxs = train_data.get_edge_train_set(config.target_etype)
        # TODO(xiangsx): Support construct feat
        return GSgnnEdgeDataLoader(train_data,
                                   train_idxs,
                                   fanout=fanout,
                                   batch_size=task_config.batch_size,
                                   node_feats=node_feats,
                                   label_field=task_config.label_field,
                                   decoder_edge_feats=task_config.decoder_edge_feat,
                                   train_task=True,
                                   reverse_edge_types_map=task_config.reverse_edge_types_map,
                                   remove_target_edge_type=task_config.remove_target_edge_type,
                                   exclude_training_targets=task_config.exclude_training_targets)
    elif task.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
        train_idxs = train_data.get_edge_train_set(config.train_etype)
        dataloader_cls = gs.get_lp_train_sampler(config)
        return dataloader_cls(train_data,
                              train_idxs,
                              fanout,
                              task_config.batch_size,
                              task_config.num_negative_edges,
                              node_feats=node_feats,
                              pos_graph_edge_feats=task_config.lp_edge_weight_for_loss,
                              train_task=True,
                              reverse_edge_types_map=task_config.reverse_edge_types_map,
                              exclude_training_targets=task_config.exclude_training_targets,
                              edge_dst_negative_field=task_config.train_etypes_negative_dstnode,
                              num_hard_negs=task_config.num_train_hard_negatives)

    return None

def create_task_val_dataloader(task, config, task_config, train_data):
    """
    """
    # All tasks share the same input encoder, so the node feats must be same.
    node_feats = config.node_feat_name
    if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        eval_ntype = config.eval_target_ntype \
            if config.eval_target_ntype is not None else config.target_ntype
        val_idxs = train_data.get_node_val_set(eval_ntype)
        fanout = config.eval_fanout if config.use_mini_batch_infer else []
        if len(val_idxs) > 0:
            # TODO(xiangsx): Support construct feat
            return GSgnnNodeDataLoader(train_data,
                                       val_idxs,
                                       fanout=fanout,
                                       batch_size=task_config.eval_batch_size,
                                       train_task=False,
                                       node_feats=node_feats,
                                       label_field=task_config.label_field)
    elif task.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
        val_idxs = train_data.get_edge_val_set(config.target_etype)
        fanout = config.eval_fanout if config.use_mini_batch_infer else []
        if len(val_idxs) > 0:
            # TODO(xiangsx): Support construct feat
            return GSgnnEdgeDataLoader(train_data,
                                       val_idxs,
                                       fanout=fanout,
                                       batch_size=task_config.eval_batch_size,
                                       node_feats=node_feats,
                                       label_field=task_config.label_field,
                                       decoder_edge_feats=task_config.decoder_edge_feat,
                                       train_task=False,
                                       reverse_edge_types_map=task_config.reverse_edge_types_map,
                                       remove_target_edge_type=task_config.remove_target_edge_type)
    elif task.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
        val_idxs = train_data.get_edge_val_set(config.eval_etype)
        dataloader_cls = gs.get_lp_eval_sampler(config)
        if len(val_idxs) > 0:
            # TODO(xiangsx): Support construct feat
            if config.eval_etypes_negative_dstnode is not None:
                return dataloader_cls(train_data, val_idxs,
                    task_config.eval_batch_size,
                    fixed_edge_dst_negative_field=task_config.eval_etypes_negative_dstnode,
                    fanout=task_config.eval_fanout,
                    fixed_test_size=task_config.fixed_test_size,
                    node_feats=node_feats,
                    pos_graph_edge_feats=task_config.lp_edge_weight_for_loss)
            else:
                return dataloader_cls(train_data, val_idxs,
                    task_config.eval_batch_size,
                    task_config.num_negative_edges_eval,
                    fanout=task_config.eval_fanout,
                    fixed_test_size=task_config.fixed_test_size,
                    node_feats=node_feats,
                    pos_graph_edge_feats=task_config.lp_edge_weight_for_loss)

    return None

def create_task_test_dataloader(task, config, task_config, train_data):
    """
    """
    # All tasks share the same input encoder, so the node feats must be same.
    node_feats = config.node_feat_name
    if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        eval_ntype = config.eval_target_ntype \
            if config.eval_target_ntype is not None else config.target_ntype
        test_idxs = train_data.get_node_test_set(eval_ntype)
        fanout = config.eval_fanout if config.use_mini_batch_infer else []
        if len(test_idxs) > 0:
            # TODO(xiangsx): Support construct feat
            return GSgnnNodeDataLoader(train_data,
                                       test_idxs,
                                       fanout=fanout,
                                       batch_size=task_config.eval_batch_size,
                                       train_task=False,
                                       node_feats=node_feats,
                                       label_field=task_config.label_field)

    elif task.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
        test_idxs = train_data.get_edge_test_set(config.target_etype)
        fanout = config.eval_fanout if config.use_mini_batch_infer else []
        if len(test_idxs) > 0:
            # TODO(xiangsx): Support construct feat
            return GSgnnEdgeDataLoader(train_data,
                                       test_idxs,
                                       fanout=fanout,
                                       batch_size=task_config.eval_batch_size,
                                       node_feats=node_feats,
                                       label_field=task_config.label_field,
                                       decoder_edge_feats=task_config.decoder_edge_feat,
                                       train_task=False,
                                       reverse_edge_types_map=task_config.reverse_edge_types_map,
                                       remove_target_edge_type=task_config.remove_target_edge_type)
    elif task.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
        test_idxs = train_data.get_edge_test_set(config.eval_etype)
        dataloader_cls = gs.get_lp_eval_sampler(config)
        if len(test_idxs) > 0:
            # TODO(xiangsx): Support construct feat
            if config.eval_etypes_negative_dstnode is not None:
                return dataloader_cls(train_data, test_idxs,
                    task_config.eval_batch_size,
                    fixed_edge_dst_negative_field=task_config.eval_etypes_negative_dstnode,
                    fanout=task_config.eval_fanout,
                    fixed_test_size=task_config.fixed_test_size,
                    node_feats=node_feats,
                    pos_graph_edge_feats=task_config.lp_edge_weight_for_loss)
            else:
                return dataloader_cls(train_data, test_idxs,
                    task_config.eval_batch_size,
                    task_config.num_negative_edges_eval,
                    task_config.eval_fanout,
                    fixed_test_size=task_config.fixed_test_size,
                    node_feats=node_feats,
                    pos_graph_edge_feats=task_config.lp_edge_weight_for_loss)
    return None

def create_task_decoder(task, g, decoder_input_dim, train_task):
    if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        return gs.create_builtin_node_decoder(decoder_input_dim, task.task_config, train_task)
    elif task.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
        return gs.create_builtin_edge_decoder(g, decoder_input_dim, task.task_config, train_task)
    elif task.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
        return gs.create_builtin_lp_decoder(g, decoder_input_dim, task.task_config, train_task)

    return None, None

def create_evaluator(task, config):
    if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION]:
        multilabel = config.multilabel[config.eval_target_ntype] \
            if isinstance(config.multilabel, dict) else config.multilabel
        return GSgnnClassificationEvaluator(config.eval_frequency,
                                            config.eval_metric,
                                            multilabel,
                                            config.use_early_stop,
                                            config.early_stop_burnin_rounds,
                                            config.early_stop_rounds,
                                            config.early_stop_strategy)

    elif task.task_type in [BUILTIN_TASK_NODE_REGRESSION]:
        return GSgnnRegressionEvaluator(config.eval_frequency,
                                        config.eval_metric,
                                        config.use_early_stop,
                                        config.early_stop_burnin_rounds,
                                        config.early_stop_rounds,
                                        config.early_stop_strategy)
    elif task.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION]:
        return GSgnnClassificationEvaluator(config.eval_frequency,
                                            config.eval_metric,
                                            config.multilabel,
                                            config.use_early_stop,
                                            config.early_stop_burnin_rounds,
                                            config.early_stop_rounds,
                                            config.early_stop_strategy)

    elif task.task_type in [BUILTIN_TASK_EDGE_REGRESSION]:
        return GSgnnRegressionEvaluator(config.eval_frequency,
                                        config.eval_metric,
                                        config.use_early_stop,
                                        config.early_stop_burnin_rounds,
                                        config.early_stop_rounds,
                                        config.early_stop_strategy)
    elif task.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
        assert len(config.eval_metric) == 1, \
        "GraphStorm doees not support computing multiple metrics at the same time."
        if config.report_eval_per_type:
            return GSgnnPerEtypeMrrLPEvaluator(eval_frequency=config.eval_frequency,
                                            major_etype=config.model_select_etype,
                                            use_early_stop=config.use_early_stop,
                                            early_stop_burnin_rounds=config.early_stop_burnin_rounds,
                                            early_stop_rounds=config.early_stop_rounds,
                                            early_stop_strategy=config.early_stop_strategy)
        else:
            return GSgnnMrrLPEvaluator(eval_frequency=config.eval_frequency,
                                    use_early_stop=config.use_early_stop,
                                    early_stop_burnin_rounds=config.early_stop_burnin_rounds,
                                    early_stop_rounds=config.early_stop_rounds,
                                    early_stop_strategy=config.early_stop_strategy)
    return None

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
    model = GSgnnMultiTaskSharedEncoderModel(config.alpha_l2norm)
    gs.gsf.set_encoder(model, train_data.g, config, train_task=True)

    tasks = config.multi_tasks
    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []
    task_evaluators = {}
    encoder_out_dims = model.gnn_encoder.out_dims \
        if model.gnn_encoder is not None \
            else model.node_input_encoder.out_dims
    for task in tasks:
        task_config = task.task_config
        train_loader = create_task_train_dataloader(task, config, task_config, train_data)
        val_loader = create_task_val_dataloader(task, config, task_config, train_data)
        test_loader = create_task_test_dataloader(task, config, task_config, train_data)
        train_dataloaders.append((task, train_loader))
        val_dataloaders.append((task, val_loader))
        test_dataloaders.append((task, test_loader))
        decoder, loss_func = create_task_decoder(task, train_data.g, encoder_out_dims, train_task=True)
        model.add_task(task.task_id, task.task_type, decoder, loss_func, task_config.weight)
        if not config.no_validation:
            if val_loader is None:
                logging.warning("The training data do not have validation set.")
            if test_loader is None:
                logging.warning("The training data do not have test set.")
            task_evaluators[task.task_id] = \
                create_evaluator(task, config)


    train_dataloader = GSgnnMultiTaskDataLoader(train_dataloaders)
    val_dataloader = GSgnnMultiTaskDataLoader(val_dataloaders)
    test_dataloader = GSgnnMultiTaskDataLoader(test_dataloaders)

    if not config.no_validation:
        evaluator = GSgnnMultiTaskEvaluator(config.eval_frequency,
                                            task_evaluators,
                                            use_early_stop=config.use_early_stop)
        trainer.setup_evaluator(evaluator)

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

    tracker = gs.create_builtin_task_tracker(config)
    if gs.get_rank() == 0:
        tracker.log_params(config.__dict__)
    trainer.setup_task_tracker(tracker)

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
        model = GSgnnMultiTaskSharedEncoderModel(config.alpha_l2norm)
        gs.set_encoder(model, train_data.g, config, train_task=True)
        best_model_path = trainer.get_best_model_path()
        # TODO(zhengda) the model path has to be in a shared filesystem.
        model.restore_model(best_model_path)
        model = model.to(get_device())
        # Preparing input layer for training or inference.
        # The input layer can pre-compute node features in the preparing step if needed.
        # For example pre-compute all BERT embeddings
        model.prepare_input_encoder(train_data)

        embeddings = do_full_graph_inference(model, train_data, fanout=config.eval_fanout,
                                             edge_mask="train_mask", task_tracker=tracker)

        save_full_node_embeddings(
            train_data.g,
            config.save_embed_path,
            embeddings,
            node_id_mapping_file=config.node_id_mapping_file,
            save_embed_format=config.save_embed_format)

def generate_parser():
    """ Generate an argument parser
    """
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    arg_parser = generate_parser()

    # Ignore unknown args to make script more robust to input arguments
    gs_args, _ = arg_parser.parse_known_args()
    main(gs_args)


