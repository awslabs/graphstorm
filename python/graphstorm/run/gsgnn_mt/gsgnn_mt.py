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

    GSgnn multi-task learning training entry point.
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
                               BUILTIN_TASK_LINK_PREDICTION,
                               BUILTIN_TASK_RECONSTRUCT_NODE_FEAT)
from graphstorm.dataloading import GSgnnData
from graphstorm.dataloading import (GSgnnNodeDataLoader,
                                    GSgnnEdgeDataLoader,
                                    GSgnnMultiTaskDataLoader)
from graphstorm.eval import GSgnnMultiTaskEvaluator
from graphstorm.model.multitask_gnn import GSgnnMultiTaskSharedEncoderModel
from graphstorm.trainer import GSgnnMultiTaskLearningTrainer
from graphstorm.model.utils import save_full_node_embeddings
from graphstorm.model import do_full_graph_inference

from graphstorm.utils import rt_profiler, sys_tracker, get_device, use_wholegraph
from graphstorm.utils import get_lm_ntypes

def create_task_train_dataloader(task, config, train_data):
    """ Create task specific dataloader for training tasks

    Parameters
    ----------
    task: TaskInfo
        Task info.
    config: GSConfig
        Training config.
    train_data: GSgnnData
        Training data.

    Return
    ------
    Task dataloader
    """
    task_config = task.task_config
    # All tasks share the same GNN model, so the fanout should be the global fanout
    fanout = config.fanout
    # All tasks share the same input encoder, so the node feats must be same.
    node_feats = config.node_feat_name

    assert task_config.train_mask is not None, \
        "For multi-task learning, train_mask field name " \
        "must be provided through mask_fields, but get None"
    logging.info("Create dataloader for %s", task.task_id)
    if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        train_idxs = train_data.get_node_train_set(
            task_config.target_ntype,
            mask=task_config.train_mask)
        # TODO(xiangsx): Support construct feat
        return GSgnnNodeDataLoader(train_data,
                                   train_idxs,
                                   fanout=fanout,
                                   batch_size=task_config.batch_size,
                                   train_task=True,
                                   node_feats=node_feats,
                                   label_field=task_config.label_field)
    elif task.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
        train_idxs = train_data.get_edge_train_set(
            task_config.target_etype,
            mask=task_config.train_mask)
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
                                   remove_target_edge_type=task_config.remove_target_edge_type)
    elif task.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
        train_idxs = train_data.get_edge_train_set(
            task_config.train_etype,
            mask=task_config.train_mask)
        dataloader_cls = gs.get_builtin_lp_train_dataloader_class(task_config)
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
    elif task.task_type in [BUILTIN_TASK_RECONSTRUCT_NODE_FEAT]:
        train_idxs = train_data.get_node_train_set(
            task_config.target_ntype,
            mask=task_config.train_mask)
        # TODO(xiangsx): Support construct feat
        return GSgnnNodeDataLoader(train_data,
                                   train_idxs,
                                   fanout=fanout,
                                   batch_size=task_config.batch_size,
                                   train_task=True,
                                   node_feats=node_feats,
                                   label_field=task_config.reconstruct_nfeat_name)

    return None

def create_task_val_dataloader(task, config, train_data):
    """ Create task specific validation dataloader.

    Parameters
    ----------
    task: TaskInfo
        Task info.
    config: GSConfig
        Training config.
    train_data: GSgnnData
        Training data.

    Return
    ------
    Task dataloader
    """
    task_config = task.task_config
    if task_config.val_mask is None:
        # There is no validation mask
        return None
    # All tasks share the same input encoder, so the node feats must be same.
    node_feats = config.node_feat_name
    # All tasks share the same GNN model, so the fanout should be the global fanout
    fanout = config.eval_fanout if task_config.use_mini_batch_infer else []
    if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        eval_ntype = task_config.eval_target_ntype \
            if task_config.eval_target_ntype is not None \
            else task_config.target_ntype
        val_idxs = train_data.get_node_val_set(eval_ntype, mask=task_config.val_mask)
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
        val_idxs = train_data.get_edge_val_set(task_config.target_etype, mask=task_config.val_mask)
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
        val_idxs = train_data.get_edge_val_set(task_config.eval_etype, mask=task_config.val_mask)
        dataloader_cls = gs.get_builtin_lp_eval_dataloader_class(task_config)
        if len(val_idxs) > 0:
            # TODO(xiangsx): Support construct feat
            if task_config.eval_etypes_negative_dstnode is not None:
                return dataloader_cls(train_data, val_idxs,
                    task_config.eval_batch_size,
                    fixed_edge_dst_negative_field=task_config.eval_etypes_negative_dstnode,
                    fanout=fanout,
                    fixed_test_size=task_config.fixed_test_size,
                    node_feats=node_feats,
                    pos_graph_edge_feats=task_config.lp_edge_weight_for_loss)
            else:
                return dataloader_cls(train_data, val_idxs,
                    task_config.eval_batch_size,
                    task_config.num_negative_edges_eval,
                    fanout=fanout,
                    fixed_test_size=task_config.fixed_test_size,
                    node_feats=node_feats,
                    pos_graph_edge_feats=task_config.lp_edge_weight_for_loss)
    elif task.task_type in [BUILTIN_TASK_RECONSTRUCT_NODE_FEAT]:
        eval_ntype = task_config.eval_target_ntype \
            if task_config.eval_target_ntype is not None \
            else task_config.target_ntype
        val_idxs = train_data.get_node_val_set(eval_ntype, mask=task_config.val_mask)
        if len(val_idxs) > 0:
            # TODO(xiangsx): Support construct feat
            return GSgnnNodeDataLoader(train_data,
                                       val_idxs,
                                       fanout=fanout,
                                       batch_size=task_config.eval_batch_size,
                                       train_task=False,
                                       node_feats=node_feats,
                                       label_field=task_config.reconstruct_nfeat_name)

    return None

def create_task_test_dataloader(task, config, train_data):
    """ Create task specific test dataloader.

    Parameters
    ----------
    task: TaskInfo
        Task info.
    config: GSConfig
        Training config.
    train_data: GSgnnData
        Training data.

    Return
    ------
    Task dataloader
    """
    task_config = task.task_config
    if task_config.test_mask is None:
        # There is no validation mask
        return None
    # All tasks share the same input encoder, so the node feats must be same.
    node_feats = config.node_feat_name
    # All tasks share the same GNN model, so the fanout should be the global fanout
    fanout = config.eval_fanout if task_config.use_mini_batch_infer else []

    if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        eval_ntype = task_config.eval_target_ntype \
            if task_config.eval_target_ntype is not None \
            else task_config.target_ntype
        test_idxs = train_data.get_node_test_set(eval_ntype, mask=task_config.test_mask)
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
        test_idxs = train_data.get_edge_test_set(
            task_config.target_etype,
            mask=task_config.test_mask)
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
        test_idxs = train_data.get_edge_test_set(task_config.eval_etype, mask=task_config.val_mask)
        dataloader_cls = gs.get_builtin_lp_eval_dataloader_class(task_config)
        if len(test_idxs) > 0:
            # TODO(xiangsx): Support construct feat
            if task_config.eval_etypes_negative_dstnode is not None:
                return dataloader_cls(train_data, test_idxs,
                    task_config.eval_batch_size,
                    fixed_edge_dst_negative_field=task_config.eval_etypes_negative_dstnode,
                    fanout=fanout,
                    fixed_test_size=task_config.fixed_test_size,
                    node_feats=node_feats,
                    pos_graph_edge_feats=task_config.lp_edge_weight_for_loss)
            else:
                return dataloader_cls(train_data, test_idxs,
                    task_config.eval_batch_size,
                    task_config.num_negative_edges_eval,
                    fanout=fanout,
                    fixed_test_size=task_config.fixed_test_size,
                    node_feats=node_feats,
                    pos_graph_edge_feats=task_config.lp_edge_weight_for_loss)
    elif task.task_type in [BUILTIN_TASK_RECONSTRUCT_NODE_FEAT]:
        eval_ntype = task_config.eval_target_ntype \
            if task_config.eval_target_ntype is not None \
            else task_config.target_ntype
        test_idxs = train_data.get_node_test_set(eval_ntype, mask=task_config.test_mask)
        if len(test_idxs) > 0:
            # TODO(xiangsx): Support construct feat
            return GSgnnNodeDataLoader(train_data,
                                       test_idxs,
                                       fanout=fanout,
                                       batch_size=task_config.eval_batch_size,
                                       train_task=False,
                                       node_feats=node_feats,
                                       label_field=task_config.reconstruct_nfeat_name)
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
    assert tasks is not None, \
        "The multi_task_learning configure block should not be empty."
    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []
    task_evaluators = {}
    encoder_out_dims = model.gnn_encoder.out_dims \
        if model.gnn_encoder is not None \
            else model.node_input_encoder.out_dims
    for task in tasks:
        train_loader = create_task_train_dataloader(task, config, train_data)
        val_loader = create_task_val_dataloader(task, config, train_data)
        test_loader = create_task_test_dataloader(task, config, train_data)
        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)
        test_dataloaders.append(test_loader)
        decoder, loss_func = gs.create_task_decoder(task,
                                                    train_data.g,
                                                    encoder_out_dims,
                                                    train_task=True)
        # For link prediction, lp_embed_normalizer may be used
        # TODO(xiangsx): add embed normalizer for other task types
        # in the future.
        node_embed_norm_method = task.task_config.lp_embed_normalizer \
            if task.task_type in [BUILTIN_TASK_LINK_PREDICTION] \
            else None
        model.add_task(task.task_id,
                       task.task_type,
                       decoder,
                       loss_func,
                       embed_norm_method=node_embed_norm_method)
        if not config.no_validation:
            if val_loader is None:
                logging.warning("The training data do not have validation set.")
            if test_loader is None:
                logging.warning("The training data do not have test set.")

            if val_loader is None and test_loader is None:
                logging.warning("Task %s does not have validation and test sets.", task.task_id)
            else:
                task_evaluators[task.task_id] = \
                    gs.create_evaluator(task)

    train_dataloader = GSgnnMultiTaskDataLoader(train_data, tasks, train_dataloaders)
    val_dataloader = GSgnnMultiTaskDataLoader(train_data, tasks, val_dataloaders)
    test_dataloader = GSgnnMultiTaskDataLoader(train_data, tasks, test_dataloaders)

    model.init_optimizer(lr=config.lr,
                         sparse_optimizer_lr=config.sparse_optimizer_lr,
                         weight_decay=config.wd_l2norm,
                         lm_lr=config.lm_tune_lr)
    trainer = GSgnnMultiTaskLearningTrainer(model, topk_model_to_save=config.topk_model_to_save)
    if not config.no_validation:
        evaluator = GSgnnMultiTaskEvaluator(config.eval_frequency,
                                            task_evaluators,
                                            use_early_stop=config.use_early_stop)
        trainer.setup_evaluator(evaluator)
    if config.restore_model_path is not None:
        trainer.restore_model(model_path=config.restore_model_path,
                              model_layer_to_load=config.restore_model_layers)
    trainer.setup_device(device=get_device())

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
        gs.gsf.set_encoder(model, train_data.g, config, train_task=True)

        for task in tasks:
            decoder, loss_func = gs.create_task_decoder(task,
                                                        train_data.g,
                                                        encoder_out_dims,
                                                        train_task=True)
            node_embed_norm_method = task.task_config.lp_embed_normalizer \
                if task.task_type in [BUILTIN_TASK_LINK_PREDICTION] \
                else None
            model.add_task(task.task_id,
                           task.task_type,
                           decoder,
                           loss_func,
                           embed_norm_method=node_embed_norm_method)
        best_model_path = trainer.get_best_model_path()
        # TODO(zhengda) the model path has to be in a shared filesystem.
        model.restore_model(best_model_path)
        model = model.to(get_device())
        # Preparing input layer for training or inference.
        # The input layer can pre-compute node features in the preparing step if needed.
        # For example pre-compute all BERT embeddings
        model.prepare_input_encoder(train_data)

        embeddings = do_full_graph_inference(model, train_data, fanout=config.eval_fanout,
                                             task_tracker=tracker)

        # Save the original embs first
        save_full_node_embeddings(
            train_data.g,
            config.save_embed_path,
            embeddings,
            node_id_mapping_file=config.node_id_mapping_file,
            save_embed_format=config.save_embed_format)

        node_norm_methods = model.node_embed_norm_methods
        # save normalized embeddings
        for task_id, norm_method in node_norm_methods.items():
            if norm_method is None:
                continue
            normed_embs = model.normalize_task_node_embs(task_id, embeddings, inplace=False)
            save_embed_path = os.path.join(config.save_embed_path, task_id)
            save_full_node_embeddings(
                train_data.g,
                save_embed_path,
                normed_embs,
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
    gs_args, unknown_args = arg_parser.parse_known_args()
    logging.warning("Unknown arguments for command "
                    "graphstorm.run.gs_multi_task_learning: %s",
                    unknown_args)
    main(gs_args)
