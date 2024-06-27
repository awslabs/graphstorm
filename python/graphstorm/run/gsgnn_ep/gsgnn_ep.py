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

    GSgnn edge prediction.
"""

import os
import logging

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.trainer import GSgnnEdgePredictionTrainer
from graphstorm.dataloading import GSgnnData, GSgnnEdgeDataLoader
from graphstorm.eval import GSgnnClassificationEvaluator
from graphstorm.eval import GSgnnRegressionEvaluator
from graphstorm.model.utils import save_full_node_embeddings
from graphstorm.model import do_full_graph_inference
from graphstorm.utils import rt_profiler, sys_tracker, get_device, use_wholegraph
from graphstorm.utils import get_lm_ntypes

def get_evaluator(config):
    """ Get evaluator class
    """
    if config.task_type == "edge_classification":
        return GSgnnClassificationEvaluator(config.eval_frequency,
                                            config.eval_metric,
                                            config.multilabel,
                                            config.use_early_stop,
                                            config.early_stop_burnin_rounds,
                                            config.early_stop_rounds,
                                            config.early_stop_strategy)
    elif config.task_type == "edge_regression":
        return GSgnnRegressionEvaluator(config.eval_frequency,
                                        config.eval_metric,
                                        config.use_early_stop,
                                        config.early_stop_burnin_rounds,
                                        config.early_stop_rounds,
                                        config.early_stop_strategy)
    else:
        raise ValueError("Unknown task type")

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
    # edge predict only handle edge feature in decoder
    train_data = GSgnnData(config.part_config,
                           node_feat_field=config.node_feat_name,
                           edge_feat_field=config.edge_feat_name,
                           lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))
    model = gs.create_builtin_edge_gnn_model(train_data.g, config, train_task=True)
    trainer = GSgnnEdgePredictionTrainer(model, topk_model_to_save=config.topk_model_to_save)
    if config.restore_model_path is not None:
        trainer.restore_model(model_path=config.restore_model_path,
                              model_layer_to_load=config.restore_model_layers)
    trainer.setup_device(device=get_device())
    if not config.no_validation:
        # TODO(zhengda) we need to refactor the evaluator.
        evaluator = get_evaluator(config)
        trainer.setup_evaluator(evaluator)
        val_idxs = train_data.get_edge_val_set(config.target_etype)
        assert len(val_idxs) > 0, "The training data do not have validation set."
        # TODO(zhengda) we need to compute the size of the entire validation set to make sure
        # we have validation data.
    tracker = gs.create_builtin_task_tracker(config)
    if gs.get_rank() == 0:
        tracker.log_params(config.__dict__)
    trainer.setup_task_tracker(tracker)
    train_idxs = train_data.get_edge_train_set(config.target_etype)
    dataloader = GSgnnEdgeDataLoader(train_data, train_idxs, fanout=config.fanout,
                                     batch_size=config.batch_size,
                                     node_feats=config.node_feat_name,
                                     label_field=config.label_field,
                                     decoder_edge_feats=config.decoder_edge_feat,
                                     train_task=True,
                                     reverse_edge_types_map=config.reverse_edge_types_map,
                                     remove_target_edge_type=config.remove_target_edge_type,
                                     exclude_training_targets=config.exclude_training_targets,
                                     construct_feat_ntype=config.construct_feat_ntype,
                                     construct_feat_fanout=config.construct_feat_fanout)
    val_dataloader = None
    test_dataloader = None
    # we don't need fanout for full-graph inference
    fanout = config.eval_fanout if config.use_mini_batch_infer else []
    val_idxs = train_data.get_edge_val_set(config.target_etype)
    test_idxs = train_data.get_edge_test_set(config.target_etype)
    if len(val_idxs) > 0:
        val_dataloader = GSgnnEdgeDataLoader(train_data, val_idxs, fanout=fanout,
            batch_size=config.eval_batch_size,
            node_feats=config.node_feat_name,
            label_field=config.label_field,
            decoder_edge_feats=config.decoder_edge_feat,
            train_task=False,
            reverse_edge_types_map=config.reverse_edge_types_map,
            remove_target_edge_type=config.remove_target_edge_type,
            construct_feat_ntype=config.construct_feat_ntype,
            construct_feat_fanout=config.construct_feat_fanout)
    if len(test_idxs) > 0:
        test_dataloader = GSgnnEdgeDataLoader(train_data, test_idxs, fanout=fanout,
            batch_size=config.eval_batch_size,
            node_feats=config.node_feat_name,
            label_field=config.label_field,
            decoder_edge_feats=config.decoder_edge_feat,
            train_task=False,
            reverse_edge_types_map=config.reverse_edge_types_map,
            remove_target_edge_type=config.remove_target_edge_type,
            construct_feat_ntype=config.construct_feat_ntype,
            construct_feat_fanout=config.construct_feat_fanout)

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
    trainer.fit(train_loader=dataloader, val_loader=val_dataloader,
                test_loader=test_dataloader, num_epochs=config.num_epochs,
                save_model_path=save_model_path,
                use_mini_batch_infer=config.use_mini_batch_infer,
                save_model_frequency=config.save_model_frequency,
                save_perf_results_path=config.save_perf_results_path,
                freeze_input_layer_epochs=config.freeze_lm_encoder_epochs,
                max_grad_norm=config.max_grad_norm,
                grad_norm_type=config.grad_norm_type)

    if config.save_embed_path is not None:
        model = gs.create_builtin_edge_gnn_model(train_data.g, config, train_task=False)
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
        # only save node embeddings of nodes with node types from target_etype
        target_ntypes = set()
        for etype in config.target_etype:
            target_ntypes.add(etype[0])
            target_ntypes.add(etype[2])

        # The order of the ntypes must be sorted
        embs = {ntype: embeddings[ntype] for ntype in sorted(target_ntypes)}
        save_full_node_embeddings(
            train_data.g,
            config.save_embed_path,
            embs,
            node_id_mapping_file=config.node_id_mapping_file,
            save_embed_format=config.save_embed_format)

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
                    "graphstorm.run.gs_edge_classification or "
                    "graphstorm.run.gs_edge_regression: %s",
                    unknown_args)
    main(gs_args)
