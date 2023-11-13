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

    GSgnn pure gpu link prediction.
"""

import os

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.trainer import GSgnnLinkPredictionTrainer
from graphstorm.dataloading import GSgnnLPTrainData
from graphstorm.dataloading import GSgnnLinkPredictionDataLoader
from graphstorm.dataloading import (GSgnnLPJointNegDataLoader,
                                    GSgnnLPLocalUniformNegDataLoader,
                                    GSgnnLPLocalJointNegDataLoader,
                                    GSgnnLPInBatchJointNegDataLoader)
from graphstorm.dataloading import GSgnnAllEtypeLPJointNegDataLoader
from graphstorm.dataloading import GSgnnAllEtypeLinkPredictionDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionTestDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionJointTestDataLoader
from graphstorm.dataloading import (BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                                    BUILTIN_LP_JOINT_NEG_SAMPLER,
                                    BUILTIN_LP_INBATCH_JOINT_NEG_SAMPLER,
                                    BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER,
                                    BUILTIN_LP_LOCALJOINT_NEG_SAMPLER)
from graphstorm.dataloading import BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER
from graphstorm.eval import GSgnnMrrLPEvaluator, GSgnnPerEtypeMrrLPEvaluator
from graphstorm.model.utils import save_full_node_embeddings
from graphstorm.model import do_full_graph_inference
from graphstorm.utils import rt_profiler, sys_tracker, setup_device

def get_evaluator(config, train_data):
    """ Get evaluator according to config

        Parameters
        ----------
        config: GSConfig
            Configuration
        train_data: GSgnnEdgeData
            Training data
    """
    assert len(config.eval_metric) == 1, \
        "GraphStorm doees not support computing multiple metrics at the same time."
    if config.report_eval_per_type:
        return GSgnnPerEtypeMrrLPEvaluator(config.eval_frequency,
                                           train_data,
                                           config.num_negative_edges_eval,
                                           config.lp_decoder_type,
                                           config.model_select_etype,
                                           config.use_early_stop,
                                           config.early_stop_burnin_rounds,
                                           config.early_stop_rounds,
                                           config.early_stop_strategy)
    else:
        return GSgnnMrrLPEvaluator(config.eval_frequency,
                                   train_data,
                                   config.num_negative_edges_eval,
                                   config.lp_decoder_type,
                                   config.use_early_stop,
                                   config.early_stop_burnin_rounds,
                                   config.early_stop_rounds,
                                   config.early_stop_strategy)

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(True)

    gs.initialize(ip_config=config.ip_config, backend=config.backend)
    rt_profiler.init(config.profile_path, rank=gs.get_rank())
    sys_tracker.init(config.verbose, rank=gs.get_rank())
    device = setup_device(config.local_rank)
    train_data = GSgnnLPTrainData(config.graph_name,
                                  config.part_config,
                                  train_etypes=config.train_etype,
                                  eval_etypes=config.eval_etype,
                                  node_feat_field=config.node_feat_name,
                                  pos_graph_feat_field=config.lp_edge_weight_for_loss)
    model = gs.create_builtin_lp_model(train_data.g, config, train_task=True)
    trainer = GSgnnLinkPredictionTrainer(model, topk_model_to_save=config.topk_model_to_save)
    if config.restore_model_path is not None:
        trainer.restore_model(model_path=config.restore_model_path,
                              model_layer_to_load=config.restore_model_layers)
    trainer.setup_device(device=device)
    if not config.no_validation:
        # TODO(zhengda) we need to refactor the evaluator.
        # Currently, we only support mrr
        evaluator = get_evaluator(config, train_data)
        trainer.setup_evaluator(evaluator)
        assert len(train_data.val_idxs) > 0, "The training data do not have validation set."
        # TODO(zhengda) we need to compute the size of the entire validation set to make sure
        # we have validation data.
    tracker = gs.create_builtin_task_tracker(config)
    if gs.get_rank() == 0:
        tracker.log_params(config.__dict__)
    trainer.setup_task_tracker(tracker)

    if config.train_negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
        dataloader_cls = GSgnnLinkPredictionDataLoader
    elif config.train_negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER:
        dataloader_cls = GSgnnLPJointNegDataLoader
    elif config.train_negative_sampler == BUILTIN_LP_INBATCH_JOINT_NEG_SAMPLER:
        dataloader_cls = GSgnnLPInBatchJointNegDataLoader
    elif config.train_negative_sampler == BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER:
        dataloader_cls = GSgnnLPLocalUniformNegDataLoader
    elif config.train_negative_sampler == BUILTIN_LP_LOCALJOINT_NEG_SAMPLER:
        dataloader_cls = GSgnnLPLocalJointNegDataLoader
    elif config.train_negative_sampler == BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER:
        dataloader_cls = GSgnnAllEtypeLinkPredictionDataLoader
    elif config.train_negative_sampler == BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER:
        dataloader_cls = GSgnnAllEtypeLPJointNegDataLoader
    else:
        raise ValueError('Unknown negative sampler')
    dataloader = dataloader_cls(train_data, train_data.train_idxs, [],
                                config.batch_size, config.num_negative_edges, device,
                                train_task=True)

    # TODO(zhengda) let's use full-graph inference for now.
    if config.eval_negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
        test_dataloader_cls = GSgnnLinkPredictionTestDataLoader
    elif config.eval_negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER:
        test_dataloader_cls = GSgnnLinkPredictionJointTestDataLoader
    else:
        raise ValueError('Unknown test negative sampler.'
            'Supported test negative samplers include '
            f'[{BUILTIN_LP_UNIFORM_NEG_SAMPLER}, {BUILTIN_LP_JOINT_NEG_SAMPLER}]')
    val_dataloader = None
    test_dataloader = None
    if len(train_data.val_idxs) > 0:
        val_dataloader = test_dataloader_cls(train_data, train_data.val_idxs,
            config.eval_batch_size, config.num_negative_edges_eval)
    if len(train_data.test_idxs) > 0:
        test_dataloader = test_dataloader_cls(train_data, train_data.test_idxs,
            config.eval_batch_size, config.num_negative_edges_eval)

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
                max_grad_norm=config.max_grad_norm,
                grad_norm_type=config.grad_norm_type)

    if config.save_embed_path is not None:
        model = gs.create_builtin_lp_model(train_data.g, config, train_task=False)
        best_model_path = trainer.get_best_model_path()
        # TODO(zhengda) the model path has to be in a shared filesystem.
        model.restore_model(best_model_path)
        # Preparing input layer for training or inference.
        # The input layer can pre-compute node features in the preparing step if needed.
        # For example pre-compute all BERT embeddings
        model.prepare_input_encoder(train_data)
        # TODO(zhengda) we may not want to only use training edges to generate GNN embeddings.
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
    arg_parser=generate_parser()

    args = arg_parser.parse_args()
    main(args)
