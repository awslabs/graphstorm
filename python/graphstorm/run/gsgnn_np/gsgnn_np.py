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

    GSgnn node prediction.
"""

import os

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.trainer import GSgnnNodePredictionTrainer
from graphstorm.trainer import GLEMNodePredictionTrainer
from graphstorm.dataloading import GSgnnNodeTrainData, GSgnnNodeDataLoader,\
    GSgnnNodeSemiSupDataLoader
from graphstorm.eval import GSgnnAccEvaluator
from graphstorm.eval import GSgnnRegressionEvaluator
from graphstorm.model.utils import save_full_node_embeddings
from graphstorm.model import do_full_graph_inference
from graphstorm.utils import rt_profiler, sys_tracker, setup_device, use_wholegraph
from graphstorm.utils import get_lm_ntypes

def get_evaluator(config):
    """ Get evaluator class
    """
    if config.task_type == "node_classification":
        multilabel = config.multilabel[config.eval_target_ntype] \
            if isinstance(config.multilabel, dict) else config.multilabel
        return GSgnnAccEvaluator(config.eval_frequency,
                                 config.eval_metric,
                                 multilabel,
                                 config.use_early_stop,
                                 config.early_stop_burnin_rounds,
                                 config.early_stop_rounds,
                                 config.early_stop_strategy)
    elif config.task_type == "node_regression":
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

    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  use_wholegraph=use_wholegraph(config.part_config))
    rt_profiler.init(config.profile_path, rank=gs.get_rank())
    sys_tracker.init(config.verbose, rank=gs.get_rank())
    device = setup_device(config.local_rank)
    train_data = GSgnnNodeTrainData(config.graph_name,
                                    config.part_config,
                                    train_ntypes=config.target_ntype,
                                    eval_ntypes=config.eval_target_ntype,
                                    node_feat_field=config.node_feat_name,
                                    label_field=config.label_field,
                                    lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))
    model = gs.create_builtin_node_gnn_model(train_data.g, config, train_task=True)

    if config.training_method["name"] == "glem":
        trainer_class = GLEMNodePredictionTrainer
    elif config.training_method["name"] == "default":
        trainer_class = GSgnnNodePredictionTrainer
    trainer = trainer_class(model, topk_model_to_save=config.topk_model_to_save)
    if config.restore_model_path is not None:
        trainer.restore_model(model_path=config.restore_model_path,
                              model_layer_to_load=config.restore_model_layers)
    trainer.setup_device(device=device)
    if not config.no_validation:
        evaluator = get_evaluator(config)
        trainer.setup_evaluator(evaluator)
        assert len(train_data.val_idxs) > 0, "The training data do not have validation set."
        # TODO(zhengda) we need to compute the size of the entire validation set to make sure
        # we have validation data.
    tracker = gs.create_builtin_task_tracker(config)
    if gs.get_rank() == 0:
        tracker.log_params(config.__dict__)
    trainer.setup_task_tracker(tracker)

    if config.use_pseudolabel:
        # Use nodes not in train_idxs as unlabeled node sets
        unlabeled_idxs = train_data.get_unlabeled_idxs()
        # semi-supervised loader
        dataloader = GSgnnNodeSemiSupDataLoader(train_data, train_data.train_idxs, unlabeled_idxs,
                                                fanout=config.fanout, batch_size=config.batch_size,
                                                device=device, train_task=True,
                                                construct_feat_ntype=config.construct_feat_ntype,
                                                construct_feat_fanout=config.construct_feat_fanout)
    else:
        dataloader = GSgnnNodeDataLoader(train_data, train_data.train_idxs, fanout=config.fanout,
                                         batch_size=config.batch_size,
                                         device=device, train_task=True,
                                         construct_feat_ntype=config.construct_feat_ntype,
                                         construct_feat_fanout=config.construct_feat_fanout)
    # we don't need fanout for full-graph inference
    fanout = config.eval_fanout if config.use_mini_batch_infer else []
    val_dataloader = None
    test_dataloader = None
    if len(train_data.val_idxs) > 0:
        val_dataloader = GSgnnNodeDataLoader(train_data, train_data.val_idxs, fanout=fanout,
                                             batch_size=config.eval_batch_size,
                                             device=device, train_task=False,
                                             construct_feat_ntype=config.construct_feat_ntype,
                                             construct_feat_fanout=config.construct_feat_fanout)
    if len(train_data.test_idxs) > 0:
        test_dataloader = GSgnnNodeDataLoader(train_data, train_data.test_idxs, fanout=fanout,
                                              batch_size=config.eval_batch_size,
                                              device=device, train_task=False,
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
        model = gs.create_builtin_node_gnn_model(train_data.g, config, train_task=False)
        best_model_path = trainer.get_best_model_path()
        # TODO(zhengda) the model path has to be in a shared filesystem.
        model.restore_model(best_model_path)
        # Preparing input layer for training or inference.
        # The input layer can pre-compute node features in the preparing step if needed.
        # For example pre-compute all BERT embeddings
        model.prepare_input_encoder(train_data)
        embeddings = do_full_graph_inference(model, train_data, fanout=config.eval_fanout,
                                             task_tracker=tracker)
        # Only save embeddings of nodes from target ntype(s).
        # Embeddings of nodes from other ntype(s) are meaningless,
        # as they are not trained. Specifically, the model parameters
        # of the weight matrics of the edge types of the last layer GNN
        # targetting these ntype(s) will not receive any gradient from
        # the training loss.
        embeddings = {ntype: embeddings[ntype] for ntype in train_data.train_ntypes}
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
