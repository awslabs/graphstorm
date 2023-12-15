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

    Inference script for edge classification/regression tasks with GNN
"""

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnEdgePredictionInferrer
from graphstorm.eval import GSgnnAccEvaluator, GSgnnRegressionEvaluator
from graphstorm.dataloading import GSgnnEdgeInferData, GSgnnEdgeDataLoader
from graphstorm.utils import setup_device, get_lm_ntypes

def get_evaluator(config): # pylint: disable=unused-argument
    """ Get evaluator class
    """
    if config.task_type == "edge_regression":
        return GSgnnRegressionEvaluator(config.eval_frequency,
                                        config.eval_metric)
    elif config.task_type == 'edge_classification':
        return GSgnnAccEvaluator(config.eval_frequency,
                                 config.eval_metric,
                                 config.multilabel)
    else:
        raise AttributeError(config.task_type + ' is not supported.')

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(False)

    gs.initialize(ip_config=config.ip_config, backend=config.backend)
    device = setup_device(config.local_rank)

    infer_data = GSgnnEdgeInferData(config.graph_name,
                                    config.part_config,
                                    eval_etypes=config.target_etype,
                                    node_feat_field=config.node_feat_name,
                                    label_field=config.label_field,
                                    decoder_edge_feat=config.decoder_edge_feat,
                                    lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))
    model = gs.create_builtin_edge_gnn_model(infer_data.g, config, train_task=False)
    model.restore_model(config.restore_model_path,
                        model_layer_to_load=config.restore_model_layers)
    # TODO(zhengda) we should use a different way to get rank.
    infer = GSgnnEdgePredictionInferrer(model)
    infer.setup_device(device=device)
    if not config.no_validation:
        evaluator = get_evaluator(config)
        infer.setup_evaluator(evaluator)
        assert len(infer_data.test_idxs) > 0, \
            "There is not test data for evaluation. " \
            "You can use --no-validation true to avoid do testing"
        target_idxs = infer_data.test_idxs
    else:
        assert len(infer_data.infer_idxs) > 0, \
            f"To do inference on {config.target_etype} without doing evaluation, " \
            "you should not define test_mask as its edge feature. " \
            "GraphStorm will do inference on the whole edge set. "
        target_idxs = infer_data.infer_idxs
    tracker = gs.create_builtin_task_tracker(config)
    infer.setup_task_tracker(tracker)
    fanout = config.eval_fanout if config.use_mini_batch_infer else []
    dataloader = GSgnnEdgeDataLoader(infer_data, target_idxs, fanout=fanout,
                                     batch_size=config.eval_batch_size,
                                     device=device, train_task=False,
                                     reverse_edge_types_map=config.reverse_edge_types_map,
                                     remove_target_edge_type=config.remove_target_edge_type,
                                     construct_feat_ntype=config.construct_feat_ntype,
                                     construct_feat_fanout=config.construct_feat_fanout)
    infer.infer(dataloader, save_embed_path=config.save_embed_path,
                save_prediction_path=config.save_prediction_path,
                use_mini_batch_infer=config.use_mini_batch_infer,
                node_id_mapping_file=config.node_id_mapping_file,
                edge_id_mapping_file=config.edge_id_mapping_file,
                return_proba=config.return_proba,
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
