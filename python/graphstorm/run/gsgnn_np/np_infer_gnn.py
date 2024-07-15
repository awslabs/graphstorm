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

    Inference script for node classification/regression tasks with GNN
"""
import logging

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnNodePredictionInferrer
from graphstorm.eval import GSgnnClassificationEvaluator, GSgnnRegressionEvaluator
from graphstorm.dataloading import GSgnnData, GSgnnNodeDataLoader
from graphstorm.utils import get_device, get_lm_ntypes, use_wholegraph

def get_evaluator(config): # pylint: disable=unused-argument
    """ Get evaluator class
    """
    if config.task_type == "node_regression":
        return GSgnnRegressionEvaluator(config.eval_frequency,
                                        config.eval_metric)
    elif config.task_type == 'node_classification':
        return GSgnnClassificationEvaluator(config.eval_frequency,
                                            config.eval_metric,
                                            config.multilabel)
    else:
        raise AttributeError(config.task_type + ' is not supported.')

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(False)

    use_wg_feats = use_wholegraph(config.part_config)
    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  local_rank=config.local_rank,
                  use_wholegraph=config.use_wholegraph_embed or use_wg_feats)

    infer_data = GSgnnData(config.part_config,
                           node_feat_field=config.node_feat_name,
                           edge_feat_field=config.edge_feat_name,
                           lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))

    model = gs.create_builtin_node_gnn_model(infer_data.g, config, train_task=False)
    model.restore_model(config.restore_model_path,
                        model_layer_to_load=config.restore_model_layers)
    infer = GSgnnNodePredictionInferrer(model)
    infer.setup_device(device=get_device())
    if not config.no_validation:
        infer_idxs = infer_data.get_node_test_set(config.target_ntype)
        evaluator = get_evaluator(config)
        infer.setup_evaluator(evaluator)
        assert len(infer_idxs) > 0, \
            "There is not test data for evaluation. " \
            "You can use --no-validation true to avoid do testing"
    else:
        infer_idxs = infer_data.get_node_infer_set(config.target_ntype)
        assert len(infer_idxs) > 0, \
            f"To do inference on {config.target_ntype} without doing evaluation, " \
            "you should not define test_mask as its node feature. " \
            "GraphStorm will do inference on the whole node set. "
    tracker = gs.create_builtin_task_tracker(config)
    infer.setup_task_tracker(tracker)
    fanout = config.eval_fanout if config.use_mini_batch_infer else []
    dataloader = GSgnnNodeDataLoader(infer_data, infer_idxs, fanout=fanout,
                                     batch_size=config.eval_batch_size,
                                     train_task=False,
                                     node_feats=config.node_feat_name,
                                     label_field=config.label_field,
                                     construct_feat_ntype=config.construct_feat_ntype,
                                     construct_feat_fanout=config.construct_feat_fanout)
    infer.infer(dataloader, save_embed_path=config.save_embed_path,
                save_prediction_path=config.save_prediction_path,
                use_mini_batch_infer=config.use_mini_batch_infer,
                node_id_mapping_file=config.node_id_mapping_file,
                return_proba=config.return_proba,
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
                    "graphstorm.run.gs_node_classification or "
                    "graphstorm.run.gs_node_regression: %s",
                    unknown_args)
    main(gs_args)
