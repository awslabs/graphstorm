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
import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnNodePredictionInferrer
from graphstorm.eval import GSgnnAccEvaluator, GSgnnRegressionEvaluator
from graphstorm.dataloading import GSgnnNodeInferData, GSgnnNodeDataLoader
from graphstorm.utils import setup_device, get_lm_ntypes

def get_evaluator(config): # pylint: disable=unused-argument
    """ Get evaluator class
    """
    if config.task_type == "node_regression":
        return GSgnnRegressionEvaluator(config.eval_frequency,
                                        config.eval_metric)
    elif config.task_type == 'node_classification':
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

    infer_data = GSgnnNodeInferData(config.graph_name,
                                    config.part_config,
                                    eval_ntypes=config.target_ntype,
                                    node_feat_field=config.node_feat_name,
                                    label_field=config.label_field,
                                    lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))

    model = gs.create_builtin_node_gnn_model(infer_data.g, config, train_task=False)
    model.restore_model(config.restore_model_path,
                        model_layer_to_load=config.restore_model_layers)
    infer = GSgnnNodePredictionInferrer(model)
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
            f"To do inference on {config.target_ntype} without doing evaluation, " \
            "you should not define test_mask as its node feature. " \
            "GraphStorm will do inference on the whole node set. "
        target_idxs = infer_data.infer_idxs
    tracker = gs.create_builtin_task_tracker(config)
    infer.setup_task_tracker(tracker)
    fanout = config.eval_fanout if config.use_mini_batch_infer else []
    dataloader = GSgnnNodeDataLoader(infer_data, target_idxs, fanout=fanout,
                                     batch_size=config.eval_batch_size, device=device,
                                     train_task=False,
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

    args = arg_parser.parse_args()
    main(args)
