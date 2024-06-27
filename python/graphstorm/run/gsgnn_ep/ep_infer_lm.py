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

    Inference script for edge classification/regression tasks with language
    model as encoder only.
"""
import logging

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnEdgePredictionInferrer
from graphstorm.eval import GSgnnClassificationEvaluator, GSgnnRegressionEvaluator
from graphstorm.dataloading import GSgnnData, GSgnnEdgeDataLoader
from graphstorm.utils import get_device

def get_evaluator(config): # pylint: disable=unused-argument
    """ Get evaluator class
    """
    if config.task_type == "edge_regression":
        return GSgnnRegressionEvaluator(config.eval_frequency,
                                        config.eval_metric)
    elif config.task_type == 'edge_classification':
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

    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  local_rank=config.local_rank)
    # The model only uses language model(s) as its encoder
    # It will not use node or edge features
    # except LM related features.
    infer_data = GSgnnData(config.part_config)
    model = gs.create_builtin_edge_model(infer_data.g, config, train_task=False)
    model.restore_model(config.restore_model_path,
                        model_layer_to_load=config.restore_model_layers)
    infer = GSgnnEdgePredictionInferrer(model)
    infer.setup_device(device=get_device())
    if not config.no_validation:
        target_idxs = infer_data.get_edge_test_set(config.target_etype)
        evaluator = get_evaluator(config)
        infer.setup_evaluator(evaluator)
        assert len(target_idxs) > 0, "There is no test data for evaluation."
    else:
        target_idxs = infer_data.get_edge_infer_set(config.target_etype)
        assert len(target_idxs) > 0, \
            f"To do inference on {config.target_etype} without doing evaluation, " \
            "you should not define test_mask as its edge feature. " \
            "GraphStorm will do inference on the whole edge set. "
    tracker = gs.create_builtin_task_tracker(config)
    infer.setup_task_tracker(tracker)
    dataloader = GSgnnEdgeDataLoader(infer_data, target_idxs, fanout=[],
                                     batch_size=config.eval_batch_size,
                                     node_feats=config.node_feat_name,
                                     label_field=config.label_field,
                                     decoder_edge_feats=config.decoder_edge_feat,
                                     train_task=False,
                                     remove_target_edge_type=False)
    # Preparing input layer for training or inference.
    # The input layer can pre-compute node features in the preparing step if needed.
    # For example pre-compute all BERT embeddings
    model.prepare_input_encoder(infer_data)
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

    # Ignore unknown args to make script more robust to input arguments
    gs_args, unknown_args = arg_parser.parse_known_args()
    logging.warning("Unknown arguments for command "
                    "graphstorm.run.gs_edge_classification or "
                    "graphstorm.run.gs_edge_regression: %s",
                    unknown_args)
    main(gs_args)
