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

    Inference script for link prediction tasks with language model as
    encoder only.
"""
import logging


import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnLinkPredictionInferrer
from graphstorm.eval import GSgnnMrrLPEvaluator, GSgnnHitsLPEvaluator
from graphstorm.dataloading import GSgnnData
from graphstorm.dataloading import (GSgnnLinkPredictionTestDataLoader,
                                    GSgnnLinkPredictionJointTestDataLoader,
                                    GSgnnLinkPredictionPredefinedTestDataLoader)
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER
from graphstorm.utils import get_device
from graphstorm.eval.eval_func import SUPPORTED_HIT_AT_METRICS

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
    model = gs.create_builtin_lp_model(infer_data.g, config, train_task=False)
    model.restore_model(config.restore_model_path,
                        model_layer_to_load=config.restore_model_layers)
    infer = GSgnnLinkPredictionInferrer(model)
    infer.setup_device(device=get_device())
    # TODO: to create a generic evaluator for LP tasks
    if len(config.eval_metric) > 1 and ("mrr" in config.eval_metric) \
            and any((x.startswith(SUPPORTED_HIT_AT_METRICS) for x in config.eval_metric)):
        logging.warning("GraphStorm does not support computing MRR and Hit@K metrics at the "
                        "same time. If both metrics are given, only 'mrr' is returned.")
    if not config.no_validation:
        infer_idxs = infer_data.get_edge_test_set(config.eval_etype)
        if len(config.eval_metric) == 0 or 'mrr' in config.eval_metric:
            infer.setup_evaluator(
                GSgnnMrrLPEvaluator(config.eval_frequency))
        else:
            infer.setup_evaluator(GSgnnHitsLPEvaluator(
                config.eval_frequency, eval_metric_list=config.eval_metric))
        assert len(infer_idxs) > 0, "There is not test data for evaluation."
    else:
        infer_idxs = infer_data.get_edge_infer_set(config.eval_etype)

    tracker = gs.create_builtin_task_tracker(config)
    infer.setup_task_tracker(tracker)
    # We only support full-graph inference for now.
    if config.eval_etypes_negative_dstnode is not None:
        # The negatives used in evaluation is fixed.
        dataloader = GSgnnLinkPredictionPredefinedTestDataLoader(
            infer_data, infer_idxs,
            batch_size=config.eval_batch_size,
            fixed_edge_dst_negative_field=config.eval_etypes_negative_dstnode,
            node_feats=config.node_feat_name)
    else:
        if config.eval_negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
            test_dataloader_cls = GSgnnLinkPredictionTestDataLoader
        elif config.eval_negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER:
            test_dataloader_cls = GSgnnLinkPredictionJointTestDataLoader
        else:
            raise ValueError('Unknown test negative sampler.'
                'Supported test negative samplers include '
                f'[{BUILTIN_LP_UNIFORM_NEG_SAMPLER}, {BUILTIN_LP_JOINT_NEG_SAMPLER}]')

        dataloader = test_dataloader_cls(infer_data, infer_idxs,
            batch_size=config.eval_batch_size,
            num_negative_edges=config.num_negative_edges_eval,
            node_feats=config.node_feat_name)
    # Preparing input layer for training or inference.
    # The input layer can pre-compute node features in the preparing step if needed.
    # For example pre-compute all BERT embeddings
    model.prepare_input_encoder(infer_data)
    infer.infer(infer_data, dataloader,
                save_embed_path=config.save_embed_path,
                edge_mask_for_gnn_embeddings=None, # LM infer does not use GNN
                use_mini_batch_infer=config.use_mini_batch_infer,
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
                    "graphstorm.run.gs_link_prediction: %s",
                    unknown_args)
    main(gs_args)
