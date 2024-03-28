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

    Inference script for link prediction tasks with GNN
"""

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnLinkPredictionInferrer
from graphstorm.eval import GSgnnMrrLPEvaluator
from graphstorm.dataloading import GSgnnEdgeInferData
from graphstorm.dataloading import (GSgnnLinkPredictionTestDataLoader,
                                    GSgnnLinkPredictionJointTestDataLoader,
                                    GSgnnLinkPredictionPredefinedTestDataLoader)
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER
from graphstorm.utils import (
    get_device,
    get_lm_ntypes,
    use_wholegraph,
)

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(False)

    use_wg_feats = use_wholegraph(config.part_config)
    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  local_rank=config.local_rank,
                  use_wholegraph=config.use_wholegraph_embed or use_wg_feats)

    infer_data = GSgnnEdgeInferData(config.graph_name,
                                    config.part_config,
                                    eval_etypes=config.eval_etype,
                                    node_feat_field=config.node_feat_name,
                                    decoder_edge_feat=config.decoder_edge_feat,
                                    lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))
    model = gs.create_builtin_lp_gnn_model(infer_data.g, config, train_task=False)
    model.restore_model(config.restore_model_path,
                        model_layer_to_load=config.restore_model_layers)
    infer = GSgnnLinkPredictionInferrer(model)
    infer.setup_device(device=get_device())
    if not config.no_validation:
        infer.setup_evaluator(
            GSgnnMrrLPEvaluator(config.eval_frequency,
                                infer_data,
                                config.num_negative_edges_eval,
                                config.lp_decoder_type))
        assert len(infer_data.test_idxs) > 0, "There is not test data for evaluation."
    tracker = gs.create_builtin_task_tracker(config)
    infer.setup_task_tracker(tracker)
    # We only support full-graph inference for now.
    if config.eval_etypes_negative_dstnode is not None:
        # The negatives used in evaluation is fixed.
        dataloader = GSgnnLinkPredictionPredefinedTestDataLoader(
            infer_data, infer_data.test_idxs,
            batch_size=config.eval_batch_size,
            fixed_edge_dst_negative_field=config.eval_etypes_negative_dstnode,
            fanout=config.eval_fanout)
    else:
        if config.eval_negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
            test_dataloader_cls = GSgnnLinkPredictionTestDataLoader
        elif config.eval_negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER:
            test_dataloader_cls = GSgnnLinkPredictionJointTestDataLoader
        else:
            raise ValueError('Unknown test negative sampler.'
                'Supported test negative samplers include '
                f'[{BUILTIN_LP_UNIFORM_NEG_SAMPLER}, {BUILTIN_LP_JOINT_NEG_SAMPLER}]')

        dataloader = test_dataloader_cls(infer_data, infer_data.test_idxs,
            batch_size=config.eval_batch_size,
            num_negative_edges=config.num_negative_edges_eval,
            fanout=config.eval_fanout)
    infer.infer(infer_data, dataloader,
                save_embed_path=config.save_embed_path,
                edge_mask_for_gnn_embeddings=None if config.no_validation else \
                    'train_mask', # if no validation,any edge can be used in message passing.
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
    gs_args, _ = arg_parser.parse_known_args()
    main(gs_args)
