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

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnLinkPredictionInfer
from graphstorm.eval import GSgnnMrrLPEvaluator
from graphstorm.dataloading import GSgnnEdgeInferData
from graphstorm.dataloading import GSgnnLinkPredictionTestDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionJointTestDataLoader
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    gs.initialize(ip_config=config.ip_config, backend=config.backend)

    infer_data = GSgnnEdgeInferData(config.graph_name,
                                    config.part_config,
                                    eval_etypes=config.eval_etype,
                                    node_feat_field=config.node_feat_name)
    model = gs.create_builtin_lp_model(infer_data.g, config, train_task=False)
    model.restore_model(config.restore_model_path)
    # TODO(zhengda) we should use a different way to get rank.
    infer = GSgnnLinkPredictionInfer(model, gs.get_rank())
    infer.setup_cuda(dev_id=config.local_rank)
    if not config.no_validation:
        infer.setup_evaluator(
            GSgnnMrrLPEvaluator(config.eval_frequency,
                                infer_data,
                                config.num_negative_edges_eval,
                                config.lp_decoder_type))
        assert len(infer_data.test_idxs) > 0, "There is not test data for evaluation."
    tracker = gs.create_builtin_task_tracker(config, infer.rank)
    infer.setup_task_tracker(tracker)
    # We only support full-graph inference for now.
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
                                     num_negative_edges=config.num_negative_edges_eval)
    # Preparing input layer for training or inference.
    # The input layer can pre-compute node features in the preparing step if needed.
    # For example pre-compute all BERT embeddings
    model.prepare_input_encoder(infer_data)
    infer.infer(infer_data, dataloader,
                save_embed_path=config.save_embed_path,
                node_id_mapping_file=config.node_id_mapping_file)

def generate_parser():
    """ Generate an argument parser
    """
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    arg_parser=generate_parser()

    args = arg_parser.parse_args()
    print(args)
    main(args)
