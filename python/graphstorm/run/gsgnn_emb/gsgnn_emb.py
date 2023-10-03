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
from graphstorm.dataloading import GSgnnLPJointNegDataLoader
from graphstorm.dataloading import GSgnnLPLocalUniformNegDataLoader
from graphstorm.dataloading import GSgnnLPLocalJointNegDataLoader
from graphstorm.dataloading import GSgnnAllEtypeLPJointNegDataLoader
from graphstorm.dataloading import GSgnnAllEtypeLinkPredictionDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionTestDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionJointTestDataLoader
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_LOCALJOINT_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER
from graphstorm.dataloading import (BUILTIN_FAST_LP_UNIFORM_NEG_SAMPLER,
                                    BUILTIN_FAST_LP_JOINT_NEG_SAMPLER,
                                    BUILTIN_FAST_LP_LOCALUNIFORM_NEG_SAMPLER,
                                    BUILTIN_FAST_LP_LOCALJOINT_NEG_SAMPLER)
from graphstorm.dataloading import (FastGSgnnLinkPredictionDataLoader,
                                    FastGSgnnLPJointNegDataLoader,
                                    FastGSgnnLPLocalUniformNegDataLoader,
                                    FastGSgnnLPLocalJointNegDataLoader)
from graphstorm.model.utils import save_embeddings
from graphstorm.model import do_full_graph_inference
from graphstorm.utils import rt_profiler, sys_tracker, setup_device, use_wholegraph


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
    train_data = GSgnnLPTrainData(config.graph_name,
                                  config.part_config,
                                  train_etypes=config.train_etype,
                                  eval_etypes=config.eval_etype,
                                  node_feat_field=config.node_feat_name,
                                  pos_graph_feat_field=config.lp_edge_weight_for_loss)

    # Preparing input layer for training or inference.
    # The input layer can pre-compute node features in the preparing step if needed.
    # For example pre-compute all BERT embeddings
    assert (config.save_embed_path is not None, "save embeded path cannot be none for gs_gen_embeddings")
    assert (config.restore_model_path is not None, "restore model path cannot be none for gs_gen_embeddings")

    model = gs.create_builtin_lp_gnn_model(train_data.g, config, train_task=False)
    model_path = config.restore_model_path
    # TODO(zhengda) the model path has to be in a shared filesystem.
    model.restore_model(model_path)
    # Preparing input layer for training or inference.
    # The input layer can pre-compute node features in the preparing step if needed.
    # For example pre-compute all BERT embeddings
    model.prepare_input_encoder(train_data)
    # TODO(zhengda) we may not want to only use training edges to generate GNN embeddings.
    embeddings = do_full_graph_inference(model, train_data, fanout=config.eval_fanout,
                                         edge_mask="train_mask", task_tracker=tracker)
    save_embeddings(config.save_embed_path, embeddings, gs.get_rank(),
                     gs.get_world_size(),
                     device=device,
                     node_id_mapping_file=config.node_id_mapping_file)


def generate_parser():
    """ Generate an argument parser
    """
    parser = get_argument_parser()
    return parser


if __name__ == '__main__':
    arg_parser = generate_parser()

    args = arg_parser.parse_args()
    main(args)
