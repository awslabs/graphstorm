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

    Inference and training script for distillation tasks with GNN
"""

import random
import os
import torch as th
import numpy as np
import graphstorm as gs
from graphstorm.gconstruct.file_io import write_data_hdf5, read_data_hdf5
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnLinkPredictionInfer
from graphstorm.distiller import GSdistiller, read_nid
from graphstorm.eval import GSgnnMrrLPEvaluator
from graphstorm.dataloading import GSgnnEdgeInferData
from graphstorm.dataloading import GSgnnLinkPredictionTestDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionJointTestDataLoader
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER
from graphstorm.utils import setup_device


def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(False)
    gs.initialize(ip_config=config.ip_config, backend=config.backend)
    device = setup_device(config.local_rank)

    # get node types for distillation
    if not config.distill_feats:
        node_types = [
            re.match(r"(.*)_id_remap.parquet", nid_file.name).group(1)
            for nid_file in Path(os.path.dirname(config.part_config)).glob("*_id_remap.parquet")
        ]
    else:
        node_types = config.distill_feats.keys()
    print (f"node types: {node_types}\n")

    # do gnn inference
    if config.need_infer:
        infer_data = GSgnnEdgeInferData(config.graph_name,
                                        config.part_config,
                                        eval_etypes=config.eval_etype,
                                        node_feat_field=config.node_feat_name)

        # TODO (HZ): test gnn checkpoints from ep and np model
        # ep or np gnn checkpoint should also work
        # since we do inference only, we load lp gnn model by default
        model = gs.create_builtin_lp_gnn_model(infer_data.g, config, train_task=False)
        model.restore_model(config.restore_model_path)
        distiller = GSdistiller(model, gs.get_rank(), config)
        distiller.setup_device(device=device)

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
        model.prepare_input_encoder(infer_data)
        gnn_embeds = distiller.infer(infer_data, dataloader,
            edge_mask_for_gnn_embeddings=None if config.no_validation else \
                'train_mask', # if no validation,any edge can be used in message passing.
            node_id_mapping_file=config.node_id_mapping_file)

        if gs.get_rank() == 0:
            print (f"Writing GNN embeddings to {os.path.join(config.textual_path, 'embed_map.hdf5')}")
            embed_dict = {}
            for node_type in node_types:
                embed_dict[node_type] = np.array(gnn_embeds[node_type][0:len(gnn_embeds[node_type])])
            # save to disk so that it can read from disk
            write_data_hdf5(embed_dict, os.path.join(config.textual_path, "embed_map.hdf5"))
        th.distributed.barrier()

        gnn_embeds = read_data_hdf5(os.path.join(config.textual_path, "embed_map.hdf5"), ["query", "asin"], True)

        for node_type in node_types:
            id_map = {k: v for v, k in enumerate(read_nid(node_type, os.path.dirname(config.part_config)))}

            file_list = os.listdir(os.path.join(config.textual_path, node_type))
            sample_k = min(int(0.1 * len(file_list)), 32)
            if sample_k < 1:
                sample_k = 1

            # need to make sure the sample of files across trainers are same
            random.seed(42)
            eval_file_list = random.sample(file_list, sample_k)
            train_file_list = [x for x in file_list if x not in eval_file_list]

            local_train_file_list = distiller.get_textual_file_names_per_worker(train_file_list, gs.get_rank(), th.distributed.get_world_size())
            local_eval_file_list = distiller.get_textual_file_names_per_worker(eval_file_list, gs.get_rank(), th.distributed.get_world_size())
            
            # maintain a queue for each trainer to write files with fixed length
            queue = {
                "node_ids": [],
                "textual_feats": [],
                "embeddings": [],
            }
            part = 0
            for index, file_name in enumerate(local_train_file_list):
                queue, part = distiller.map_text_embeds(
                    queue,
                    node_type,
                    os.path.join(config.textual_path, node_type, file_name),
                    config.distill_feats[node_type]["id_field"],
                    config.distill_feats[node_type]["textual_field"],
                    gnn_embeds[node_type],
                    id_map,
                    index,
                    len(local_train_file_list),
                    config.concat_field_name,
                    "train",
                    config.chunk_size,
                    part=part,
                    rank=gs.get_rank(),
                )
            for index, file_name in enumerate(local_eval_file_list):
                queue, part = distiller.map_text_embeds(
                    queue,
                    node_type,
                    os.path.join(config.textual_path, node_type, file_name),
                    config.distill_feats[node_type]["id_field"],
                    config.distill_feats[node_type]["textual_field"],
                    gnn_embeds[node_type],
                    id_map,
                    index,
                    len(local_train_file_list),
                    config.concat_field_name,
                    "val",
                    config.chunk_size,
                    part=part,
                    rank=gs.get_rank(),
                )
    else:
        # no need to pass GNN model if no GNN inference
        distiller = GSdistiller(None, gs.get_rank(), config)
        distiller.setup_device(device=device)
    th.distributed.barrier()

    # start distilling the model
    for node_type in node_types:
        distiller.distill(
            node_type, 
            config.tsf_name, 
            config.hidden_size,
            os.path.join(config.textual_path, \
            node_type + "_distill_data"),
            config.distill_batch_size,
            config.distill_lr,
            saved_path=config.save_student_path,
        )

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
