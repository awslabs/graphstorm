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
import pandas as pd
import json
import random
from multiprocessing import Pool
import os
import torch as th
import numpy as np
# import dask.dataframe as dd
import graphstorm as gs
from graphstorm.gconstruct.file_io import write_data_hdf5, read_data_hdf5
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnLinkPredictionInfer, read_nid, read_embed
from graphstorm.distiller import GSdistiller
from graphstorm.eval import GSgnnMrrLPEvaluator
from graphstorm.dataloading import GSgnnEdgeInferData
from graphstorm.dataloading import GSgnnLinkPredictionTestDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionJointTestDataLoader
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER
from graphstorm.utils import setup_device

def get_textual_file_names_per_worker(total_file_list, local_rank, world_size):
        num_files = len(total_file_list)
        # if local_rank > num_files:
        #     remainder = local_rank % num_files
        #     part_start = 0
        #     part_end = num_files
        #     part_len = part_end
        # else:
        part_len = num_files // world_size
        remainder = num_files % world_size
        part_start = part_len * local_rank + min(local_rank, remainder)
        part_end = part_start + part_len + (local_rank < remainder)
        part_len = part_end - part_start

        if part_len == 0:
            return []

        local_file_list = []
        for i in range(part_len):
            local_file_list.append(total_file_list[part_start+i])
        return local_file_list

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(False)
    gs.initialize(ip_config=config.ip_config, backend=config.backend)
    device = setup_device(config.local_rank)

    infer_data = GSgnnEdgeInferData(config.graph_name,
                                    config.part_config,
                                    eval_etypes=config.eval_etype,
                                    node_feat_field=config.node_feat_name)
    model = gs.create_builtin_lp_gnn_model(infer_data.g, config, train_task=False)
    model.restore_model(config.restore_model_path)
    # TODO(zhengda) we should use a different way to get rank.
    # infer = GSgnnLinkPredictionInfer(model, gs.get_rank())
    # infer.setup_device(device=device)
    distiller = GSdistiller(model, None, gs.get_rank(), config)
    distiller.setup_device(device=device)
    # if not config.no_validation:
    #     infer.setup_evaluator(
    #         GSgnnMrrLPEvaluator(config.eval_frequency,
    #                             infer_data,
    #                             config.num_negative_edges_eval,
    #                             config.lp_decoder_type))
    #     assert len(infer_data.test_idxs) > 0, "There is not test data for evaluation."
    tracker = gs.create_builtin_task_tracker(config, distiller.rank)
    # infer.setup_task_tracker(tracker)
    distiller.setup_task_tracker(tracker)
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
                                     num_negative_edges=config.num_negative_edges_eval,
                                     fanout=config.eval_fanout)
    # Preparing input layer for training or inference.
    # The input layer can pre-compute node features in the preparing step if needed.
    # For example pre-compute all BERT embeddings

    # get node types for distillation
    if not config.distill_feats:
        node_types = [
            re.match(r"(.*)_id_remap.parquet", nid_file.name).group(1)
            for nid_file in Path(os.path.dirname(config.part_config)).glob("*_id_remap.parquet")
        ]
    else:
        node_types = config.distill_feats.keys()
    print (f"node types: {node_types}\n")

    if config.need_infer:
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
    # id_maps = {}
    for node_type in node_types:
        id_map = {k: v for v, k in enumerate(read_nid(node_type, os.path.dirname(config.part_config)))}
        # id_map = read_nid(node_type, os.path.dirname(config.part_config))
        # print ("haha", gnn_embeds[node_type][[345,325,211]])

        file_list = os.listdir(os.path.join(config.textual_path, node_type))
        sample_k = min(int(0.1 * len(file_list)), 32)
        if sample_k < 1:
            sample_k = 1
        random.seed(42)
        eval_file_list = random.sample(file_list, sample_k)
        train_file_list = [x for x in file_list if x not in eval_file_list]

        local_train_file_list = get_textual_file_names_per_worker(train_file_list, gs.get_rank(), th.distributed.get_world_size())
        local_eval_file_list = get_textual_file_names_per_worker(eval_file_list, gs.get_rank(), th.distributed.get_world_size())
        
        queue = {
            "node_ids": [],
            "textual_feats": [],
            "embeddings": [],
        }
        part = 0
        for index, file_name in enumerate(local_train_file_list):
            queue, part = map_text_embeds(
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
            queue, part = map_text_embeds(
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
    print ("done")
    exit()

    


    # TODO: 1. gnn inference, 2. embedding mapping with textual features, 3. distill
    # option2: don't need to map textual with embeddings, store embeddings into hdf5 file and load embeddings in distilled dataloader
    # final conclusion: split inference+shuffle and distillation. 
    # All trainers take care of id mapping. Embeddings saved in hdf5 file so there will be no memory conern.
    # each trainer need to have their own string ID to interger ID

    # if gs.get_rank() == 0:
    #     embed_dict = {}
    #     for node_type in node_types:
    #         embed_dict[node_type] = np.array(gnn_embeds[node_type][0:len(gnn_embeds[node_type])])
    #     write_data_hdf5(embed_dict, os.path.join(config.textual_path, "embed_map.hdf5"))
    #     print ("haha done")
    # th.distributed.barrier()
    # exit()

    if gs.get_rank() == 0:
        embed_map = read_data_hdf5(os.path.join(config.textual_path, "embed_map.hdf5"), ["query", "asin"], True)
        print (type(embed_map["asin"][394]))
        print ("haha")
    th.distributed.barrier()
    exit()

    if gs.get_rank() == 0:
        gnn_embed_orig_id = {}
        for node_type in node_types:
            # map embeddings with original ID into disk
            print (f"mapping {node_type} embeddings with orig ID")
            # node_embed2, _ = read_embed(node_type, config.save_embed_path, node_map=None)
            # node_embed = gnn_embeds[node_type]
            node_embed = gnn_embeds[node_type][0:len(gnn_embeds[node_type])]
            node_ids = read_nid(node_type, os.path.dirname(config.part_config))

            assert len(node_ids) == node_embed.shape[0]

            print (f"concatenating {node_type} embeddings")

            embed_pddf = pd.DataFrame({
                "ids": node_ids,
                "embeddings": (node_embed[i].tolist() for i in range(len(node_embed)))
            }).set_index("ids")
            # embed_pddf = pd.DataFrame({
            #     "ids":
            #     node_ids,
            #     "embeddings": (node_embed_i for node_embed_i in node_embed)
            # }).set_index("ids")

            # embed_daskdf = dd.from_pandas(embed_pddf,
            #                               npartitions=args.save_npartitions,
            #                               sort=False)
            del node_ids, node_embed
            gnn_embed_orig_id[node_type] = embed_pddf

            # print (f"writing {node_type} embeddings ...")
            # embed_daskdf.to_parquet(os.path.join(args.save_path,
            #                                      node_type + "_embed_df"),
            #                         overwrite=True,
            #                         schema={
            #                             "embeddings":
            #                             pa.list_(pa.from_numpy_dtype(node_embed_dtype))
            #                         })

            # join embeddings with textual features by original ID into disk
            file_list = os.listdir(os.path.join(config.textual_path, node_type))
            sample_k = min(int(0.1 * len(file_list)), 32)
            if sample_k < 1:
                sample_k = 1

            eval_file_list = random.sample(file_list, sample_k)
            train_file_list = [x for x in file_list if x not in eval_file_list]
            process_workers = min(8, len(train_file_list))

            with Pool(processes=process_workers) as mp_pool:
                argument_list = [(node_type, os.path.join(config.textual_path, node_type, file_name), \
                    config.distill_feats[node_type]["id_field"], \
                    config.distill_feats[node_type]["textual_field"], embed_pddf, \
                    file_index, len(file_list), config.concat_field_name, "train") \
                    for file_index, file_name in enumerate(train_file_list)]
                mp_pool.starmap(join_textual_feature, argument_list)

            process_workers = min(8, len(eval_file_list))
            with Pool(processes=process_workers) as mp_pool:
                argument_list = [(node_type, os.path.join(config.textual_path, node_type, file_name), \
                    config.distill_feats[node_type]["id_field"], \
                    config.distill_feats[node_type]["textual_field"], embed_pddf, \
                    file_index, len(file_list), config.concat_field_name, "val") \
                    for file_index, file_name in enumerate(eval_file_list)]
                mp_pool.starmap(join_textual_feature, argument_list)

            print (f"done converting {node_type} embeddings\n")

            # distill Transformer-based model per node type
            # from transformers import DistilBertConfig, DistilBertModel
            # configuration = DistilBertConfig()
            # model = DistilBertModel(configuration)
            # output = model(torch.LongTensor([[0,1,2,3]]))
            # output_embed = embed.last_hidden_state # shape torch.Size([1, 4, 768])
            # project Transformer-based embed to the same dim of GNN embed

            # TODO: 1. need a dt trainer, 2. need to create a distill model under model/ folder, 3. in the model class, add BERT part and projection part (MLP)
    th.distributed.barrier()
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

def map_text_embeds(
    queue,
    node_type,
    file_name,
    id_field,
    textual_field_list,
    disk_embeds,
    id_map,
    index,
    total_num,
    concat_field_name=False,
    split="train",
    chunk_size=10000,
    part=0,
    rank=0,
):
    print ("Worker {}: Map {} textual features: {} of {}".format(gs.get_rank(), node_type, index, total_num))
    assert len(queue["node_ids"]) < chunk_size, "Queue already greater than chunk size"

    embed_offset = []

    with open(file_name, "r") as reader:
        for line in reader:
            # TODO (HZ): add the option for reading from parquet data
            json_data = json.loads(line)
            text = ""
            for i, f in enumerate(textual_field_list):
                if f in json_data and json_data[f] is not None and str(json_data[f]):
                    if concat_field_name:
                        text += f + ": "
                    if not isinstance(json_data[f], str):
                        print (f"Warning: {f} is not a string feature")
                    field_value = str(json_data[f])
                    if i != len(textual_field_list) - 1:
                        text += field_value + " "
                    else:
                        text += field_value
            if text == "":
                print (f"Warning: empty textual features for node {json_data[id_field]}, fill with node id")
                text = str(json_data[id_field])
            queue["node_ids"].append(json_data[id_field])
            queue["textual_feats"].append(text)
            embed_offset.append(id_map[json_data[id_field]])

            if len(queue["node_ids"]) == chunk_size:
                queue["embeddings"] = queue["embeddings"] + embed_offset
                textual_embed_pddf = pd.DataFrame({
                    "ids": queue["node_ids"],
                    "textual_feats": queue["textual_feats"], 
                    "embeddings": disk_embeds[queue["embeddings"]].tolist()
                }).set_index("ids")
                outdir = os.path.join(os.path.dirname(os.path.dirname(file_name)), \
                    node_type + "_distill_data", split)
                if not os.path.exists(outdir):
                    os.makedirs(outdir, exist_ok=True)
                textual_embed_pddf.to_parquet(os.path.join(outdir, f"rank-{rank}-part-{part}.parquet"))
                for k in queue.keys():
                    queue[k] = []
                embed_offset = []
                part += 1

    queue["embeddings"] = queue["embeddings"] + embed_offset
    return queue, part



def join_textual_feature(
    node_type, 
    file_name, 
    id_field, 
    textual_field_list, 
    embed_dict, 
    index, 
    total_num, 
    concat_field_name=False,
    split="train",
):
    print ("Join {} textual features: {} of {}".format(node_type, index, total_num))
    node_ids = []
    textual_feats = []
    embeds = []
    with open(file_name, "r") as reader:
        for line in reader:
            # TODO (HZ): add the option for reading from parquet data
            json_data = json.loads(line)
            text = ""
            for i, f in enumerate(textual_field_list):
                if f in json_data and json_data[f] is not None and str(json_data[f]):
                    if concat_field_name:
                        text += f + ": "
                    if not isinstance(json_data[f], str):
                        print (f"Warning: {f} is not a string feature")
                    field_value = str(json_data[f])
                    if i != len(textual_field_list) - 1:
                        text += field_value + " "
                    else:
                        text += field_value
            if text == "":
                print (f"Warning: empty textual features for node {json_data[id_field]}, fill with node id")
                text = str(json_data[id_field])
            node_ids.append(json_data[id_field])
            textual_feats.append(text)
            embeds.append(embed_dict['embeddings'][json_data[id_field]])
    textual_embed_pddf = pd.DataFrame({
        "ids": node_ids,
        "textual_feats": textual_feats, 
        "embeddings": embeds
    }).set_index("ids")
    # textual_embed_daskdf = dd.from_pandas(textual_embed_pddf,
    #                               npartitions=1,
    #                               sort=False)
    outdir = os.path.join(os.path.dirname(os.path.dirname(file_name)), \
        node_type + "_distill_data", split)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    textual_embed_pddf.to_parquet(os.path.join(outdir, file_name.split("/")[-1].split(".")[0]+".parquet"))

    return None

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
