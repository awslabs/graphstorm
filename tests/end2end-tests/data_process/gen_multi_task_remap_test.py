"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import os
import argparse
import json
import yaml
from argparse import Namespace

import numpy as np
import torch as th

from graphstorm.config import GSConfig
from graphstorm.gconstruct.id_map import IdMap
from graphstorm.model.utils import pad_file_index

def main(args):
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)
    save_prediction_path = os.path.join(output_path, "predict")
    save_embed_path = os.path.join(output_path, "emb")
    ntype0 = "n0"
    ntype1 = "n1"
    etype0 = ("n0", "access", "n1")
    etype1 = ("n1", "access", "n0")

    # generate dummy graph data
    graph_part_dir = os.path.join(output_path, "graph")
    os.makedirs(graph_part_dir, exist_ok=True)
    part_config = os.path.join(graph_part_dir, "graph.json")
    with open(part_config, 'w', encoding='utf-8') as f:
        json.dump({"graph_name":"dummy"}, f, indent=4)
    os.makedirs(os.path.join(graph_part_dir, "part0"), exist_ok=True)

    task_config = {
        "gsf": {
            "basic": {
                "backend": "gloo",
                "batch_size": 32,
                "part_config": part_config
            },
            "gnn": {
                "model_encoder_type": "rgcn",
                "num_layers": 1,
                "fanout": "5",
                "hidden_size": 32
            },
            "output": {
                "save_prediction_path": save_prediction_path,
                "save_embed_path": save_embed_path
            },
            "multi_task_learning": [
                {
                    "edge_classification": {
                        "target_etype": [
                            ",".join(etype0)
                        ],
                        "label_field": "test_ec0",
                        "num_classes": 1000,
                        "batch_size": 64
                    },
                },
                {
                    "edge_classification": {
                        "target_etype": [
                            ",".join(etype1)
                        ],
                        "label_field": "test_ec1",
                        "num_classes": 1000,
                        "batch_size": 64
                    },
                },
                {
                    "node_classification": {
                        "target_ntype": ntype0,
                        "label_field": "test_nc1",
                        "num_classes": 1000,
                        "batch_size": 64
                    },
                },
                {
                    # will be ignored
                    "link_prediction": {
                        "exclude_training_targets": False
                    }
                },
                {
                    # will be ignored
                    "reconstruct_node_feat": {
                        "target_ntype": ntype0,
                        "reconstruct_nfeat_name": "feat"
                    }
                }
            ]
        }
    }
    task_config_path = os.path.join(output_path, "task.yaml")
    with open(task_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(task_config, f)

    args = Namespace(yaml_config_file=task_config_path, local_rank=0)
    config = GSConfig(args)
    tasks = config.multi_tasks

    # generate random node ids for nodes
    nid0 = np.random.randint(10000, size=1000) * 10000 + np.arange(1000)
    nid1 = np.random.randint(10000, size=1000) * 10000 + np.arange(1000)
    nid0_str = nid0.astype('str')
    nid1_str = nid1.astype('str')

    nid0_map = IdMap(nid0_str)
    nid1_map = IdMap(nid1_str)

    mapping_subpath = "raw_id_mappings"
    os.makedirs(os.path.join(graph_part_dir, mapping_subpath), exist_ok=True)
    nid0_map.save(os.path.join(graph_part_dir, mapping_subpath, ntype0, "part-00000.parquet"))
    nid1_map.save(os.path.join(graph_part_dir, mapping_subpath, ntype1, "part-00000.parquet"))

    # generate faked edge results
    src0 = nid0[np.random.randint(1000, size=2000)]
    dst0 = nid1[np.random.randint(1000, size=2000)]
    pred0 = np.stack((src0, dst0), axis=1)
    src0_new, _ = nid0_map.map_id(src0.astype('str'))
    dst0_new, _ = nid1_map.map_id(dst0.astype('str'))

    src1 = nid0[np.random.randint(1000, size=2000)]
    dst1 = nid1[np.random.randint(1000, size=2000)]
    pred1 = np.stack((src1, dst1), axis=1)
    src1_new, _ = nid0_map.map_id(src1.astype('str'))
    dst1_new, _ = nid1_map.map_id(dst1.astype('str'))

    meta_info = {
        "format": "pytorch",
        "world_size": 2,
        "etypes": [etype0]
    }

    # For task 0 (edge classification task on etype0)
    pred_output = os.path.join(save_prediction_path, tasks[0].task_id)
    os.makedirs(pred_output)
    meta_fname = os.path.join(pred_output, "result_info.json")
    with open(meta_fname, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=4)

    pred_output_etype0 = os.path.join(pred_output, "_".join(etype0))
    os.makedirs(pred_output_etype0)
    th.save(th.tensor(pred0), os.path.join(pred_output_etype0, f"predict-{pad_file_index(0)}.pt"))
    th.save(th.tensor(src0_new), os.path.join(pred_output_etype0, f"src_nids-{pad_file_index(0)}.pt"))
    th.save(th.tensor(dst0_new), os.path.join(pred_output_etype0, f"dst_nids-{pad_file_index(0)}.pt"))
    th.save(th.tensor(pred1), os.path.join(pred_output_etype0, f"predict-{pad_file_index(1)}.pt"))
    th.save(th.tensor(src1_new), os.path.join(pred_output_etype0, f"src_nids-{pad_file_index(1)}.pt"))
    th.save(th.tensor(dst1_new), os.path.join(pred_output_etype0, f"dst_nids-{pad_file_index(1)}.pt"))

    # generate faked edge results
    src2 = nid1[np.random.randint(1000, size=2000)]
    dst2 = nid0[np.random.randint(1000, size=2000)]
    pred2 = np.stack((src2, dst2), axis=1)
    src2_new, _ = nid1_map.map_id(src2.astype('str'))
    dst2_new, _ = nid0_map.map_id(dst2.astype('str'))

    src3 = nid1[np.random.randint(1000, size=2000)]
    dst3 = nid0[np.random.randint(1000, size=2000)]
    pred3 = np.stack((src3, dst3), axis=1)
    src3_new, _ = nid1_map.map_id(src3.astype('str'))
    dst3_new, _ = nid0_map.map_id(dst3.astype('str'))

    meta_info = {
        "format": "pytorch",
        "world_size": 2,
        "etypes": [etype1]
    }

    # For task 1 (edge classification task on etype0)
    pred_output = os.path.join(save_prediction_path, tasks[1].task_id)
    os.makedirs(pred_output)
    meta_fname = os.path.join(pred_output, "result_info.json")
    with open(meta_fname, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=4)

    pred_output_etype1 = os.path.join(pred_output, "_".join(etype1))
    os.makedirs(pred_output_etype1)
    th.save(th.tensor(pred2), os.path.join(pred_output_etype1, f"predict-{pad_file_index(0)}.pt"))
    th.save(th.tensor(src2_new), os.path.join(pred_output_etype1, f"src_nids-{pad_file_index(0)}.pt"))
    th.save(th.tensor(dst2_new), os.path.join(pred_output_etype1, f"dst_nids-{pad_file_index(0)}.pt"))
    th.save(th.tensor(pred3), os.path.join(pred_output_etype1, f"predict-{pad_file_index(1)}.pt"))
    th.save(th.tensor(src3_new), os.path.join(pred_output_etype1, f"src_nids-{pad_file_index(1)}.pt"))
    th.save(th.tensor(dst3_new), os.path.join(pred_output_etype1, f"dst_nids-{pad_file_index(1)}.pt"))


    # generate faked edge results
    nid0_0 = nid0[np.random.randint(1000, size=2000)]
    pred0_0 = np.stack((nid0_0, nid0_0), axis=1)
    nid0_new_0, _ = nid0_map.map_id(nid0_0.astype('str'))

    nid0_1 = nid0[np.random.randint(1000, size=2000)]
    pred0_1 = np.stack((nid0_1, nid0_1), axis=1)
    nid0_new_1, _ = nid0_map.map_id(nid0_1.astype('str'))

    meta_info = {
        "format": "pytorch",
        "world_size": 2,
        "ntypes": [ntype0]
    }
    # For task 1 (edge classification task on etype0)
    pred_output = os.path.join(save_prediction_path, tasks[2].task_id)
    os.makedirs(pred_output)
    meta_fname = os.path.join(pred_output, "result_info.json")
    with open(meta_fname, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=4)
    pred_output_ntype0 = os.path.join(pred_output, ntype0)

    os.makedirs(pred_output_ntype0)
    th.save(th.tensor(pred0_0), os.path.join(pred_output_ntype0, f"predict-{pad_file_index(0)}.pt"))
    th.save(th.tensor(nid0_new_0), os.path.join(pred_output_ntype0, f"predict_nids-{pad_file_index(0)}.pt"))

    th.save(th.tensor(pred0_1), os.path.join(pred_output_ntype0, f"predict-{pad_file_index(1)}.pt"))
    th.save(th.tensor(nid0_new_1), os.path.join(pred_output_ntype0, f"predict_nids-{pad_file_index(1)}.pt"))

    # Only part of nodes have node embeddings
    # generate faked node embeddings
    nid0_0 = nid0[np.random.randint(1000, size=250)]
    nid1_0 = nid1[np.random.randint(1000, size=249)]
    emb0_0 = np.stack((nid0_0, nid0_0), axis=1)
    emb1_0 = np.stack((nid1_0, nid1_0), axis=1)
    nid0_new_0, _ = nid0_map.map_id(nid0_0.astype('str'))
    nid1_new_0, _ = nid1_map.map_id(nid1_0.astype('str'))

    nid0_1 = nid0[np.random.randint(1000, size=250)]
    nid1_1 = nid1[np.random.randint(1000, size=249)]
    emb0_1 = np.stack((nid0_1, nid0_1), axis=1)
    emb1_1 = np.stack((nid1_1, nid1_1), axis=1)
    nid0_new_1, _ = nid0_map.map_id(nid0_1.astype('str'))
    nid1_new_1, _ = nid1_map.map_id(nid1_1.astype('str'))

    meta_info = {
        "format": "pytorch",
        "world_size": 2,
        "emb_name": [ntype0, ntype1],
    }
    emb_output = save_embed_path
    meta_fname = os.path.join(emb_output, "emb_info.json")
    os.makedirs(emb_output)
    with open(meta_fname, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=4)

    emb_output_ntype0 = os.path.join(emb_output, ntype0)
    emb_output_ntype1 = os.path.join(emb_output, ntype1)
    os.makedirs(emb_output_ntype0)
    os.makedirs(emb_output_ntype1)
    th.save(th.tensor(emb0_0), os.path.join(emb_output_ntype0, f"embed-{pad_file_index(0)}.pt"))
    th.save(th.tensor(nid0_new_0), os.path.join(emb_output_ntype0, f"embed_nids-{pad_file_index(0)}.pt"))

    th.save(th.tensor(emb1_0), os.path.join(emb_output_ntype1, f"embed-{pad_file_index(0)}.pt"))
    th.save(th.tensor(nid1_new_0), os.path.join(emb_output_ntype1, f"embed_nids-{pad_file_index(0)}.pt"))

    th.save(th.tensor(emb0_1), os.path.join(emb_output_ntype0, f"embed-{pad_file_index(1)}.pt"))
    th.save(th.tensor(nid0_new_1), os.path.join(emb_output_ntype0, f"embed_nids-{pad_file_index(1)}.pt"))

    th.save(th.tensor(emb1_1), os.path.join(emb_output_ntype1, f"embed-{pad_file_index(1)}.pt"))
    th.save(th.tensor(nid1_new_1), os.path.join(emb_output_ntype1, f"embed_nids-{pad_file_index(1)}.pt"))




if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Check multi task remapping")
    argparser.add_argument("--output", type=str, required=True,
                           help="Path to save the generated data")

    args = argparser.parse_args()

    main(args)