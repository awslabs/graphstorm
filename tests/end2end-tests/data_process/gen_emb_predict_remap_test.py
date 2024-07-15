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

    Generate data for testing edge prediction remapping
"""

import os
import argparse
import json

import numpy as np
import torch as th

from graphstorm.gconstruct.id_map import IdMap
from graphstorm.model.utils import pad_file_index

def main(args):
    output_path = args.output
    ntype0 = "n0"
    ntype1 = "n1"

    # generate random node ids for nodes
    nid0 = np.random.randint(10000, size=1000) * 10000 + np.arange(1000)
    nid1 = np.random.randint(10000, size=1000) * 10000 + np.arange(1000)
    nid0_str = nid0.astype('str')
    nid1_str = nid1.astype('str')

    nid0_map = IdMap(nid0_str)
    nid1_map = IdMap(nid1_str)

    mapping_subpath = "id_mapping"
    os.makedirs(os.path.join(output_path, mapping_subpath), exist_ok=True)
    nid0_map.save(os.path.join(output_path, mapping_subpath, ntype0, "part-00000.parquet"))
    nid1_map.save(os.path.join(output_path, mapping_subpath, ntype1, "part-00000.parquet"))

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

    emb_output = os.path.join(output_path, "partial-emb")
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

    # Case two: All the nodes have node embeddings
    emb0_0 = np.stack((nid0[:500], nid0[:500]), axis=1)
    emb1_0 = np.stack((nid1[:500], nid1[:500]), axis=1)
    emb0_1 = np.stack((nid0[500:], nid0[500:]), axis=1)
    emb1_1 = np.stack((nid1[500:], nid1[500:]), axis=1)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Check edge prediction remapping")
    argparser.add_argument("--output", type=str, required=True,
                           help="Path to save the generated data")

    args = argparser.parse_args()

    main(args)
