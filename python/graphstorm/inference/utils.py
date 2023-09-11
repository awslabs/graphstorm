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

    Utility functions and classes.
"""

import os
import torch as th
import pandas as pd

def read_nid(node_type, dist_dgl_graph_dir):
    nid_file = os.path.join(dist_dgl_graph_dir, f"{node_type}_id_remap.parquet")
    print (f"reading nid-{node_type}: {nid_file}")
    node_ids = pd.read_parquet(nid_file)["orig"].tolist()
    print (f"read nid of size {len(node_ids)}")
    return node_ids

def read_embed(node_type, embed_path, node_map=None):
    embed_all = []
    embed_part_file_path = lambda part_idx: os.path.join(
        embed_path, f"{node_type}_emb.part{part_idx}.bin")

    print (f"reading {node_type} embeddings")
    shard_idx = 0
    while os.path.exists(embed_part_file_path(shard_idx)):
        print (f"reading {embed_part_file_path(shard_idx)}")
        embed_all.append(th.load(embed_part_file_path(shard_idx)))
        shard_idx += 1
    num_shards = shard_idx
    
    embed_all = th.cat(embed_all, 0)

    if node_map is not None:
        embed_original_order = th.empty_like(embed_all)
        embed_original_order[node_map[node_type]] = embed_all
    else:
        embed_original_order = embed_all

    print (f"read embeddings of shape {embed_original_order.shape}")
    return embed_original_order.numpy(), num_shards