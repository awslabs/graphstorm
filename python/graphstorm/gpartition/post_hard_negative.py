"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import json
import os

from dgl.data.utils import load_tensors, save_tensors
from graphstorm.model.utils import load_dist_nid_map
from graphstorm.gconstruct.utils import get_gnid2pnid_map


def load_hard_negative_config(gsprocessing_config: str):
    """Load GSProcessing Config to extract hard negative config

    Parameters
    ----------------
    gsprocessing_config: str
        Path to the gsprocessing config.

    Returns
    -------
    list of dicts
        A list of dict for each hard negative feature transformation.
        Each dict will look like:
        {
            "dst_node_type": destination node type for hard negative,
            "edge_type": edge_type,
            "hard_neg_feat_name": feature name
        }
    """
    with open(gsprocessing_config, 'r', encoding='utf-8') as file:
        config = json.load(file)

    # Hard Negative only supports link prediction
    edges_config = config['graph']['edges']
    hard_neg_list = []
    for single_edge_config in edges_config:
        if "features" not in single_edge_config:
            continue
        feature_dict = single_edge_config["features"]
        for single_feature in feature_dict:
            if single_feature["transformation"]["name"] \
                    == "edge_dst_hard_negative":
                edge_type = ":".join([single_edge_config["source"]["type"],
                                     single_edge_config["relation"]["type"],
                                     single_edge_config["dest"]["type"]])
                hard_neg_feat_name = single_feature['name']
                hard_neg_list.append({"dst_node_type": single_edge_config["dest"]["type"],
                                          "edge_type": edge_type,
                                          "hard_neg_feat_name": hard_neg_feat_name})
    return hard_neg_list


def shuffle_hard_negative_nids(gsprocessing_config: str,
                               num_parts: int, graph_path: str):
    """Shuffle hard negative edge feature ids with int-to-int node id mapping.
    The function here align with the shuffle_hard_nids in graphstorm.gconstruct.utils.

    Parameters
    ----------------
    gsprocessing_config: str
        Path to the gsprocessing config.
    num_parts: int
        Number of parts.
    graph_path: str
        Path to the output DGL graph.
    """
    shuffled_edge_config = load_hard_negative_config(gsprocessing_config)

    node_type_list = []
    for single_shuffled_edge_config in shuffled_edge_config:
        node_type = single_shuffled_edge_config["dst_node_type"]
        node_type_list.append(node_type)
    node_mapping = load_dist_nid_map(f"{graph_path}/dist_graph", node_type_list)
    gnid2pnid_mapping = {}

    # iterate all the partitions to convert hard negative node ids.
    for i in range(num_parts):
        part_path = os.path.join(f"{graph_path}/dist_graph", f"part{i}")
        edge_feat_path = os.path.join(part_path, "edge_feat.dgl")

        # load edge features first
        edge_feats = load_tensors(edge_feat_path)
        for single_shuffled_edge_config in shuffled_edge_config:
            etype = single_shuffled_edge_config["edge_type"]
            neg_feat = single_shuffled_edge_config["hard_neg_feat_name"]
            neg_ntype = single_shuffled_edge_config["dst_node_type"]
            efeat_name = f"{etype}/{neg_feat}"
            hard_nids = edge_feats[efeat_name].long()
            hard_nid_idx = hard_nids > -1
            gnid2pnid_map = get_gnid2pnid_map(neg_ntype, node_mapping,
                                              gnid2pnid_mapping)
            hard_nids[hard_nid_idx] = gnid2pnid_map[hard_nids[hard_nid_idx]]

        # replace the edge_feat.dgl with the updated one.
        os.remove(edge_feat_path)
        save_tensors(edge_feat_path, edge_feats)
