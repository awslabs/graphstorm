import json
import os

import torch as th
from dgl.data.utils import load_tensors, save_tensors
from graphstorm.model.utils import load_dist_nid_map

def load_hard_negative_config(gsprocessing_config):
    with open(gsprocessing_config, 'r') as file:
        config = json.load(file)

    # Hard Negative only supports link prediction
    edges_config = config['graph']['edges']
    mapping_edge_list = []
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
                mapping_edge_list.append({"dst_node_type": single_edge_config["dest"]["type"],
                                          "edge_type": edge_type,
                                          "hard_neg_feat_name": hard_neg_feat_name})
    return mapping_edge_list


def shuffle_hard_negative_nids(gsprocessing_config, num_parts, output_path):
    shuffled_edge_config = load_hard_negative_config(gsprocessing_config)

    node_type_list = []
    for single_shuffled_edge_config in shuffled_edge_config:
        node_type = single_shuffled_edge_config["dst_node_type"]
        node_type_list.append(node_type)
    node_mapping = load_dist_nid_map(f"{output_path}/dist_graph", node_type_list)
    gnid2pnid_mapping = {}

    def get_gnid2pnid_map(ntype):
        if ntype in gnid2pnid_mapping:
            return gnid2pnid_mapping[ntype]
        else:
            pnid2gnid_map = node_mapping[ntype]
            gnid2pnid_map = th.argsort(pnid2gnid_map)
            gnid2pnid_mapping[ntype] = gnid2pnid_map
            # del ntype in node_mapping to save memory
            del node_mapping[ntype]
            return gnid2pnid_mapping[ntype]

    # iterate all the partitions to convert hard negative node ids.
    for i in range(num_parts):
        part_path = os.path.join(f"{output_path}/dist_graph", f"part{i}")
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
            gnid2pnid_map = get_gnid2pnid_map(neg_ntype)
            hard_nids[hard_nid_idx] = gnid2pnid_map[hard_nids[hard_nid_idx]]

        # replace the edge_feat.dgl with the updated one.
        os.remove(edge_feat_path)
        save_tensors(edge_feat_path, edge_feats)
    
