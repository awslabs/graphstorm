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

    Generation of Dummy Supernode-based Graph Classification Raw Data

    This script can generate the supernode based graph classification raw data for GraphStorm.

    The general requirements are:

    1. allow change of graph size in terms of 
        a. the number of individual subgraphs, 
        b. the overall number of node/edge types, and
        c. the max and min number of nodes/edges in subgraphs.
    2. better to have multiple processes methods.

    The implementaion ideas are:

    1. set up the overall number of node/edge types;
    2. randomly create one subgraph in the following steps:
        a. assign a subgraph id;
        b. randomly pick some node types and edge types for the subgraph;
        c. generate the nodes (ntype + type specific id + subgraph id) in the subgraph
        following the max and min;
        d. generate the edges (src_ntype, etype, dst_ntype) in the subgraph following the max and min;
        e. return node lists and edge lists
    3. repeat the step 2 until reach the total number of individual subgraphs;
    4. collect all subgraph node lists and edge lists, and concatenate together to form a collective 
    node lists and a collective edge lists.
    5. from the collective node lists, extract all subgraph ids to form a new "supernode" list;
    6. based on the collective node lists, create a new set of (ntype, "to_super" "supernode") edges;
    7. save the collective node lists, and collective edge list plus the new "to_super" edge list;
    8. create the construction JSON file.

"""

import os
import argparse
import json
import pandas as pd
import numpy as np


def gengerate_subgraph(etype_list, sg_id, max_nodes, min_nodes):
    node_set = set()

    # extract all node types
    for src_ntype, _, dst_ntype in etype_list:
        node_set.add(src_ntype)
        node_set.add(dst_ntype)

    # generate string type node ids and node lists
    node_dict = {}
    for ntype in node_set:
        num_nodes = np.random.randint(min_nodes, (max_nodes + 1))
        int_idx = np.arange(num_nodes)
        node_dict[ntype] = np.array([f'{ntype}_{int_id}_{sg_id}' for int_id in int_idx])

    # generate edge lists, normally 3 times of the max number of nodes
    edge_dict = {}
    for src_ntype, etype, dst_ntype in etype_list:
        num_edges = max(node_dict[src_ntype].shape[0], node_dict[dst_ntype].shape[0]) * 3

        src_idx = np.random.randint(0, node_dict[src_ntype].shape[0], num_edges)
        src_nodes = node_dict[src_ntype][src_idx].reshape(-1, 1)
        dst_idx = np.random.randint(0, node_dict[dst_ntype].shape[0], num_edges)
        dst_nodes = node_dict[dst_ntype][dst_idx].reshape(-1, 1)

        edge_list = np.hstack([src_nodes, dst_nodes])
        edge_list = np.unique(edge_list, axis=0)

        edge_dict[(src_ntype, etype, dst_ntype)] = (edge_list[:, 0], edge_list[:, 1])

    # generate to_super edges between all nodes and the subgraph id
    for ntype, nids in node_dict.items():
        edge_dict[(ntype, f'{ntype}_to_super', 'super')] = (nids, np.array([sg_id] * len(nids)))

    return node_dict, edge_dict


def random_generate_edge_types(ntypes):
    num_ntypes = len(ntypes)

    # create edge types
    min_etypes = num_ntypes
    max_etypes = num_ntypes**2
    num_etypes = np.random.randint(min_etypes, max_etypes)

    src_ntypes = ntypes[np.random.randint(0, num_ntypes, num_etypes)].reshape(-1, 1)
    dst_ntypes = ntypes[np.random.randint(0, num_ntypes, num_etypes)].reshape(-1, 1)

    etypes = np.hstack([src_ntypes, dst_ntypes])
    etypes = np.unique(etypes, axis=0)
    can_etypes = []
    for src_ntype, dst_ntype in etypes:
        can_etypes.append((src_ntype, f'{src_ntype}_to_{dst_ntype}', dst_ntype))

    return can_etypes


def save_as_raw_table(save_path, node_dict, edge_dict):
    # save node raw tabels
    node_file_base = os.path.join(save_path, 'nodes')
    os.makedirs(node_file_base, exist_ok=True)
    node_file_paths = {}
    for ntype, node_df in node_dict.items():
        node_file_path = os.path.join(node_file_base, ntype + '.parquet')
        node_df.to_parquet(node_file_path)
        node_file_paths[ntype] = node_file_path

    # save edge raw tables
    edge_file_base = os.path.join(save_path, 'edges')
    os.makedirs(edge_file_base, exist_ok=True)
    edge_file_paths = {}
    for can_etype, edge_df in edge_dict.items():
        edge_file_path = os.path.join(edge_file_base, '-'.join(can_etype) + '.parquet')
        edge_df.to_parquet(edge_file_path)
        edge_file_paths[can_etype] = edge_file_path

    # generate the raw table JSON file that graphstorm needs
    node_jsons = []
    for ntype, node_df in node_dict.items():
        node_obj = {}
        node_obj['node_type'] = ntype
        node_obj['format'] = {'name': 'parquet'}       # In this example, we just use parquet
        node_obj['files'] = node_file_paths[ntype]

        labels_list = []
        feats_list = []
        # check all dataframe columns
        for col in node_df.columns:
            label_dict = {}
            feat_dict = {}
            if col == 'nid':
                node_obj['node_id_col'] = col
            elif col == 'label':
                label_dict['label_col'] = col
                label_dict['task_type'] = 'classification'
                label_dict['split_pct'] = [0.8, 0.1, 0.1]
                label_dict['label_stats_type'] = 'frequency_cnt'
                labels_list.append(label_dict)
            elif col == 'text':
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = col
                feat_dict['transform'] = {"name": "tokenize_hf",
                                          "bert_model": "bert-base-uncased",
                                          "max_seq_length": 16}
                feats_list.append(feat_dict)
            else:
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = col
                # for this example, we do not have transform for features
                feats_list.append(feat_dict)
        # set up the rest fileds of this node type
        if feats_list:
            node_obj['features'] = feats_list
        if labels_list:
            node_obj['labels'] = labels_list

        node_jsons.append(node_obj)

    # generate edge json object
    edge_jsons = []
    for (canonical_etype, edge_df) in edge_dict.items():
        edge_obj = {}
        edge_obj['relation'] = canonical_etype
        edge_obj['format'] = {'name': 'parquet'}       # In this example, we just use parquet
        edge_obj['files'] = edge_file_paths[canonical_etype]

        labels_list = []
        feats_list = []
        src_ntype, etype, dst_ntype = canonical_etype
        # check all dataframe columns
        for col in edge_df.columns:
            label_dict = {}
            feat_dict = {}
            if col == 'source_id':
                edge_obj['source_id_col'] = col
            elif col == 'dest_id':
                edge_obj['dest_id_col'] = col
            else:
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = col
                # for this example, we do not have transform for features
                feats_list.append(feat_dict)
        # set up the rest fileds of this node type
        if feats_list:
            edge_obj['features'] = feats_list
        if labels_list:
            edge_obj['labels'] = labels_list

        edge_jsons.append(edge_obj)

    # generate the configuration JSON file
    data_json = {}
    data_json['version'] = 'gconstruct-v0.1'
    data_json['nodes'] = node_jsons
    data_json['edges'] = edge_jsons

    # save raw data configration JSON
    json_file_path = os.path.join(save_path, 'config.json')

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_json, f, indent=4)


def main(args):
    # create node types
    ntypes = np.array([f'nt_{i:04d}' for i in range(args.num_ntypes)])

    # generate all subgraphs
    sg_list = []
    sg_nids = []
    for i in range(args.num_subgraphs):
        sg_id = f's{i}'
        sg_nids.append(sg_id)
        sub_can_etypes = random_generate_edge_types(ntypes)
        sg_node_dict, sg_edge_dict = gengerate_subgraph(sub_can_etypes,
                                                        sg_id, 
                                                        args.max_num_nodes,
                                                        args.min_num_nodes)
        sg_list.append((sg_node_dict, sg_edge_dict))

    # form collective node lists and edge lists in the form of dictionary
    nids_dict = {}
    eids_dict = {}
    for sg_node_dict, sg_edge_dict in sg_list:
        # collect nodes in a subgraph
        for ntype, nids in sg_node_dict.items():
            if nids_dict.get(ntype) is not None:
                nid_list = nids_dict.get(ntype)
                nid_list.append(nids)
                nids_dict[ntype] = nid_list
            else:
                nid_list = []
                nid_list.append(nids)
                nids_dict[ntype] = nid_list
        # collect edges in a subgraph
        for can_etype, (src_nids, dst_nids) in sg_edge_dict.items():
            if eids_dict.get(can_etype) is not None:
                src_nid_list, dst_nid_list = eids_dict.get(can_etype)
                src_nid_list.append(src_nids)
                dst_nid_list.append(dst_nids)
                eids_dict[can_etype] = (src_nid_list, dst_nid_list)
            else:
                src_nid_list = []
                dst_nid_list = []
                src_nid_list.append(src_nids)
                dst_nid_list.append(dst_nids)
                eids_dict[can_etype] = (src_nid_list, dst_nid_list)

    # concatenate all node lists
    node_dict = {}
    edge_dict = {}
    for ntype, nid_list in nids_dict.items():
        node_dict[ntype] = np.concatenate(nid_list, axis=0)
    for can_etype, (src_nid_list, dst_nid_list) in eids_dict.items():
        edge_dict[can_etype] = (np.concatenate(src_nid_list, axis=0), np.concatenate(dst_nid_list, axis=0))

    node_dict['super'] = sg_nids

    # generate node features and convert nids and features to a Pandas DataFrame
    node_df_dict = {}
    for ntype, nids in node_dict.items():
        # for super node, generate classification labels
        if ntype == 'super':
            # for supernodes, give an all-zero 16D feature
            n_feat = np.zeros([len(nids), 16])
            n_feat = [n_feat[i] for i in range(len(n_feat))]
            labels = np.random.randint(0, 2, len(nids))
            node_df = pd.DataFrame({
                'nid': nids,
                'n_feat': n_feat,
                'label': labels
            })
        else:
            # for common node types, generate a random 16D floats, uniform between [-1, 1)
            n_feat = (np.random.rand(len(nids), 16) - 0.5) * 2
            n_feat = [n_feat[i] for i in range(len(n_feat))]
            node_df = pd.DataFrame({
                'nid': nids,
                'n_feat': n_feat
            })
        node_df_dict[ntype] = node_df
    # generate edge dataframe dict
    edge_df_dict = {}
    for can_etype, (src_nids, dst_nids) in edge_dict.items():
        edge_df = pd.DataFrame({
            'source_id': src_nids,
            'dest_id': dst_nids
        })
        edge_df_dict[can_etype] = edge_df

    # save graph to local disk
    save_as_raw_table(args.save_path, node_df_dict, edge_df_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dummy supernode graph generator.")
    parser.add_argument('--num-subgraphs', type=int, default='100',
                        help="The number of subgraphs in the dummy graph data. Default is 100.")
    parser.add_argument('--num-ntypes', type=int, default='6',
                        help="The number of node types in the dummy graph data. Default is 6.")
    parser.add_argument('--max-num-nodes', type=int, default='50',
                        help="The maximum number of nodes in a subgraph. Default is 50.")
    parser.add_argument('--min-num-nodes', type=int, default='5',
                        help="The minum number of nodes in a subgraph. Default is 5.")
    parser.add_argument('--save-path', type=str, default='./',
                        help="The path for saving generated graph data. Default is the \'./\'.")

    args = parser.parse_args()
    print(args)
    main(args)
