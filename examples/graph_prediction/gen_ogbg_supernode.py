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

    Processor for adding super nodes to OGBG dataset
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from ogb.graphproppred import DglGraphPropPredDataset

SUPPORTED_OGBG_DATAS = ['molhiv']


# save split files
def output_index_json(file_path, ids):
    with open(file_path, 'w', encoding="utf-8") as f:
        for id in ids:
            f.write(json.dumps(id) + "\n")


def generate_ogbg_config_json(node_metadata, edge_metadata):
    # node json objects
    node_jsons = []
    for ntype, metadata in node_metadata.items():
        node_dict = {}
        node_dict['node_type'] = ntype
        node_dict['format'] = {'name': 'parquet'}
        node_dict['files'] = metadata['data_path']

        labels_list = []
        feats_list = []
        for col in metadata['feat_names']:
            label_dict = {}
            feat_dict = {}
            if col == 'NID':
                node_dict['node_id_col'] = col
            elif col == 'labels':
                label_dict['label_col'] = col
                label_dict['task_type'] = 'classification'  # This version just assumes this task
                if metadata.get('custom_split_filenames', None) is not None:
                    # Have custom split files
                    label_dict['custom_split_filenames'] = metadata.get('custom_split_filenames')
                else:
                    label_dict['split_pct'] = [0.8, 0.1, 0.1]
                label_dict['label_stats_type'] = 'frequency_cnt'
                labels_list.append(label_dict)
            else:
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = col
                feats_list.append(feat_dict)

        # set up the rest fileds of this node type
        if feats_list:
            node_dict['features'] = feats_list
        if labels_list:
            node_dict['labels'] = labels_list

        node_jsons.append(node_dict)
        
    # edge json objects
    edge_jsons = []
    for canonical_etype, metadata in edge_metadata.items():

        edge_dict = {}
        edge_dict['relation'] = canonical_etype
        edge_dict['format'] = {'name': 'parquet'}
        edge_dict['files'] = metadata['data_path']

        labels_list = []
        feats_list = []
        # check all dataframe columns
        for col in metadata['feat_names']:
            label_dict = {}
            feat_dict = {}
            if col == 'source_id':
                edge_dict['source_id_col'] = col
            elif col == 'dest_id':
                edge_dict['dest_id_col'] = col
            else:
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = col
                feats_list.append(feat_dict)

        if feats_list:
            edge_dict['features'] = feats_list
        if labels_list:
            edge_dict['labels'] = labels_list
        
        edge_jsons.append(edge_dict)
        
    # generate the configuration JSON object

    data_json = {}
    data_json['version'] = 'gconstruct-v0.1'
    data_json['nodes'] = node_jsons
    data_json['edges'] = edge_jsons

    return data_json


def main(args):
    """ Main tranformation function
    
    Downlaod OGBG data, and add a super node to each individual graph. Then reconstruct
    each individual graph with new node IDs, i.e., a graph ID + original node ID. The
    super node will have inbound edges from all nodes in an individual graph that the super
    node will represent. Finally, all individual graphs are considered as one giant graph
    with many disconnnected components (subgraphs). This giant graph then is saved in the
    table format for GraphStorm graph construction.
    
    Notes:
        1. The code only handle ogbg graph with one node type and one edge type.
        2. We keep the original node/edge features in the new giant graph.
        3. We give all zeros node features to super nodes, so that they will only
           aggregate messages from the original individual graph without its own
           representation upddated.
    
    """
    # download and process the original OGBN data
    print(f'\n============= Download and Process OGBG Data: {args.ogbg_data_name} =============')
    data_name = 'ogbg-' + args.ogbg_data_name
    dataset = DglGraphPropPredDataset(name = data_name)

    # get subgrahps and their labels
    gs, lbls = dataset.graphs, dataset.labels

    # build node list
    new_nids = []
    origin_nfeats = []
    new_srcs = []
    new_dsts = []
    subg_ids = []

    # assign original subgraph a graph IDs
    print('\n============= Create Super Node Data =============')
    for i, g in enumerate(gs):
        # create subgraph IDs
        sg_id = f's{i}'
        subg_ids.append(sg_id)

        # extract original IDs and create new node IDs
        new_nid = [f'{sg_id}_N_{i:06d}' for i in g.nodes()]
        # extract original node features
        new_nfeats = g.ndata['feat'].numpy()

        new_nids.append(new_nid)
        origin_nfeats.append(new_nfeats)
        # extract original edge list and create new edge list
        src, dst = g.edges()
        new_src = [f'{sg_id}_N_{i:06d}' for i in src]
        new_dst = [f'{sg_id}_N_{i:06d}' for i in dst]
        new_srcs.append(new_src)
        new_dsts.append(new_dst)

    # generate new node tables
    ntype1 = np.concatenate(new_nids)
    new_nfeats = np.concatenate(origin_nfeats)
    srcs = np.concatenate(new_srcs)
    dsts = np.concatenate(new_dsts)
    subg_ids = np.array(subg_ids)

    print(f'Totally there are {subg_ids.shape[0]} subgraphs, including ' + \
          f'{ntype1.shape[0]} nodes, and {srcs.shape[0]} edges.')

    # convert to pandas dataframes
    new_nfeats = [row for row in new_nfeats]
    ntype1_df = pd.DataFrame({
        'NID': ntype1,
        'n_feat': new_nfeats
    })
    etype1_df = pd.DataFrame({
        'source_id': srcs,
        'dest_id': dsts
    })

    # generate super node dataframe
    labels = lbls.squeeze().numpy()
    zero_feat = [row for row in np.zeros([labels.shape[0], 16])]        # Set initial 16D zero feature

    super_df = pd.DataFrame({
        'NID': subg_ids,
        'n_feat': zero_feat,
        'labels': labels
    })

    # generate the to_super edges
    sg_srcs = []
    sg_dsts = []
    for sg_nid, origin_nid in zip(subg_ids, new_nids):
        sg_srcs.append(origin_nid)
        sg_dsts.append([sg_nid] * len(origin_nid))

    sg_src = np.concatenate(sg_srcs)
    sg_dst = np.concatenate(sg_dsts)

    # generate the to_super edge DataFrame
    to_super_df = pd.DataFrame({
        'source_id': sg_src,
        'dest_id': sg_dst
    })

    # process original splits, and regenerate the train/val/test split jsons
    print('\n============= Process OGB Splits and Assign to Super Nodes =============')
    split_base_path = './dataset/ogbg_molhiv/split/scaffold/'
    train_idx_path = os.path.join(split_base_path, 'train.csv.gz')
    val_idx_path = os.path.join(split_base_path, 'valid.csv.gz')
    test_idx_path = os.path.join(split_base_path, 'test.csv.gz')

    train_idx = pd.read_csv(train_idx_path, header=None)
    val_idx = pd.read_csv(val_idx_path, header=None)
    test_idx = pd.read_csv(test_idx_path, header=None)

    train_idx.rename(columns={0:'idx'}, inplace=True)
    val_idx.rename(columns={0:'idx'}, inplace=True)
    test_idx.rename(columns={0:'idx'}, inplace=True)

    # add prefix to split index
    train_idx['NID'] = 's' + train_idx['idx'].apply(str)
    val_idx['NID'] = 's' + val_idx['idx'].apply(str)
    test_idx['NID'] = 's' + test_idx['idx'].apply(str)

    print(f'The OGB split information is {train_idx.shape[0]} subgraphs in training, ' + \
          f'{val_idx.shape[0]} subgraphs in validation, and {test_idx.shape[0]} ' + \
          'subgraphs in testing.')

    # prepare ndoe and edge dictionary
    gs_data_base_path = args.output_path
    node_base_path = os.path.join(gs_data_base_path, 'nodes')
    edge_base_path = os.path.join(gs_data_base_path, 'edges')

    ntype_dict = {'node': ntype1_df, 'super': super_df}
    etype_dict = {('node', 'to', 'node'): etype1_df, ('node', 'to_super', 'super'): to_super_df}

    # save node dataframes and create node_metadata
    print('\n============= Save Processed OGBG Data =============')
    node_metadata = {}
    for ntype, node_df in ntype_dict.items():
        ntype_base_path = os.path.join(node_base_path, ntype)
        os.makedirs(ntype_base_path, exist_ok=True)
        node_file_path = os.path.join(ntype_base_path, ntype + '.parquet')
        node_df.to_parquet(node_file_path)
        print(f'Saved {ntype} data to {node_file_path} ...')

        node_metadata[ntype] = {'data_path': node_file_path,
                                'feat_names': list(node_df.columns)}

        # Save super nodes split jsons. This is dedicated for the super-node method.
        if ntype == 'super':
            train_id_path = os.path.join(ntype_base_path, 'train_idx.json')
            val_id_path = os.path.join(ntype_base_path, 'val_idx.json')
            test_id_path = os.path.join(ntype_base_path, 'test_idx.json')
            output_index_json(train_id_path, train_idx['NID'].to_numpy())
            output_index_json(val_id_path, val_idx['NID'].to_numpy())
            output_index_json(test_id_path, test_idx['NID'].to_numpy())

            node_metadata[ntype]['custom_split_filenames'] = {'train': train_id_path,
                                                              'valid': val_id_path,
                                                              'test': test_id_path}

    # save edge dataframes and create edge_metadata
    edge_metadata = {}
    for can_etype, edge_df in etype_dict.items():
        etype_base_path = os.path.join(edge_base_path, can_etype[0] + '-' + can_etype[1] + '-' + can_etype[2])
        os.makedirs(etype_base_path, exist_ok=True)
        edge_file_path = os.path.join(etype_base_path, can_etype[0] + '-' + can_etype[1] + '-' + can_etype[2] + '.parquet')
        edge_df.to_parquet(edge_file_path)
        print(f'Save {can_etype} data to {edge_file_path} ...')

        edge_metadata[can_etype] = {'data_path': edge_file_path,
                                    'feat_names': list(edge_df.columns)}
    
    # generate the config json file for GraphStorm graph construction
    json_object = generate_ogbg_config_json(node_metadata, edge_metadata)
    print(f'\nGenerate Config JSON for GraphStorm graph construction:')
    print(json_object)

    # save the config json
    json_file_path = os.path.join(args.output_path, 'config.json')
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_object, f, indent=4)

    print(f'\n============= All Artifacts are Saved at {args.output_path} =============')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Process OGB Graph Data in Super Node Format")
    
    parser.add_argument('--ogbg-data-name', type=str, choices=SUPPORTED_OGBG_DATAS,
                        default='molhiv',
                        help="The OGB grahp data name.")
    parser.add_argument('--output-path', type=str, required=True,
                        help="The path of folder to store processed OGBG data.")
    args = parser.parse_args()
    print(args)

    main(args)