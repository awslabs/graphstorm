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

    Generate example graph data using built-in datasets for node classifcation,
    node regression, edge classification and edge regression.
"""

"""Tools to generate dummy raw graph datasets, used to test single instance
    graph construction pipeline
    Data design ideas:
        1. Heterogenous graphs with three node types and two edge types.
        2. Two node types have random features associated, while one type is featureless.
        3. All edges are featureless.
        4. One node type has labels for only a small number of nodes.
        5. One edge type has labels for some edges.
        6. Another edge type is used to do LP task.
        
    This code should be used for regression test with different variances, such as graph sizes,
    split policies, and different data file formats.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np


NUM_NTYPE1 = 1000
NUM_NTYPE2 = 100
NUM_NTYPE3 = 500
NUM_ETYPE1 = 3000
NUM_ETYPE2 = 9000

def generate_data(args):
    """ Generate node dataframes and edge dataframes based on the given parameters
    Output:
        n_dfs: dataframes for all node types
        e_dfs: dataframes for all edge types
        base_json_data: the basic json object that has all infor except for format and files
    """
    factor = 1
    if args.graph_size == 'tiny':
        factor = factor * 10**0
    elif args.graph_size == 'small':
        factor = factor * 10**2
    elif args.graph_size == 'medium':
        factor = factor * 10**4
    elif args.graph_size == 'big':
        factor = factor * 10**5
    elif args.graph_size == 'large':
        factor = factor * 10**6
        
    n_dfs_dict = {}
    # -------- build node1 table ---------
    # create node id strings 
    n1ids = np.random.choice(np.arange(NUM_NTYPE1 * factor * 2), NUM_NTYPE1 * factor, replace=False)
    n1_ids = ['n1id-' + str(i) for i in n1ids]
    # n1_ids = ids
    # create two feature columns
    f1 = np.random.randn(NUM_NTYPE1 * factor)
    f2 = np.random.randint(0, 10, size=NUM_NTYPE1 * factor)
    # create the node 1 pandas dataframe
    n1_df = pd.DataFrame({'node1-id': n1_ids,
                          'f1': f1,
                          'f2': f2})
    n_dfs_dict['n0'] = n1_df

    # -------- build node2 table ---------
    # create node id strings
    n2ids = np.random.choice(np.arange(NUM_NTYPE2 * factor * 2), NUM_NTYPE2 * factor, replace=False)
    n2_ids = ['n2id-' + str(i) for i in n2ids]
    # n2_ids = ids
    # create three feaure columns
    f1 = np.random.randn(NUM_NTYPE2 * factor)
    f2 = np.random.randint(0, 2, size=NUM_NTYPE2 * factor)
    f3 = np.random.rand(NUM_NTYPE2 * factor)
    # create the node 2 pandas dataframe
    n2_df = pd.DataFrame({'node2-id': n2_ids,
                          'f1': f1,
                          'f2-label': f2,       # labels for all nodes
                          'f3': f3})
    n_dfs_dict['n1'] = n2_df

    # -------- build node3 table ---------
    # create node id strings
    n3ids = np.random.choice(np.arange(NUM_NTYPE3 * factor * 2), NUM_NTYPE3 * factor, replace=False)
    n3_ids = ['n3id-' + str(i) for i in n3ids]
    # n3_ids = ids
    # create the node 3 pandas dataframe
    n3_df = pd.DataFrame({'node3-id': n3_ids})
    n_dfs_dict['n2'] = n3_df
    
    e_dfs_dict = {}
    # -------- build edge1 between node1 and node2 ---------
    e1_src_ids = np.random.choice(n1_ids, NUM_ETYPE1 * factor + 100, replace=True)
    e1_dst_ids = np.random.choice(n2_ids, NUM_ETYPE1 * factor + 100, replace=True)
    e1_df = pd.DataFrame({'src_id': e1_src_ids,
                          'dst_id': e1_dst_ids})
    e1_df = e1_df.drop_duplicates().iloc[ :NUM_ETYPE1 * factor]
    e_dfs_dict[('n0', 'e0', 'n1')] = e1_df

    # -------- build edge2 between node1 and node3 ---------
    e2_src_ids = np.random.choice(n1_ids, NUM_ETYPE2 * factor + 100, replace=True)
    e2_dst_ids = np.random.choice(n3_ids, NUM_ETYPE2 * factor + 100, replace=True)
    e2_df = pd.DataFrame({'src_id': e2_src_ids,
                          'dst_id': e2_dst_ids})
    e2_df = e2_df.drop_duplicates().iloc[ :NUM_ETYPE2 * factor]
    
    # [TODO] James: Keep these commented codes for later test after construct graph fix this issue.
    # elabels = np.random.randint(0, 2, int(e2_df.shape[0] * 0.5))      # labels for half edges
    # e2_df['f1-label'] = np.nan
    # e_idx = np.random.randint(0, e2_df.shape[0], int(e2_df.shape[0] * 0.5))
    # e2_df['f1-label'].iloc[e_idx] = elabels
    
    e_dfs_dict[('n0', 'e1', 'n2')] = e2_df

    return n_dfs_dict, e_dfs_dict

    
def split_files(n_dfs_dict, e_dfs_dict, split_policy='no_split'):
    """
    n_dfs_dict: dict of list
        dictionary of list of pandas dataframes
    e_dfs_dict: dict of list
        dictionary of list of pandas datagrames
    split_policy: str
        policy used to split nodes and edges.
        - Options:
            no_split: do not split nodes and edges. One node type and edge type will be
                      saved to one file only.
            random_split: split one node/edge type to up to three files
    Output:
        n_split_dict: dict of list
            dictionary of splited node dataframe list, key is 'n' + i as node type, value is list
        e_split_dict: dict of list
            dictionary of splited edge dataframe list, key is 'e' + i as edge type, valie is list
    """
    n_split_dict = dict()
    e_split_dict = dict()
    
    if split_policy == 'no_split':
        for ntype, n_df in n_dfs_dict.items():
            n_split_dict[ntype] = [n_df]
        for etype, e_df in e_dfs_dict.items():
            e_split_dict[etype] = [e_df]
    elif split_policy == 'random_split':
        for ntype, n_df in n_dfs_dict.items():
            num_splits = np.random.choice([1,2,3], 1)[0]
            split_interval = n_df.shape[0] // num_splits + 1
            n_df_splits = []
            for i in range(num_splits):
                start_pt = i * split_interval
                end_pt = (i + 1) * split_interval
                n_df_split = n_df.iloc[start_pt: end_pt]
                n_df_splits.append(n_df_split)
            n_split_dict[ntype] = n_df_splits
        
        for etype, e_df in e_dfs_dict.items():
            num_splits = np.random.choice([1,2,3], 1)[0]
            split_interval = e_df.shape[0] // num_splits + 1
            e_df_splits = []
            for i in range(num_splits):
                start_pt = i * split_interval
                end_pt = (i + 1) * split_interval
                e_df_split = e_df.iloc[start_pt: end_pt]
                e_df_splits.append(e_df_split)
            e_split_dict[etype] = e_df_splits
    else:
        raise NotImplementedError(f'Not support the given {split_policy} split policy ...')

    return n_split_dict, e_split_dict


def generate_format_dict(format):
    if format == 'csv':
        format_dict = {'name': 'csv', 'separator': ','}
    elif format == 'json':
        format_dict = {'name': 'json'}
    elif format == 'parquet':
        format_dict = {'name': 'parquet'}
    else:
        raise NotImplementedError('So far only support generating \'csv\', \'json\', \
            \'parquet\' format...')
    return format_dict


def generate_file_paths(type, df_splits, format_name, output_path):
    """ The path should be absolute path
    """
    if isinstance(type, tuple):
        type = "-".join(type)
    base_path = os.path.join(output_path, type)
    file_paths = []
    for i, _ in enumerate(df_splits):
        file_path = os.path.join(base_path, type + '_part' + str(i) + '.' + format_name)
        file_paths.append(file_path)
    return base_path, file_paths


def generate_json_file(n_split_dict, e_split_dict, format='parquet', output_path='./'):
    """based on the node/edge dataframe splits. generate the json object.
        n_split_list: dict of list
        e_split_list: dict of list
        format: str
            saved data format. Options:
            - parquet
            - csv
            - json
    """
    data_json = {}
    
    n_list = []
    for ntype, n_df_splits in n_split_dict.items():
        n_dict = {}
        n_dict['node_type'] = ntype
        n_dict['format'] = generate_format_dict(format)
        _, file_paths = generate_file_paths(ntype, n_df_splits, format, output_path)
        n_dict['files'] = file_paths

        n_df = n_df_splits[0]
        feat_cols = []
        label_cols = []
        for col in n_df.columns:
            label_dict = {}
            feat_dict = {}
            if col.endswith('-id'):
                n_dict['node_id_col'] = col
            elif not col.endswith('-label'):
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = 'feat'
                feat_cols.append(feat_dict)
            else:
                label_dict['label_col'] = col
                label_dict['task_type'] = 'classification'
                label_dict['split_pct'] = [0.2, 0.1, 0.1]
                label_cols.append(label_dict)
        if feat_cols:
            n_dict['features'] = feat_cols
        if label_cols:
            n_dict['labels'] = label_cols
        n_list.append(n_dict)

    data_json['nodes'] = n_list

    e_list = []
    for etype, e_df_splits in e_split_dict.items():
        e_dict = {}
        
        e_df = e_df_splits[0]
        feat_list = []
        label_list = []
        for col in e_df.columns:
            label_dict = {}
            feat_dict = {}
            if col.startswith('src_'):
                e_dict['source_id_col'] = col
            elif col.startswith('dst_'):
                e_dict['dest_id_col'] = col
            elif col.endswith('-label'):
                label_dict['label_col'] = col
                label_dict['task_type'] = 'classification'
                label_dict['split_pct'] = [0.2, 0.2, 0.6]
                label_list.append(label_dict)
            else:
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = 'feat'
                feat_list.append(feat_dict)

        if feat_list:
            e_dict['features'] = feat_list
        if label_list:
            e_dict['labels'] = label_list
        
        e_dict['relation'] = etype
        
        e_dict['format'] = generate_format_dict(format)
        _, file_paths = generate_file_paths(etype, e_df_splits, format, output_path)
        e_dict['files'] = file_paths
        
        e_list.append(e_dict)

    data_json['edges'] = e_list

    return data_json
    

def output_json(json_obj, save_path):
    """ save the json object to a file
    """
    save_json_path = os.path.join(save_path, 'input_data.json')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_obj, f, indent=4)


def output_graph(n_split_dict, e_split_dict, format, output_path):
    """save the node and edge DataFrames to output paths based on the given format
    """
    for ntype, n_df_splits in n_split_dict.items():
        base_path, file_paths = generate_file_paths(ntype, n_df_splits, format, output_path)
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        for i, n_df_split in enumerate(n_df_splits):
            if format == 'csv':
                n_df_split.to_csv(file_paths[i], index=False)
            elif format == 'parquet':
                n_df_split.to_parquet(file_paths[i], index=False)
            elif format == 'json':
                n_df_split.to_json(file_paths[i], orient='records')

    for etype, e_df_splits in e_split_dict.items():
        base_path, file_paths = generate_file_paths(etype, e_df_splits, format, output_path)
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
            
        for i, e_df_split in enumerate(e_df_splits):
            if format == 'csv':
                e_df_split.to_csv(file_paths[i], index=False)
            elif format == 'parquet':
                e_df_split.to_parquet(file_paths[i], index=False)
            elif format == 'json':
                e_df_split.to_json(file_paths[i], orient='records')


def generate(args):
    n_dfs, e_dfs = generate_data(args)
    n_split_dict, e_split_dict = split_files(n_dfs, e_dfs, args.split_policy)
    data_json = generate_json_file(n_split_dict, e_split_dict, 
                                   format=args.data_format, output_path=args.save_path)
    output_json(data_json, args.save_path)
    output_graph(n_split_dict, e_split_dict, args.data_format, args.save_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dummy graphs for pipeline test.')
    parser.add_argument('--graph-name', type=str, default='dummy',
                        help='The name for the generated graph. Default is \'dummy\'')
    parser.add_argument('--data-format', type=str, default='parquet',
                        help='Saved data format. Optioins: \'csv\', \'parquet\', \'json\'. \
                            default: \'parquet\'')
    parser.add_argument('--split-policy', type=str, default='no_split', 
                        choices=['no_split', 'random_split'],
                        help='How to split nodes/edge files. \
                            Choice: \'no_split\', keep all nodes or edges in one type in one file; \
                            \'random_split\', split nodes or edges in one type to up to 3 splits.')
    parser.add_argument('--graph-size', type=str, default='tiny',
                        choices=['tiny', 'small', 'medium', 'big', 'large'],
                        help='How large the generated graph will be. \
                            Choice: \'tiny\', hundreds of nodes and thousands of edges;\
                                \'small\', 100 times of tiny data;\
                                \'medium\', 1K times of tiny;\
                                \'big\', 10K times of tiny;\
                                \'large\', 100k times of tiny')
    parser.add_argument('--save-path', type=str, required=True,
                        help='The path to save the generated graph.')
    
    args = parser.parse_args()
    print(args)
    generate(args)
