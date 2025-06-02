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

    Prepare ACM data for node classification task in GraphStorm
"""

import argparse
import scipy.io
import urllib.request
import os
import json
import pickle
import dgl
import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn


def convert_tensor_to_list_arrays(tensor):
    """ Convert Pytorch Tensor to a list of arrays
    
    Since a pandas DataFrame cannot save a 2D numpy array in parquet format, it is necessary to
    convert the tensor (1D or 2D) into a list of lists or a list of array. This converted tensor
    can then be used to build a pandas DataFrame, which can be saved in parquet format. However,
    tensor with a dimension greater than or equal to 3D cannot be processed or saved into parquet
    files.

    Parameters:
    tensor: Pytorch Tensor
        The input Pytorch tensor (1D or 2D) to be converted
    
    Returns:
    list_array: list of numpy arrays
        A list of numpy arrays
    """
    
    np_array = tensor.numpy()
    list_array = [np_array[i] for i in range(len(np_array))]

    return list_array


def add_str_prefix(ntype, node_ids, num_prefix=1):
    """ Add the first N letter as prefix to integer node ids to form string node ids
    """
    if isinstance(node_ids, th.Tensor):
        return np.array([f'{ntype[:num_prefix]}{i}' for i in node_ids.numpy()])
    else:
        return np.array([f'{ntype[:num_prefix]}{i}' for i in node_ids])


def convert_to_lists(values):
    """ Conver values to lists for JSON serializers.
    
    values could be torch Tensors or numpy ndarray.
    """
    if isinstance(values, th.Tensor):
        values = values.numpy()

    if isinstance(values, np.ndarray):
        values = values.tolist()
    
    return values


def create_acm_raw_data(graph,
                        text_feat=None,
                        output_path=None):
    """Generate ACM data from DGL graph object created by create_acm_dgl_graph()

    Convert ACM data into multiple tables and a JSON file that required by the GraphStorm's 
    construction tool for demo purpose.
    
    This graph is based on the DGL graph created by the create_acm_dgl_graph() function. Because we
    only use three relationships in the original ACM data, the number of graph nodes could be less 
    than the papers, authors, and subjects in the original node lists.
    
    In addition, to demonstrate the use of string type node ids in the raw graph data, we add the 
    first letter of each node type name to the original numerical ids, i.e., "author" -> "a", 
    "paper" -> "p", and "subject" -> "s".

    Parameters
    ----------
    graph : DGL heterogeneous graph
        The generated dgl.heterograph object.
    text_feat: dict
        The raw text of "paper", "author", and "subject" nodes.
            For "paper" nodes, the text is paper's title plus abstract;
            For "author" nodes, the text is author's full name;
            For "subject" nodes, the text is the ACM's subject code, e.g., "A.0".
    output_path: str
        The folder path to save output files

    Returns
    -------
        No return values, but save generated files into the given output_path
    """

    # generate node dataframe: we use the graph node ids and node name as node_type
    node_list = []

    for ntype in graph.ntypes:
        node_dict = {}
        # generate the id column
        node_ids = graph.nodes(ntype)
        # pad a prefix before each node id
        str_node_ids = add_str_prefix(ntype, node_ids)
        
        node_dict['node_id'] = str_node_ids

        # generate the feature columns and label column
        if graph.nodes[ntype].data:
            for feat_name, val in graph.nodes[ntype].data.items():
                # Here we just hard code the 'label' string
                if feat_name == 'label':
                   # convert tensor to list of arrays for saving in parquet format
                    node_dict[feat_name] = convert_tensor_to_list_arrays(val)
                    continue
                # Here we assume all others are node features
                # convert tensor to list of arrays for saving in parquet format
                node_dict[feat_name] = convert_tensor_to_list_arrays(val)

        # generate the raw text features column
        if text_feat is not None:
            node_dict['text'] = text_feat[ntype]

        # generate the pandas DataFrame that combine ids, and, if have, features and labels
        node_df = pd.DataFrame(node_dict)
        print(f'{ntype} nodes have: {node_df.columns} columns ......')
        # add node type name and node dataframe as a tuple
        node_list.append((ntype, node_df))

    # genreate edge dataframe
    edge_list = []
    
    for src_ntype, etype, dst_ntype in graph.canonical_etypes:
        edge_dict = {}
        # generate the ids columns for both source nodes and destination nodes
        src_ids, dst_ids = graph.edges(etype=(src_ntype, etype, dst_ntype))
        # pad a prefix before each node id
        str_src_ids = add_str_prefix(src_ntype, src_ids)
        str_dst_ids = add_str_prefix(dst_ntype, dst_ids)
        edge_dict['source_id'] = str_src_ids
        edge_dict['dest_id'] = str_dst_ids
        
        # generate feature columns and label col
        if graph.edges[(src_ntype, etype, dst_ntype)].data:
            for feat_name, val in graph.edges[(src_ntype, etype, dst_ntype)].data.items():
                if feat_name == 'label':
                    # Here we just hard code the 'label' string
                    # convert tensor to list of arrays for saving in parquet format
                    edge_dict['label'] = convert_tensor_to_list_arrays(val)
                    continue
                # Here we assume all others are edge features
                # convert tensor to list of arrays for saving in parquet format
                edge_dict[feat_name] = convert_tensor_to_list_arrays(val)

        # Give ('paper', 'citing', 'paper') edge type categorical features for different tasks
        if src_ntype == 'paper' and etype == 'citing' and dst_ntype == 'paper':
            cates = ['C_' + str(i) for i in range(1, 17, 3)]
            num_pvp_edges = graph.num_edges((src_ntype, etype, dst_ntype))
            edge_dict['cate_feat'] = np.random.choice(cates, num_pvp_edges)

        # generate the pandas DataFrame that combine ids, and, if have, features and labels
        edge_df = pd.DataFrame(edge_dict)
        # add canonical edge type name and edge dataframe as a tuple
        edge_list.append(((src_ntype, etype, dst_ntype), edge_df))
    
    # output raw data files
    node_base_path = os.path.join(output_path, 'nodes')
    if not os.path.exists(node_base_path):
        os.makedirs(node_base_path)
    # save node data files
    node_file_paths = {}
    for (ntype, node_df) in node_list:
        node_file_path = os.path.join(node_base_path, ntype + '.parquet')
        node_df.to_parquet(node_file_path)
        node_file_paths[ntype]= [node_file_path]
        print(f'Saved {ntype} node data to {node_file_path}.')

    edge_base_path = os.path.join(output_path, 'edges')
    if not os.path.exists(edge_base_path):
        os.makedirs(edge_base_path)
    # save edge data files
    edge_file_paths = {}
    for (canonical_etype, edge_df) in edge_list:
        src_ntype, etype, dst_ntype = canonical_etype
        edge_file_name = src_ntype + '_' + etype + '_' + dst_ntype
        edge_file_path = os.path.join(edge_base_path, edge_file_name + '.parquet')
        edge_df.to_parquet(edge_file_path)
        edge_file_paths[canonical_etype] = [edge_file_path]
        print(f'Saved {canonical_etype} edge data to {edge_file_path}')

    # generate node json object
    node_jsons = []
    for (ntype, node_df) in node_list:
        node_dict = {}
        node_dict['node_type'] = ntype
        node_dict['format'] = {'name': 'parquet'}       # In this example, we just use parquet
        node_dict['files'] = node_file_paths[ntype]

        labels_list = []
        feats_list = []
        # check all dataframe columns
        for col in node_df.columns:
            label_dict = {}
            feat_dict = {}
            if col == 'node_id':
                node_dict['node_id_col'] = col
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
            node_dict['features'] = feats_list
        if labels_list:
            node_dict['labels'] = labels_list
        
        node_jsons.append(node_dict)

    # generate edge json object
    edge_jsons = []
    for (canonical_etype, edge_df) in edge_list:
        edge_dict = {}
        edge_dict['relation'] = canonical_etype
        edge_dict['format'] = {'name': 'parquet'}       # In this example, we just use parquet
        edge_dict['files'] = edge_file_paths[canonical_etype]

        labels_list = []
        feats_list = []
        src_ntype, etype, dst_ntype = canonical_etype
        # check all dataframe columns
        for col in edge_df.columns:
            label_dict = {}
            feat_dict = {}
            if col == 'source_id':
                edge_dict['source_id_col'] = col
            elif col == 'dest_id':
                edge_dict['dest_id_col'] = col
            elif col == 'label':
                label_dict['task_type'] = 'link_prediction'     # In ACM data, we do not have this
                                                                # edge task. Here is just for demo
                label_dict['split_pct'] = [0.8, 0.1, 0.1]       # Same as the label_split filed.
                                                                # The split pct values are just for
                                                                # demonstration purpose.
                labels_list.append(label_dict)
            elif col.startswith('cate_'):                       # Dummy categorical features that ask
                feat_dict['feature_col'] = col                  # for a "to_categorical" tranformation
                feat_dict['feature_name'] = col                 # operation
                feat_dict['transform'] = {"name": "to_categorical"}
                feats_list.append(feat_dict)
            else:
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = col
                feats_list.append(feat_dict)
        # set up the rest fileds of this node type
        if feats_list:
            edge_dict['features'] = feats_list
        if labels_list:
            edge_dict['labels'] = labels_list
        
        edge_jsons.append(edge_dict)
        
    # generate the configuration JSON file
    data_json = {}
    data_json['version'] = 'gconstruct-v0.1'
    data_json['nodes'] = node_jsons
    data_json['edges'] = edge_jsons
        
    # output configration JSON
    json_file_path = os.path.join(output_path, 'config.json')
        
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_json, f, indent=4)


def create_acm_dgl_graph(dowload_path='/tmp/ACM.mat',
                         dataset_name=None,
                         is_split=False,
                         output_path=None):
    """Create ACM graph data from URL downloading.
    1. Assign paper nodes with a random 256D feature;
    2. No edge features
    
    Parameters
    ----------
    download_path: str
        The file path to save download ACM data
    dataset_name: str
        The graph name.
    is_split: boolean
        Decide if split the label and generate train/val/test mask feature on paper nodes.
    output_path: str
        The folder path to save output DGL graph

    Returns
    -------
    graph_acm: DGL graph
        Return the generated DGL graph, and save it to the given output_path
        - The graph has three types of nodes, "paper", "author", and "subject", and six types of
        edges, including reversed edge types. 
        - Each node have a 256 dimension random feature, and raw text, which is stored in the
        text_feat dictionary.
        - For "paper" nodes, each has a label that is the category of a paper, coming from the
        PvsC relation. There are total 14 classes for paper nodes.
    text_feat: dict
        The raw text of "paper", "author", and "subject" nodes.
            For "paper" nodes, the text is paper's title plus abstract;
            For "author" nodes, the text is author's full name;
            For "subject" nodes, the text is the ACM's subject code, e.g., "A.0".
    """
    if not os.path.exists(dowload_path):
        data_url = 'https://data.dgl.ai/dataset/ACM.mat'
        urllib.request.urlretrieve(data_url, dowload_path)

    data = scipy.io.loadmat(dowload_path)
    graph_acm = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
    })

    # Extract text features from paper, author, and subject nodes.
    paper_text = []
    for paper in data['P']:
        paper_text.append(paper[0][0])
    author_text = []
    for author in data['A']:
        author_text.append(author[0][0])
    subject_text = []
    for subject in data['L']:
        subject_text.append(subject[0][0])

    text_feat = {
        'paper': paper_text,
        'author': author_text,
        'subject': subject_text
    }

    pvc = data['PvsC'].tocsr()
    p_selected = pvc.tocoo()
    # generate labels
    target_ntype = 'paper'
    labels = pvc.indices
    labels = th.tensor(labels).long()
    graph_acm.nodes[target_ntype].data['label'] = labels

    # generate train/val/test split and assign them to the paper nodes
    if is_split:
        pid = p_selected.row
        shuffle = np.random.permutation(pid)
        train_idx = th.tensor(shuffle[0:800]).long()
        val_idx = th.tensor(shuffle[800:900]).long()
        test_idx = th.tensor(shuffle[900:]).long()
        train_mask = th.zeros(pid.shape[0]).long()
        train_mask[train_idx] = 1
        val_mask = th.zeros(pid.shape[0]).long()
        val_mask[val_idx] = 1
        test_mask = th.zeros(pid.shape[0]).long()
        test_mask[test_idx] = 1
        graph_acm.nodes['paper'].data['train_mask'] = train_mask
        graph_acm.nodes['paper'].data['val_mask'] = val_mask
        graph_acm.nodes['paper'].data['test_mask'] = test_mask

    # Give all nodes a 256D random values as their feature.
    for n_type in graph_acm.ntypes:
        emb = nn.Parameter(th.Tensor(graph_acm.number_of_nodes(n_type), 256), requires_grad = False)
        nn.init.xavier_uniform_(emb)
        graph_acm.nodes[n_type].data['feat'] = emb

    # For link prediction task, use "paper, citing, paper" edges as targe-etype and create labels.
    target_etype = ('paper', 'citing', 'paper')
    graph_acm.edges[target_etype].data['label'] = th.ones(graph_acm.num_edges(target_etype))

    print(graph_acm)
    print(f'\n Number of classes: {labels.max() + 1}')
    print(f'\n Paper node labels: {labels.shape}')
    print(f'\n {target_etype} edge labels:{graph_acm.num_edges(target_etype)}')
    
    # Save the graph for later partition
    if dataset_name is None:
        dataset_name = 'acm'
    if output_path is None:
        output_path = '/tmp'

    if output_path is not None:
        # Save DGL graph
        output_graph_file_path = os.path.join(output_path, dataset_name + '.dgl')
        print(f'Saving ACM data to {output_graph_file_path} ......')
        dgl.save_graphs(output_graph_file_path, [graph_acm], None)
        print(f'{output_graph_file_path} saved.')
        # Save raw node text
        output_text_file_path = os.path.join(output_path, dataset_name + '_text.pkl')
        print(f'Saving ACM node text to {output_text_file_path} ......')
        with open(output_text_file_path, 'wb') as tfile:
            pickle.dump(text_feat, tfile)
        print(f'{output_text_file_path} saved.')

    return graph_acm, text_feat


def create_acm_sub_graph(graph, num_targets=2, num_hops=2, fanout=20):
    """ Generate a subgraph with a few randomly selected target nodes from the given graph.
    
    For real-time inference example, this function will generate a subgraph based on a few
    randomly selected target nodes. The number of target nodes is specified by the num_targets
    input argument.

    The detailed method of subgraph extraction is based on full neighbor sampling in at least
    two hops or greater with the num_hos argument.

    """
    assert num_targets > 0 and num_targets < 10, 'The number of target nodes should be greater ' \
                                                 'than 0 and less than 10, but got {num_targets}.'
    assert num_hops >= 2, f'Must have at least two hops to sample a subgraph, but got {num_hops}.'
    assert fanout > 0

    # Give ('paper', 'citing', 'paper') edge type categorical features for different tasks
    cates = [i for i in range(1, 17, 3)]
    num_pvp_edges = graph.num_edges(('paper', 'citing', 'paper'))
    graph.edges[('paper', 'citing', 'paper')].data['cate_feat'] = \
        th.from_numpy(np.random.choice(cates, num_pvp_edges))

    print(graph.edges[('paper', 'citing', 'paper')].data)

    # find one node type having labels associated
    target_ntype = None
    for ntype in graph.ntypes:
        # use 'label' to check if an ntype is a target ntype. Assume at least one ntype has
        if 'label' in graph.nodes[ntype].data.keys():
            # use the last one for target ntype
            target_ntype = ntype

    assert target_ntype is not None, 'The given graph should have at least one node type ' \
                                     'with label associated, but got None.'
    # get target nids
    nids = np.random.choice(np.arange(graph.num_nodes(target_ntype)), [num_targets], replace=False)
    target_nid = {target_ntype: nids}
    target_labels = graph.nodes[target_ntype].data['label'][nids]

    sampler = dgl.dataloading.NeighborSampler([fanout] * num_hops)
    dataloader = dgl.dataloading.DataLoader(g, target_nid, sampler,
        batch_size=num_targets, shuffle=True, drop_last=False)

    all_nodes = []
    for input_nodes, _, blocks in dataloader:
        # as batch size equals to num_targets, will only have one batch
        all_nodes = input_nodes

    if len(all_nodes) > 0:
        subgraph = dgl.node_subgraph(graph, all_nodes)

    return subgraph, target_nid, target_labels
    

def convert_graph_to_json(graph, target_ids, target_labels=None):
    """ convert a DGL graph into a JSON object following GraphStorm real-time API specifications
    
    This function provide a utility method to generate the JSON object for real-time inference
    requests. Detailed JSON specification can be found in GraphStorm real-time inference user
    guide.
    TODO(james): provide the document link here.
    
    ** This JSON specification sets one node or edge in one object. **
    """
    graph_json = {}
    graph_json['version'] = 'gs-realtime-v0.1'
    graph_json['gml_task'] = 'node_classification'
    graph_json['graph'] = {'nodes':[], 'edges':[]}
    graph_json['targets'] = {}

    # iterate each node type and convert nodes to json objects
    nodes = []
    for ntype in graph.ntypes:
        # initialize the node list of this node type
        ntype_nodes = [{'node_type': ntype, 'features': {}} for _ in range(graph.num_nodes(ntype))]
        for key, vals in graph.nodes[ntype].data.items():
            if key == dgl.NID:
                org_nids = graph.nodes[ntype].data[dgl.NID]
                str_org_nids = add_str_prefix(ntype, org_nids)
                assert str_org_nids.shape[0] == len(ntype_nodes), 'The number of node ids should ' \
                                                                  'equal to the number of nodes.'
                for i, str_org_id in enumerate(str_org_nids):
                    ntype_nodes[i]['node_id'] = str_org_id
            elif key == 'label':
                continue
            else:
                assert vals.shape[0] == len(ntype_nodes), 'The number of features should equal ' \
                                                          'to the number of nodes.'
                for i, val in enumerate(vals):
                    ntype_nodes[i]['features'][key] = convert_to_lists(val)
        nodes.extend(ntype_nodes)

    # set graph json node objects
    graph_json['graph']['nodes'] = nodes

    # iterate each edge type and convert edges to json objects
    edges = []
    for src_ntype, etype, dst_ntype in graph.canonical_etypes:
        # initialize the edges of this edge type
        etype_edges =[{'edge_type':(src_ntype, etype, dst_ntype), 'features': {}} for _ in \
                        range(graph.num_edges(etype=(src_ntype, etype, dst_ntype)))]
        # extract src node ids and dst node ids
        src_nids, dst_nids, _ = graph.edges(form='all', etype=(src_ntype, etype, dst_ntype))
        org_src_nids = graph.nodes[src_ntype].data[dgl.NID][src_nids]
        org_dst_nids = graph.nodes[dst_ntype].data[dgl.NID][dst_nids]
        str_org_src_nids = add_str_prefix(src_ntype, org_src_nids)
        str_org_dst_nids = add_str_prefix(dst_ntype, org_dst_nids)
        # set node ids to node objects
        assert str_org_src_nids.shape[0] == len(etype_edges), 'The number of source nodes should' \
                                                              ' equal to the number of edges.'
        assert str_org_dst_nids.shape[0] == len(etype_edges), 'The number of destination nodes ' \
                                                              'should equal to the number of ' \
                                                              'edges.'
        for i, (str_org_src_nid, str_org_dst_nid) in enumerate(zip(str_org_src_nids,
                                                                   str_org_dst_nids)):
            etype_edges[i]['src_node_id'] = str_org_src_nid
            etype_edges[i]['dest_node_id'] = str_org_dst_nid
        # set features of edge if there is
        for key, vals in graph.edges[(src_ntype, etype, dst_ntype)].data.items():
            if key in ['label', dgl.EID]:
                continue
            else:
                assert vals.shape[0] == len(etype_edges), 'The number of features should equal ' \
                                                          'to the number of edges.'
                for i, val in enumerate(vals):
                    etype_edges[i]['features'][key] = convert_to_lists(val)

        edges.extend(etype_edges)
    
    # set graph json edge object
    graph_json['graph']['edges'] = edges

    # process targets
    targets = []
    for key, ids in target_ids.items():
        if isinstance(key, str):
            # node type
            str_ids = add_str_prefix(key, ids)
            for str_id in str_ids:
                node = {
                    'node_type': key,
                    'node_id': str_id
                }
                targets.append(node)
        elif isinstance(key, tuple) and len(key) == 3:
            # edge type
            assert len(ids[0]) == 2, 'The edge ids should be a list of two-element tuples.'
            src_ids, dst_ids = ids
            str_src_ids = add_str_prefix(key[0], src_ids)
            str_dst_ids = add_str_prefix(key[2], dst_ids)
            for str_src_id, str_dst_id in zip(str_src_ids, str_dst_ids):
                edge = {
                    'edge_type': key,
                    'src_node_id': str_src_id,
                    'dest_node_id': str_dst_id
                }
                targets.append(edge)
        else:
            continue
    
    # set target objects
    assert len(targets) > 0, 'For inference, there must be some target nodes or edges, but got 0.'
    graph_json['targets'] = targets

    return graph_json


def convert_graph_to_json_vec(graph, target_ids, target_labels=None):
    """ convert a DGL graph into a JSON object following GraphStorm real-time API specifications
    
    This function provide a utility method to generate the JSON object for real-time inference
    requests. Detailed JSON specification can be found in GraphStorm real-time inference user
    guide.
    TODO(james): provide the document link here.

    ** This JSON specification sets one node or edge TYPE in one object with vectorization. **
    """
    graph_json = {}
    graph_json['version'] = 'gs-realtime-v0.1'
    graph_json['gml_task'] = 'node_classification'
    graph_json['graph'] = {'nodes':[], 'edges':[]}
    graph_json['targets'] = {}

    # iterate each node type and convert nodes to a json object
    type_nodes = []
    for ntype in graph.ntypes:
        type_node = {}
        type_node['node_type'] = ntype
        type_node['features'] = {}

        for key, vals in graph.nodes[ntype].data.items():
            if key == dgl.NID:
                org_nids = graph.nodes[ntype].data[dgl.NID]
                str_org_nids = add_str_prefix(ntype, org_nids)
                type_node['node_ids'] = convert_to_lists(str_org_nids)
            elif key == 'label':
                continue
            else:
                type_node['features'][key] = convert_to_lists(vals)
        type_nodes.append(type_node)

    # add type_nodes to graph json object
    graph_json['graph']['nodes'] = type_nodes
    
    # iterate each edge type and convert edges to a json object
    type_edges = []
    for src_ntype, etype, dst_ntype in graph.canonical_etypes:
        type_edge = {}
        type_edge['edge_type'] = (src_ntype, etype, dst_ntype)
        type_edge['features'] = {}

        # extract src node ids and dst node ids
        src_nids, dst_nids, _ = graph.edges(form='all', etype=(src_ntype, etype, dst_ntype))
        org_src_nids = graph.nodes[src_ntype].data[dgl.NID][src_nids]
        org_dst_nids = graph.nodes[dst_ntype].data[dgl.NID][dst_nids]
        str_org_src_nids = add_str_prefix(src_ntype, org_src_nids)
        str_org_dst_nids = add_str_prefix(dst_ntype, org_dst_nids)
        type_edge['src_node_ids'] = convert_to_lists(str_org_src_nids)
        type_edge['dest_node_ids'] = convert_to_lists(str_org_dst_nids)

        for key, vals in graph.edges[(src_ntype, etype, dst_ntype)].data.items():
            if key in ['label', dgl.EID]:
                continue
            else:
                type_edge['features'][key] = convert_to_lists(vals)

        type_edges.append(type_edge)

    # add type_edges to graph json object
    graph_json['graph']['edges'] = type_edges

    # process targets
    targets = []
    for key, ids in target_ids.items():
        if isinstance(key, str):
            # node type
            str_ids = add_str_prefix(key, ids)
            type_node = {
                    'node_type': key,
                    'node_ids': convert_to_lists(str_ids)
                }
            targets.append(type_node)
        elif isinstance(key, tuple) and len(key) == 3:
            # edge type
            assert len(ids[0]) == 2, 'The edge ids should be a list of two-element tuples.'
            src_ids, dst_ids = ids
            str_src_ids = add_str_prefix(key[0], src_ids)
            str_dst_ids = add_str_prefix(key[2], dst_ids)
            type_edge = {
                    'edge_type': key,
                    'src_node_ids': convert_to_lists(str_src_ids),
                    'dest_node_ids': convert_to_lists(str_dst_ids)
                }
            targets.append(type_edge)
        else:
            continue
    
    # set target objects
    assert len(targets) > 0, 'For inference, there must be some target nodes or edges, but got 0.'
    graph_json['targets'] = targets

    return graph_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Prepare ACM data for using GraphStorm")
    
    parser.add_argument('--download-path', type=str, default='/tmp/ACM.mat',
                        help="The path of folder to store downloaded ACM raw data")
    parser.add_argument('--dataset-name', type=str, default='acm',
                        help="The given name of the graph. Default: \'acm\'.")
    parser.add_argument('--output-type', type=str,
                        choices=['dgl', 'raw', 'raw_w_text', 'subg_json'], default='raw',
                        help="The output graph data type. It could be in DGL heterogeneous graph \
                              that can be used for partition; Or in a specific raw format that \
                              could be used for the GraphStorm\'s graph construction script; Or \
                              in raw format and also include text contexts on all three node \
                              types; Or a subgraph in JSON format sampled by two target nodes \
                              randomly selected. Default is \'raw\'.")
    parser.add_argument('--output-path', type=str, required=True,
                        help="The path of folder to store processed ACM data.")

    args = parser.parse_args()
    print(args)

    # call create_acm_graph to build graph 
    if args.output_type == 'dgl':
        create_acm_dgl_graph(dowload_path=args.download_path,
                             dataset_name=args.dataset_name,
                             is_split=True,
                             output_path=args.output_path)
    elif args.output_type == 'raw':
        g, _ = create_acm_dgl_graph(dowload_path=args.download_path,
                                    is_split=False,
                                    dataset_name=args.dataset_name)
        create_acm_raw_data(graph=g,
                            text_feat=None,
                            output_path=args.output_path)
    elif args.output_type == 'raw_w_text':
        g, text_feat = create_acm_dgl_graph(dowload_path=args.download_path,
                                            is_split=False,
                                            dataset_name=args.dataset_name)
        create_acm_raw_data(graph=g,
                            text_feat=text_feat,
                            output_path=args.output_path)
    elif args.output_type == 'subg_json':
        g, _ = create_acm_dgl_graph(dowload_path=args.download_path,
                                    is_split=False,
                                    dataset_name=args.dataset_name)
        subgraph, target_nid, target_labels = create_acm_sub_graph(g, num_targets=2)
        subgraph_json = convert_graph_to_json(subgraph, target_nid, target_labels)

        subgraph_json_path = os.path.join(args.output_path, args.dataset_name+'_subg.json')
        with open(subgraph_json_path, 'w') as f:
            json.dump(subgraph_json, f, indent=4)