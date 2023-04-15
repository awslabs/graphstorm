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

import time
from functools import partial
import multiprocessing
from multiprocessing import Process
import glob
import os
import json
import argparse
import gc
import queue

import numpy as np
from transformers import BertTokenizer
import torch as th
import dgl

from ..utils import sys_tracker
from .file_io import parse_node_file_format, parse_edge_file_format
from .file_io import get_in_files

def process_features(data, ops):
    """ Process the data with the specified operations.

    This function runs the input operations on the corresponding data
    and returns the processed results.

    Parameters
    ----------
    data : dict
        The data stored as a dict.
    ops : list of tuples
        The operations. Each tuple contains two elements. The first element
        is the data name and the second element is a Python function
        to process the data.

    Returns
    -------
    dict : the key is the data name, the value is the processed data.
    """
    new_data = {}
    for feat_col, feat_name, dtype, op in ops:
        # If the transformation is defined on the feature.
        if op is not None:
            res = op(data[feat_col])
            if isinstance(res, dict):
                for key, val in res.items():
                    new_data[key] = val
            else:
                new_data[feat_name] = res
        # If the required data type is defined on the feature.
        elif dtype is not None:
            new_data[feat_name] = data[feat_col].astype(dtype)
        # If no transformation is defined for the feature.
        else:
            new_data[feat_name] = data[feat_col]
    return new_data

def get_valid_label_index(label):
    """ Get the index of the samples with valid labels.

    Some of the samples may not have labels. We require users to use
    NaN to indicate the invalid labels.

    Parameters
    ----------
    label : Numpy array
        The labels of the samples.

    Returns
    -------
    Numpy array : the index of the samples with valid labels in the list.
    """
    if np.issubdtype(label.dtype, np.floating):
        return np.logical_not(np.isnan(label)).nonzero()[0]
    elif np.issubdtype(label.dtype, np.integer):
        return np.arange(len(label))
    else:
        raise ValueError("GraphStorm only supports label data of integers and float." + \
                         f"This label data has data type of {label.dtype}.")

def process_labels(data, label_confs, is_node):
    """ Process labels

    Parameters
    ----------
    data : dict
        The data stored as a dict.
    label_conf : list of dict
        The list of configs to construct labels.
    is_node : bool
        Whether or not to process labels on nodes.

    Returns
    -------
    dict of tensors : labels (optional) and train/validation/test masks.
    """
    assert len(label_confs) == 1, "We only support one label per node/edge type."
    label_conf = label_confs[0]
    assert 'task_type' in label_conf, "'task_type' must be defined in the label field."
    if label_conf['task_type'] == 'classification':
        assert 'label_col' in label_conf, \
                "'label_col' must be defined in the label field."
        label_col = label_conf['label_col']
        label = data[label_col]
        assert np.issubdtype(label.dtype, np.integer) \
                or np.issubdtype(label.dtype, np.floating), \
                "The labels for classification have to be integers."
        valid_label_idx = get_valid_label_index(label)
        label = np.int32(label)
        num_samples = len(label)
    elif label_conf['task_type'] == 'regression':
        assert 'label_col' in label_conf, \
                "'label_col' must be defined in the label field."
        label_col = label_conf['label_col']
        label = data[label_col]
        valid_label_idx = get_valid_label_index(label)
        num_samples = len(label)
    else:
        assert label_conf['task_type'] == 'link_prediction', \
                "The task type must be classification, regression or link_prediction."
        assert not is_node, "link_prediction task must be defined on edges."
        label_col = label = None
        valid_label_idx = None
        # Any column in the data can define the number of samples in the data.
        for val in data.values():
            num_samples = len(val)
            break

    if 'split_pct' in label_conf:
        train_split, val_split, test_split = label_conf['split_pct']
        assert train_split + val_split + test_split <= 1, \
                "The data split of training/val/test cannot be more than the entire dataset."
        if valid_label_idx is None:
            rand_idx = np.random.permutation(num_samples)
        else:
            rand_idx = np.random.permutation(valid_label_idx)
        num_labels = len(rand_idx)
        num_train = int(num_labels * train_split)
        num_val = int(num_labels * val_split)
        num_test = int(num_labels * test_split)
        val_start = num_train
        val_end = num_train + num_val
        test_end = num_train + num_val + num_test
        train_idx = rand_idx[0:num_train]
        val_idx = rand_idx[val_start:val_end]
        test_idx = rand_idx[val_end:test_end]
        train_mask = np.zeros((num_samples,), dtype=np.int8)
        val_mask = np.zeros((num_samples,), dtype=np.int8)
        test_mask = np.zeros((num_samples,), dtype=np.int8)
        train_mask[train_idx] = 1
        val_mask[val_idx] = 1
        test_mask[test_idx] = 1
    if label_col is None:
        return {'train_mask': train_mask,
                'val_mask': val_mask,
                'test_mask': test_mask}
    else:
        return {label_col: label,
                'train_mask': train_mask,
                'val_mask': val_mask,
                'test_mask': test_mask}

def parse_node_data(in_file, feat_ops, node_id_col, label_conf, read_file):
    """ Parse node data.

    The function parses a node file that contains node IDs, features and labels
    The node file is parsed according to users' configuration
    and performs some feature transformation.

    Parameters
    ----------
    in_file : str
        The path of the input node file.
    feat_ops : dict
        The operations run on the node features of the node file.
    node_id_col : str
        The column name that contains the node ID.
    label_conf : dict
        The configuration of labels.
    read_file : callable
        The function to read the node file

    Returns
    -------
    tuple : node ID array and a dict of node feature tensors.
    """
    data = read_file(in_file)
    feat_data = process_features(data, feat_ops) if feat_ops is not None else {}
    if label_conf is not None:
        label_data = process_labels(data, label_conf, True)
        for key, val in label_data.items():
            feat_data[key] = val
    return (data[node_id_col], feat_data)

def map_node_ids(src_ids, dst_ids, edge_type, node_id_map, skip_nonexist_edges):
    """ Map node IDs of source and destination nodes of edges.

    In the ID mapping, we need to handle multiple errors in the input data:
    1) we handle the case that endpoint nodes of edges don't exist;
    2) we handle the case that the data type of node IDs of the endpoint nodes don't
    match the data type of the keys of the ID map.

    Parameters
    ----------
    src_ids : tensor
        The source nodes.
    dst_ids : tensor
        The destination nodes.
    edge_type : tuple
        It contains source node type, relation type, destination node type.
    node_id_map : dict
        The key is the node type and value is IdMap or NoopMap.
    skip_nonexist_edges : bool
        Whether or not to skip edges whose endpoint nodes don't exist.

    Returns
    -------
    tuple of tensors : the remapped source and destination node IDs.
    """
    src_type, _, dst_type = edge_type
    new_src_ids, orig_locs = node_id_map[src_type].map_id(src_ids)
    # If some of the source nodes don't exist in the node set.
    if len(orig_locs) != len(src_ids):
        bool_mask = np.ones(len(src_ids), dtype=bool)
        bool_mask[orig_locs] = False
        if skip_nonexist_edges:
            print(f"source nodes of {src_type} do not exist: {src_ids[bool_mask]}")
        else:
            raise ValueError(f"source nodes of {src_type} do not exist: {src_ids[bool_mask]}")
        dst_ids = dst_ids[orig_locs]
    src_ids = new_src_ids

    new_dst_ids, orig_locs = node_id_map[dst_type].map_id(dst_ids)
    # If some of the dest nodes don't exist in the node set.
    if len(orig_locs) != len(dst_ids):
        bool_mask = np.ones(len(dst_ids), dtype=bool)
        bool_mask[orig_locs] = False
        if skip_nonexist_edges:
            print(f"dest nodes of {dst_type} do not exist: {dst_ids[bool_mask]}")
        else:
            raise ValueError(f"dest nodes of {dst_type} do not exist: {dst_ids[bool_mask]}")
        # We need to remove the source nodes as well.
        src_ids = src_ids[orig_locs]
    dst_ids = new_dst_ids
    return src_ids, dst_ids

def parse_edge_data(in_file, feat_ops, node_id_map, read_file, conf, skip_nonexist_edges):
    """ Parse edge data.

    The function parses an edge file that contains the source and destination node
    IDs, edge features and potentially edge labels. The edge file is parsed
    according to users' configuration and performs some feature transformation.

    Parameters
    ----------
    in_file : str
        The path of the input edge file.
    feat_ops : dict
        The operations run on the edge features of the edge file.
    node_id_map : dict
        Contains the ID mapping for every node type.
    read_file : callable
        The function to read the node file
    conf : dict
        The configuration for parsing edge data.
    skip_nonexist_edges : bool
        Whether or not to skip edges that don't exist.

    Returns
    -------
    a tuple : source ID vector, destination ID vector, a dict of edge feature tensors.
    """
    src_id_col = conf['source_id_col']
    dst_id_col = conf['dest_id_col']
    edge_type = conf['relation']
    label_conf = conf['labels'] if 'labels' in conf else None

    data = read_file(in_file)
    feat_data = process_features(data, feat_ops) if feat_ops is not None else {}
    if label_conf is not None:
        label_data = process_labels(data, label_conf, False)
        for key, val in label_data.items():
            feat_data[key] = val
    src_ids = data[src_id_col]
    dst_ids = data[dst_id_col]
    src_ids, dst_ids = map_node_ids(src_ids, dst_ids, edge_type, node_id_map,
                                    skip_nonexist_edges)
    return (src_ids, dst_ids, feat_data)

def process_node_data(process_confs, convert2ext_mem, remap_id, num_processes=1):
    """ Process node data

    We need to process all node data before we can process edge data.
    Processing node data will generate the ID mapping.

    The node data of a node type is defined as follows:
    {
        "node_id_col":  "<column name>",
        "node_type":    "<node type>",
        "format":       {"name": "csv", "separator": ","},
        "files":        ["<paths to files>", ...],
        "features":     [
            {
                "feature_col":  ["<column name>", ...],
                "feature_name": "<feature name>",
                "data_type":    "<feature data type>",
                "transform":    {"name": "<operator name>", ...}
            },
        ],
        "labels":       [
            {
                "label_col":    "<column name>",
                "task_type":    "<task type: e.g., classification>",
                "split_type":   [0.8, 0.2, 0.0],
                "custom_train": "<the file with node IDs in the train set>",
                "custom_valid": "<the file with node IDs in the validation set>",
                "custom_test":  "<the file with node IDs in the test set>",
            },
        ],
    }

    Parameters
    ----------
    process_confs: list of dicts
        The configurations to process node data.
    convert2ext_mem : ExtMemArrayConverter
        A callable to convert a Numpy array to external-memory Numpy array.
    remap_id: bool
        Whether or not to remap node IDs
    num_processes: int
        The number of processes to process the input files.

    Returns
    -------
    dict: node ID map
    dict: node features.
    """
    node_data = {}
    node_id_map = {}
    for process_conf in process_confs:
        # each iteration is to process a node type.
        # TODO(zhengda) we need to make it optional to support
        # multiple node dictionaries for one node type.
        assert 'node_id_col' in process_conf, \
                "'node_id_col' must be defined for a node type."
        node_id_col = process_conf['node_id_col']
        assert 'node_type' in process_conf, \
                "'node_type' must be defined for a node type"
        node_type = process_conf['node_type']
        assert 'format' in process_conf, \
                "'format' must be defined for a node type"
        read_file = parse_node_file_format(process_conf)
        assert 'files' in process_conf, \
                "'files' must be defined for a node type"
        in_files = get_in_files(process_conf['files'])
        feat_ops = parse_feat_ops(process_conf['features']) \
                if 'features' in process_conf else None
        label_conf = process_conf['labels'] if 'labels' in process_conf else None

        user_parser = partial(parse_node_data, feat_ops=feat_ops,
                              node_id_col=node_id_col,
                              label_conf=label_conf,
                              read_file=read_file)
        start = time.time()
        pool = WorkerPool(node_type, in_files, num_processes, user_parser)
        return_dict = pool.get_data()
        pool.close()
        dur = time.time() - start
        print(f"Processing data files for node {node_type} takes {dur:.3f} seconds.")

        type_node_id_map = [None] * len(return_dict)
        type_node_data = {}
        for i, (node_ids, data) in return_dict.items():
            for feat_name in data:
                if feat_name not in type_node_data:
                    type_node_data[feat_name] = [None] * len(return_dict)
                type_node_data[feat_name][i] = data[feat_name]
            type_node_id_map[i] = node_ids
        return_dict = None

        for i, id_map in enumerate(type_node_id_map):
            assert id_map is not None, f"We do not get ID map in part {i}."
        type_node_id_map = np.concatenate(type_node_id_map)
        gc.collect()
        print(f"node type {node_type} has {len(type_node_id_map)} nodes")
        # We don't need to create ID map if the node IDs are integers,
        # all node Ids are in sequence start from 0 and
        # the user doesn't force to remap node IDs.
        if np.issubdtype(type_node_id_map.dtype, np.integer) \
                and np.all(type_node_id_map == np.arange(len(type_node_id_map))) \
                and not remap_id:
            num_nodes = len(type_node_id_map)
            type_node_id_map = NoopMap(num_nodes)
        else:
            type_node_id_map = IdMap(type_node_id_map)
            num_nodes = len(type_node_id_map)
        sys_tracker.check(f'Create node ID map of {node_type}')

        for feat_name in type_node_data:
            type_node_data[feat_name] = np.concatenate(type_node_data[feat_name])
            # If we allow to store features in external memory, we store node features
            # that have large feature dimensions in a file and use memmap to access
            # the array.
            type_node_data[feat_name] = convert2ext_mem(type_node_data[feat_name],
                                                        node_type + "_" + feat_name)
            assert len(type_node_data[feat_name]) == num_nodes
            feat_shape = type_node_data[feat_name].shape
            print(f"node type {node_type} has feature {feat_name} of {feat_shape}")
            gc.collect()
            sys_tracker.check(f'Merge node data {feat_name} of {node_type}')

        # Some node types don't have data.
        if len(type_node_data) > 0:
            node_data[node_type] = type_node_data
        if type_node_id_map is not None:
            node_id_map[node_type] = type_node_id_map

    sys_tracker.check('Finish processing node data')
    return (node_id_map, node_data)

def process_edge_data(process_confs, node_id_map, convert2ext_mem,
                      num_processes=1,
                      skip_nonexist_edges=False):
    """ Process edge data

    The edge data of an edge type is defined as follows:
    {
        "source_id_col":    "<column name>",
        "dest_id_col":      "<column name>",
        "relation":         "<src type, relation type, dest type>",
        "format":           {"name": "csv", "separator": ","},
        "files":            ["<paths to files>", ...],
        "features":         [
            {
                "feature_col":  ["<column name>", ...],
                "feature_name": "<feature name>",
                "data_type":    "<feature data type>",
                "transform":    {"name": "<operator name>", ...}
            },
        ],
        "labels":           [
            {
                "label_col":    "<column name>",
                "task_type":    "<task type: e.g., classification>",
                "split_pct":   [0.8, 0.2, 0.0],
                "custom_train": "<the file with node IDs in the train set>",
                "custom_valid": "<the file with node IDs in the validation set>",
                "custom_test":  "<the file with node IDs in the test set>",
            },
        ],
    }

    Parameters
    ----------
    process_confs: list of dicts
        The configurations to process edge data.
    node_id_map: dict
        The node ID map.
    convert2ext_mem : ExtMemArrayConverter
        A callable to convert a Numpy array to external-memory Numpy array.
    num_processes: int
        The number of processes to process the input files.
    skip_nonexist_edges : bool
        Whether or not to skip edges that don't exist.

    Returns
    -------
    dict: edge features.
    """
    edges = {}
    edge_data = {}

    for process_conf in process_confs:
        # each iteration is to process an edge type.
        assert 'source_id_col' in process_conf, \
                "'source_id_col' is not defined for an edge type."
        assert 'dest_id_col' in process_conf, \
                "'dest_id_col' is not defined for an edge type."
        assert 'relation' in process_conf, \
                "'relation' is not defined for an edge type."
        edge_type = process_conf['relation']
        assert 'format' in process_conf, \
                "'format' is not defined for an edge type."
        read_file = parse_edge_file_format(process_conf)
        assert 'files' in process_conf, \
                "'files' is not defined for an edge type."
        in_files = get_in_files(process_conf['files'])
        feat_ops = parse_feat_ops(process_conf['features']) \
                if 'features' in process_conf else None

        # We don't need to copy all node ID maps to the worker processes.
        # Only the node ID maps of the source node type and destination node type
        # are sufficient.
        id_map = {edge_type[0]: node_id_map[edge_type[0]],
                  edge_type[2]: node_id_map[edge_type[2]]}
        user_parser = partial(parse_edge_data, feat_ops=feat_ops,
                              node_id_map=id_map,
                              read_file=read_file,
                              conf=process_conf,
                              skip_nonexist_edges=skip_nonexist_edges)
        start = time.time()
        pool = WorkerPool(edge_type, in_files, num_processes, user_parser)
        return_dict = pool.get_data()
        pool.close()
        dur = time.time() - start
        print(f"Processing data files for edges of {edge_type} takes {dur:.3f} seconds")

        type_src_ids = [None] * len(return_dict)
        type_dst_ids = [None] * len(return_dict)
        type_edge_data = {}
        for i, (src_ids, dst_ids, part_data) in return_dict.items():
            type_src_ids[i] = src_ids
            type_dst_ids[i] = dst_ids
            for feat_name in part_data:
                if feat_name not in type_edge_data:
                    type_edge_data[feat_name] = [None] * len(return_dict)
                type_edge_data[feat_name][i] = part_data[feat_name]
        return_dict = None

        type_src_ids = np.concatenate(type_src_ids)
        type_dst_ids = np.concatenate(type_dst_ids)
        assert len(type_src_ids) == len(type_dst_ids)
        gc.collect()
        print(f"finish merging edges of {edge_type}")

        for feat_name in type_edge_data:
            type_edge_data[feat_name] = np.concatenate(type_edge_data[feat_name])
            # If we allow to store features in external memory, we store edge features
            # that have large feature dimensions in a file and use memmap to access
            # the array.
            etype_str = "-".join(edge_type)
            type_edge_data[feat_name] = convert2ext_mem(type_edge_data[feat_name],
                                                        etype_str + "_" + feat_name)
            assert len(type_edge_data[feat_name]) == len(type_src_ids)
            feat_shape = type_edge_data[feat_name].shape
            print(f"edge type {edge_type} has feature {feat_name} of {feat_shape}")
            gc.collect()

        edge_type = tuple(edge_type)
        edges[edge_type] = (type_src_ids, type_dst_ids)
        # Some edge types don't have edge data.
        if len(type_edge_data) > 0:
            edge_data[edge_type] = type_edge_data

    return edges, edge_data

def verify_confs(confs):
    """ Verify the configuration of the input data.
    """
    ntypes = {conf['node_type'] for conf in confs["nodes"]}
    etypes = [conf['relation'] for conf in confs["edges"]]
    for src_type, _, dst_type in etypes:
        assert src_type in ntypes, \
                f"source node type {src_type} does not exist. Please check your input data."
        assert dst_type in ntypes, \
                f"dest node type {dst_type} does not exist. Please check your input data."

def partition_graph(g, node_data, edge_data, graph_name, num_partitions, output_dir,
                    part_method=None):
    """ Partition a graph

    This takes advantage of the graph partition function in DGL.
    To save memory consumption for graph partition. We only pass the graph object
    with the graph structure to DGL's graph partition function.
    We will split the node/edge feature tensors based on the graph partition results.
    By doing so, we can keep the node/edge features in external memory to further
    save memory.

    Parameters
    ----------
    g : DGLGraph
        The full graph object.
    node_data : dict of tensors
        The node feature tensors.
    edge_data : dict of tensors
        The edge feature tensors.
    graph_name : str
        The graph name.
    num_partitions : int
        The number of partitions.
    output_dir : str
        The directory where we will save the partitioned results.
    part_method : str (optional)
        The partition algorithm used to partition the graph.
    """
    from dgl.distributed.graph_partition_book import _etype_tuple_to_str
    orig_id_name = "__gs_orig_id"
    for ntype in g.ntypes:
        g.nodes[ntype].data[orig_id_name] = th.arange(g.number_of_nodes(ntype))
    for etype in g.canonical_etypes:
        g.edges[etype].data[orig_id_name] = th.arange(g.number_of_edges(etype))
    sys_tracker.check('Before partitioning starts')
    if part_method is None:
        part_method = "None" if num_partitions == 1 else "metis"
    dgl.distributed.partition_graph(g, graph_name, num_partitions, output_dir,
                                    part_method=part_method,
                                    # TODO(zhengda) we need to enable balancing node types.
                                    balance_ntypes=None,
                                    balance_edges=True)
    sys_tracker.check('Graph partitioning')
    for i in range(num_partitions):
        part_dir = os.path.join(output_dir, "part" + str(i))
        data = dgl.data.utils.load_tensors(os.path.join(part_dir, "node_feat.dgl"))
        # Get the node features for the partition and save the node features in node_feat.dgl.
        for ntype in node_data:
            # We store the original node IDs as a node feature when we partition the graph.
            # We can get the original node IDs from the node features and now
            # we use them to retrieve the right node features.
            orig_ids = data[ntype + "/" + orig_id_name]
            for name, ndata in node_data[ntype].items():
                data[ntype + "/" + name] = th.tensor(ndata[orig_ids])
            sys_tracker.check(f'Get node data of node {ntype} in partition {i}')
        # Delete the original node IDs from the node data.
        for ntype in g.ntypes:
            del data[ntype + "/" + orig_id_name]
        dgl.data.utils.save_tensors(os.path.join(part_dir, "node_feat.dgl"), data)

        data = dgl.data.utils.load_tensors(os.path.join(part_dir, "edge_feat.dgl"))
        # Get the edge features for the partition and save the edge features in edge_feat.dgl.
        for etype in edge_data:
            # We store the original edge IDs as a edge feature when we partition the graph.
            # We can get the original edge IDs from the edge features and now
            # we use them to retrieve the right edge features.
            orig_ids = data[_etype_tuple_to_str(etype) + '/' + orig_id_name]
            for name, edata in edge_data[etype].items():
                data[_etype_tuple_to_str(etype) + "/" + name] = th.tensor(edata[orig_ids])
            sys_tracker.check(f'Get edge data of edge {etype} in partition {i}')
        for etype in g.canonical_etypes:
            del data[_etype_tuple_to_str(etype) + '/' + orig_id_name]
        dgl.data.utils.save_tensors(os.path.join(part_dir, "edge_feat.dgl"), data)

def process_graph(args):
    """ Process the graph.
    """
    with open(args.conf_file, 'r', encoding="utf8") as json_file:
        process_confs = json.load(json_file)

    sys_tracker.set_rank(0)
    num_processes_for_nodes = args.num_processes_for_nodes \
            if args.num_processes_for_nodes is not None else args.num_processes
    num_processes_for_edges = args.num_processes_for_edges \
            if args.num_processes_for_edges is not None else args.num_processes
    verify_confs(process_confs)
    # We only store data to external memory if we partition a graph for distributed training.
    ext_mem_workspace = args.ext_mem_workspace if args.output_format == "DistDGL" else None
    convert2ext_mem = ExtMemArrayConverter(ext_mem_workspace, args.ext_mem_feat_size)
    node_id_map, node_data = process_node_data(process_confs['nodes'], convert2ext_mem,
                                               args.remap_node_id,
                                               num_processes=num_processes_for_nodes)
    edges, edge_data = process_edge_data(process_confs['edges'], node_id_map,
                                         convert2ext_mem,
                                         num_processes=num_processes_for_edges,
                                         skip_nonexist_edges=args.skip_nonexist_edges)
    num_nodes = {ntype: len(node_id_map[ntype]) for ntype in node_id_map}
    if args.add_reverse_edges:
        edges1 = {}
        for etype in edges:
            e = edges[etype]
            assert isinstance(e, tuple) and len(e) == 2
            assert isinstance(etype, tuple) and len(etype) == 3
            edges1[etype] = e
            edges1[etype[2], etype[1] + "-rev", etype[0]] = (e[1], e[0])
        edges = edges1
        sys_tracker.check('Add reverse edges')
    g = dgl.heterograph(edges, num_nodes_dict=num_nodes)
    sys_tracker.check('Construct DGL graph')

    if args.output_format == "DistDGL":
        partition_graph(g, node_data, edge_data, args.graph_name,
                        args.num_partitions, args.output_dir)
    elif args.output_format == "DGL":
        for ntype in node_data:
            for name, ndata in node_data[ntype].items():
                g.nodes[ntype].data[name] = th.tensor(ndata)
        for etype in edge_data:
            for name, edata in edge_data[etype].items():
                g.edges[etype].data[name] = th.tensor(edata)
        dgl.save_graphs(os.path.join(args.output_dir, args.graph_name + ".dgl"), [g])
    else:
        raise ValueError('Unknown output format: {}'.format(args.output_format))
    for ntype in node_id_map:
        kv_pairs = node_id_map[ntype].get_key_vals()
        if kv_pairs is not None:
            map_data = {}
            map_data["orig"], map_data["new"] = kv_pairs
            write_data_parquet(map_data, os.path.join(args.output_dir,
                                                      ntype + "_id_remap.parquet"))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Preprocess graphs")
    argparser.add_argument("--conf_file", type=str, required=True,
                           help="The configuration file.")
    argparser.add_argument("--num_processes", type=int, default=1,
                           help="The number of processes to process the data simulteneously.")
    argparser.add_argument("--num_processes_for_nodes", type=int,
                           help="The number of processes to process node data simulteneously.")
    argparser.add_argument("--num_processes_for_edges", type=int,
                           help="The number of processes to process edge data simulteneously.")
    argparser.add_argument("--output_dir", type=str, required=True,
                           help="The path of the output data folder.")
    argparser.add_argument("--graph_name", type=str, required=True,
                           help="The graph name")
    argparser.add_argument("--remap_node_id", action='store_true',
                           help="Whether or not to remap node IDs.")
    argparser.add_argument("--add_reverse_edges", action='store_true',
                           help="Add reverse edges.")
    argparser.add_argument("--output_format", type=str, default="DistDGL",
                           help="The output format of the constructed graph.")
    argparser.add_argument("--num_partitions", type=int, default=1,
                           help="The number of graph partitions. " + \
                                   "This is only valid if the output format is DistDGL.")
    argparser.add_argument("--skip_nonexist_edges", action='store_true',
                           help="Skip edges that whose endpoint nodes don't exist.")
    argparser.add_argument("--ext_mem_workspace", type=str,
                           help="The directory where we can store data during graph construction.")
    argparser.add_argument("--ext_mem_feat_size", type=int, default=64,
                           help="The minimal number of feature dimensions that features " + \
                                   "can be stored in external memory.")
    process_graph(argparser.parse_args())
