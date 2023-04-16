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
import os
import json
import argparse
import gc

import numpy as np
import torch as th
import dgl

from ..utils import sys_tracker
from .file_io import parse_node_file_format, parse_edge_file_format
from .file_io import get_in_files, write_data_parquet
from .transform import parse_feat_ops, process_features
from .transform import process_labels
from .id_map import NoopMap, IdMap, map_node_ids
from .utils import WorkerPool, ExtMemArrayConverter, partition_graph

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
