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
import logging

import numpy as np
import torch as th
import dgl

from ..utils import sys_tracker
from .file_io import parse_node_file_format, parse_edge_file_format
from .file_io import get_in_files, HDF5Array
from .transform import parse_feat_ops, process_features, preprocess_features
from .transform import parse_label_ops, process_labels
from .transform import (do_multiprocess_transform,
                        TwoPhaseFeatTransform,
                        GlobalProcessFeatTransform)
from .id_map import NoopMap, IdMap, map_node_ids
from .utils import (multiprocessing_data_read,
                    update_two_phase_feat_ops, ExtMemArrayMerger,
                    partition_graph)

def prepare_node_data(in_file, feat_ops, read_file):
    """ Parse node data.

    The function parses a node file that contains node IDs, features and labels
    The node file is parsed according to users' configuration
    and performs some feature transformation.

    Parameters
    ----------
    in_file : str
        The path of the input node file.
    feat_ops : dict of FeatTransform
        The operations run on the node features of the node file.
    read_file : callable
        The function to read the node file

    Returns
    -------
    dict : A dict of node feature info.
    """
    data = read_file(in_file)
    assert feat_ops is not None, "feat_ops must exist when prepare_node_data is called."
    feat_info = preprocess_features(data, feat_ops)

    return feat_info

def parse_node_data(in_file, feat_ops, label_ops, node_id_col, read_file):
    """ Parse node data.

    The function parses a node file that contains node IDs, features and labels
    The node file is parsed according to users' configuration
    and performs some feature transformation.

    Parameters
    ----------
    in_file : str
        The path of the input node file.
    feat_ops : dict of FeatTransform
        The operations run on the node features of the node file.
    label_ops : dict of LabelProcessor
        The operations run on the node labels of the node file.
    node_id_col : str
        The column name that contains the node ID.
    read_file : callable
        The function to read the node file

    Returns
    -------
    tuple : node ID array and a dict of node feature tensors.
    """
    data = read_file(in_file)
    feat_data = process_features(data, feat_ops) if feat_ops is not None else {}
    if label_ops is not None:
        label_data = process_labels(data, label_ops)
        for key, val in label_data.items():
            feat_data[key] = val
    node_ids = data[node_id_col] if node_id_col in data else None
    return (node_ids, feat_data)

def parse_edge_data(in_file, feat_ops, label_ops, node_id_map, read_file,
                    conf, skip_nonexist_edges):
    """ Parse edge data.

    The function parses an edge file that contains the source and destination node
    IDs, edge features and potentially edge labels. The edge file is parsed
    according to users' configuration and performs some feature transformation.

    Parameters
    ----------
    in_file : str
        The path of the input edge file.
    feat_ops : dict of FeatTransform
        The operations run on the edge features of the edge file.
    label_ops : dict of LabelProcessor
        The operations run on the node labels of the edge file.
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
    src_id_col = conf['source_id_col'] if 'source_id_col' in conf else None
    dst_id_col = conf['dest_id_col'] if 'dest_id_col' in conf else None
    assert not ((src_id_col is None) ^ (dst_id_col is None)), \
        f"{in_file} should either have both source_id_col and dest_id_col" \
        "or have none."
    edge_type = conf['relation']

    data = read_file(in_file)
    feat_data = process_features(data, feat_ops) if feat_ops is not None else {}
    if label_ops is not None:
        label_data = process_labels(data, label_ops)
        for key, val in label_data.items():
            feat_data[key] = val
    src_ids = data[src_id_col] if src_id_col is not None else None
    dst_ids = data[dst_id_col] if dst_id_col is not None else None
    if src_ids is not None:
        src_ids, dst_ids = map_node_ids(src_ids, dst_ids, edge_type, node_id_map,
                                        skip_nonexist_edges)
    return (src_ids, dst_ids, feat_data)

def process_node_data(process_confs, arr_merger, remap_id, num_processes=1):
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
                "transform":    {"name": "<operator name>", ...}
            },
        ],
        "labels":       [
            {
                "label_col":    "<column name>",
                "task_type":    "<task type: e.g., classification>",
                "split_type":   [0.8, 0.2, 0.0],
            },
        ],
    }

    Parameters
    ----------
    process_confs: list of dicts
        The configurations to process node data.
    arr_merger : ExtMemArrayMerger
        A callable to merge multiple arrays.
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
        assert 'node_type' in process_conf, \
                "'node_type' must be defined for a node type"
        node_type = process_conf['node_type']
        assert 'files' in process_conf, \
                "'files' must be defined for a node type"
        in_files = get_in_files(process_conf['files'])
        feat_ops = parse_feat_ops(process_conf['features']) \
                if 'features' in process_conf else None
        label_ops = parse_label_ops(process_conf['labels'], is_node=True) \
                if 'labels' in process_conf else None
        assert 'format' in process_conf, \
                "'format' must be defined for a node type"
        multiprocessing = do_multiprocess_transform(process_conf, feat_ops, label_ops, in_files)
        # If it requires multiprocessing, we need to read data to memory.
        read_file = parse_node_file_format(process_conf, in_mem=multiprocessing)
        node_id_col = process_conf['node_id_col'] if 'node_id_col' in process_conf else None
        num_proc = num_processes if multiprocessing else 0

        two_phase_feat_ops = []
        after_merge_feat_ops = {}
        if feat_ops is not None:
            for op in feat_ops:
                if isinstance(op, TwoPhaseFeatTransform):
                    two_phase_feat_ops.append(op)
                if isinstance(op, GlobalProcessFeatTransform):
                    after_merge_feat_ops[op.feat_name] = op

        if len(two_phase_feat_ops) > 0:
            user_pre_parser = partial(prepare_node_data, feat_ops=two_phase_feat_ops,
                                      read_file=read_file)
            pre_parse_start = time.time()
            phase_one_ret = multiprocessing_data_read(in_files, num_proc, user_pre_parser)
            update_two_phase_feat_ops(phase_one_ret, two_phase_feat_ops)

            dur = time.time() - pre_parse_start
            logging.debug("Preprocessing data files for node %s takes %.3f seconds.",
                        node_type, dur)

        user_parser = partial(parse_node_data, feat_ops=feat_ops,
                              label_ops=label_ops,
                              node_id_col=node_id_col,
                              read_file=read_file)
        start = time.time()
        return_dict = multiprocessing_data_read(in_files, num_proc, user_parser)
        dur = time.time() - start
        logging.debug("Processing data files for node %s takes %.3f seconds.",
                      node_type, dur)

        type_node_id_map = [None] * len(return_dict)
        type_node_data = {}
        for i, (node_ids, data) in return_dict.items():
            for feat_name in data:
                if feat_name not in type_node_data:
                    type_node_data[feat_name] = [None] * len(return_dict)
                type_node_data[feat_name][i] = data[feat_name]
            # If it's HDF5Array, it's better to convert it into a Numpy array.
            # This will make the next operations on it more efficiently.
            if isinstance(node_ids, HDF5Array):
                type_node_id_map[i] = node_ids.to_numpy()
            else:
                type_node_id_map[i] = node_ids
        return_dict = None

        # Construct node Id map.
        if type_node_id_map[0] is not None:
            assert all(id_map is not None for id_map in type_node_id_map)
            if len(type_node_id_map) > 1:
                type_node_id_map = np.concatenate(type_node_id_map)
            else:
                type_node_id_map = type_node_id_map[0]
            logging.debug("node type %s has %d nodes.", node_type, len(type_node_id_map))
        else:
            assert all(id_map is None for id_map in type_node_id_map)
            type_node_id_map = None
        gc.collect()
        # We don't need to create ID map if the node IDs are integers,
        # all node Ids are in sequence start from 0 and
        # the user doesn't force to remap node IDs.
        if type_node_id_map is not None \
                and np.issubdtype(type_node_id_map.dtype, np.integer) \
                and np.all(type_node_id_map == np.arange(len(type_node_id_map))) \
                and not remap_id:
            type_node_id_map = NoopMap(len(type_node_id_map))
        elif type_node_id_map is not None:
            type_node_id_map = IdMap(type_node_id_map)
            sys_tracker.check(f'Create node ID map of {node_type}')

        for feat_name in type_node_data:
            merged_feat = arr_merger(type_node_data[feat_name],
                                     node_type + "_" + feat_name)
            if feat_name in after_merge_feat_ops:
                # do data transformation with the entire feat array.
                merged_feat = after_merge_feat_ops[feat_name].after_merge_transform(merged_feat)
            type_node_data[feat_name] = merged_feat
            gc.collect()
            sys_tracker.check(f'Merge node data {feat_name} of {node_type}')

        # If we didn't see the node data for this node type before.
        if len(type_node_data) > 0 and node_type not in node_data:
            node_data[node_type] = type_node_data
        # If we have seen the node data for this node type before
        # because there are multiple blocks that contain data for the same node type.
        elif len(type_node_data) > 0:
            for key, val in type_node_data.items():
                # Make sure the node data has duplicated names.
                assert key not in node_data[node_type], \
                        f"The node data {key} has exist in node type {node_type}."
                node_data[node_type][key] = val
        if type_node_id_map is not None:
            assert node_type not in node_id_map, \
                    f"The ID map of node type {node_type} has existed."
            node_id_map[node_type] = type_node_id_map

    for node_type in node_data:
        assert node_type in node_id_map, \
                f"The input files do not contain node Ids for node type {node_type}."
        for data in node_data[node_type].values():
            assert len(data) == len(node_id_map[node_type]), \
                    f"Node data and node IDs for node type {node_type} does not match: " + \
                    f"{len(data)} vs. {len(node_id_map[node_type])}"
    sys_tracker.check('Finish processing node data')
    return (node_id_map, node_data)

def process_edge_data(process_confs, node_id_map, arr_merger,
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
                "transform":    {"name": "<operator name>", ...}
            },
        ],
        "labels":           [
            {
                "label_col":    "<column name>",
                "task_type":    "<task type: e.g., classification>",
                "split_pct":   [0.8, 0.2, 0.0],
            },
        ],
    }

    Parameters
    ----------
    process_confs: list of dicts
        The configurations to process edge data.
    node_id_map: dict
        The node ID map.
    arr_merger : ExtMemArrayMerger
        A callable to merge multiple arrays.
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
        assert 'relation' in process_conf, \
                "'relation' is not defined for an edge type."
        edge_type = process_conf['relation']
        assert 'files' in process_conf, \
                "'files' is not defined for an edge type."
        in_files = get_in_files(process_conf['files'])
        assert 'format' in process_conf, \
                "'format' is not defined for an edge type."
        feat_ops = parse_feat_ops(process_conf['features']) \
                if 'features' in process_conf else None
        label_ops = parse_label_ops(process_conf['labels'], is_node=False) \
                if 'labels' in process_conf else None
        multiprocessing = do_multiprocess_transform(process_conf, feat_ops, label_ops, in_files)
        # If it requires multiprocessing, we need to read data to memory.
        read_file = parse_edge_file_format(process_conf, in_mem=multiprocessing)

        # We don't need to copy all node ID maps to the worker processes.
        # Only the node ID maps of the source node type and destination node type
        # are sufficient.
        id_map = {edge_type[0]: node_id_map[edge_type[0]],
                  edge_type[2]: node_id_map[edge_type[2]]}
        user_parser = partial(parse_edge_data, feat_ops=feat_ops,
                              label_ops=label_ops,
                              node_id_map=id_map,
                              read_file=read_file,
                              conf=process_conf,
                              skip_nonexist_edges=skip_nonexist_edges)
        start = time.time()
        num_proc = num_processes if multiprocessing else 0
        return_dict = multiprocessing_data_read(in_files, num_proc, user_parser)
        dur = time.time() - start
        logging.debug("Processing data files for edges of %s takes %.3f seconds",
                      str(edge_type), dur)

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

        # handle edge type
        for feat_name in type_edge_data:
            etype_str = "-".join(edge_type)
            type_edge_data[feat_name] = arr_merger(type_edge_data[feat_name],
                                                   etype_str + "_" + feat_name)
            gc.collect()
            sys_tracker.check(f'Merge edge data {feat_name} of {edge_type}')

        edge_type = tuple(edge_type)
        if type_src_ids[0] is not None: # handle src_ids and dst_ids
            assert all(src_ids is not None for src_ids in type_src_ids)
            assert all(dst_ids is not None for dst_ids in type_dst_ids)
            type_src_ids = np.concatenate(type_src_ids)
            type_dst_ids = np.concatenate(type_dst_ids)
            assert len(type_src_ids) == len(type_dst_ids)

            edges[edge_type] = (type_src_ids, type_dst_ids)
        gc.collect()
        logging.debug("Finish merging edges of %s", str(edge_type))

        # If we didn't see the edge data for this edge type before.
        if len(type_edge_data) > 0 and edge_type not in edge_data:
            edge_data[edge_type] = type_edge_data
        # If we have seen the edge data for this edge type before
        # because there are multiple blocks that contain data for the same edge type.
        elif len(type_edge_data) > 0:
            for key, val in type_edge_data.items():
                # Make sure the edge data has duplicated names.
                assert key not in edge_data[edge_type], \
                        f"The edge data {key} has exist in edge type {edge_type}."
                edge_data[edge_type][key] = val

    for edge_type, edge_feats in edge_data.items():
        assert edge_type in edges, \
            f"source_id_col and dest_id_col is not defined for {edge_type}"
        for feat_name, efeats in edge_feats.items():
            assert len(efeats) == len(edges[edge_type][0]), \
                f"The length of edge feature {feat_name} of etype {edge_type} " \
                f"does not match the number of edges of {edge_type}. " \
                f"Expecting {len(edges[edge_type][0])}, but get {len(efeats)}"

    return edges, edge_data

def verify_confs(confs):
    """ Verify the configuration of the input data.
    """
    ntypes = {conf['node_type'] for conf in confs["nodes"]}
    etypes = [conf['relation'] for conf in confs["edges"]]
    for etype in etypes:
        assert len(etype) == 3, \
                "The edge type must be (source node type, relation type, dest node type)."
        src_type, _, dst_type = etype
        assert src_type in ntypes, \
                f"source node type {src_type} does not exist. Please check your input data."
        assert dst_type in ntypes, \
                f"dest node type {dst_type} does not exist. Please check your input data."

def get_log_level(log_level):
    """ Map the logging level.
    """
    if log_level == "debug":
        return logging.DEBUG
    elif log_level == "info":
        return logging.INFO
    elif log_level == "warning":
        return logging.WARNING
    elif log_level == "error":
        return logging.ERROR
    else:
        raise ValueError(f"Unknown logging level {log_level}. " + \
                "The possible values are: debug, info, warning, error.")

def print_graph_info(g, node_data, edge_data):
    """ Print graph information.

    Parameters
    ----------
    g : DGLGraph
        The constructed graph.
    node_data : dict of dict of Numpy arrays.
        Node features
    edge_data : dict of dict of Numpy arrays.
        Edge features
    """
    logging.info("The graph has %d node types and %d edge types.",
                 len(g.ntypes), len(g.etypes))
    for ntype in node_data:
        feat_names = list(node_data[ntype].keys())
        logging.info("Node type %s has %d nodes with features: %s.",
                     ntype, g.number_of_nodes(ntype), str(feat_names))
        num_train = np.sum(node_data[ntype]["train_mask"]) \
                if "train_mask" in node_data[ntype] else 0
        num_val = np.sum(node_data[ntype]["val_mask"]) \
                if "val_mask" in node_data[ntype] else 0
        num_test = np.sum(node_data[ntype]["test_mask"]) \
                if "test_mask" in node_data[ntype] else 0
        if num_train + num_val + num_test > 0:
            logging.info("Train/val/test on %s: %d, %d, %d",
                         ntype, num_train, num_val, num_test)
    for etype in edge_data:
        feat_names = list(edge_data[etype].keys())
        logging.info("Edge type %s has %d edges with features: %s.",
                     str(etype), g.number_of_edges(etype), str(feat_names))
        num_train = np.sum(edge_data[etype]["train_mask"]) \
                if "train_mask" in edge_data[etype] else 0
        num_val = np.sum(edge_data[etype]["val_mask"]) \
                if "val_mask" in edge_data[etype] else 0
        num_test = np.sum(edge_data[etype]["test_mask"]) \
                if "test_mask" in edge_data[etype] else 0
        if num_train + num_val + num_test > 0:
            logging.info("Train/val/test on %s: %d, %d, %d",
                         str(etype), num_train, num_val, num_test)

def process_graph(args):
    """ Process the graph.
    """
    logging.basicConfig(level=get_log_level(args.logging_level))
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
    convert2ext_mem = ExtMemArrayMerger(ext_mem_workspace, args.ext_mem_feat_size)
    node_id_map, node_data = process_node_data(process_confs['nodes'], convert2ext_mem,
                                               args.remap_node_id,
                                               num_processes=num_processes_for_nodes)
    edges, edge_data = process_edge_data(process_confs['edges'], node_id_map,
                                         convert2ext_mem,
                                         num_processes=num_processes_for_edges,
                                         skip_nonexist_edges=args.skip_nonexist_edges)
    num_nodes = {ntype: len(node_id_map[ntype]) for ntype in node_id_map}
    if args.output_conf_file is not None:
        # Save the new config file.
        with open(args.output_conf_file, "w", encoding="utf8") as outfile:
            json.dump(process_confs, outfile, indent=4)
    sys_tracker.check('Process input data')

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
    print_graph_info(g, node_data, edge_data)
    sys_tracker.check('Construct DGL graph')

    # reshape customized mask
    for srctype_etype_dsttype in edge_data:
        if "train_mask" in edge_data[srctype_etype_dsttype].keys() and \
            len(edge_data[srctype_etype_dsttype]["train_mask"].shape) == 2:
            edge_data[srctype_etype_dsttype]["train_mask"] = \
                edge_data[srctype_etype_dsttype]["train_mask"].squeeze(1).astype('int8')
        if "val_mask" in edge_data[srctype_etype_dsttype].keys() and \
            len(edge_data[srctype_etype_dsttype]["val_mask"].shape) == 2:
            edge_data[srctype_etype_dsttype]["val_mask"] = \
                edge_data[srctype_etype_dsttype]["val_mask"].squeeze(1).astype('int8')
        if "test_mask" in edge_data[srctype_etype_dsttype].keys() and \
            len(edge_data[srctype_etype_dsttype]["test_mask"].shape) == 2:
            edge_data[srctype_etype_dsttype]["test_mask"] = \
                edge_data[srctype_etype_dsttype]["test_mask"].squeeze(1).astype('int8')

    if args.output_format == "DistDGL":
        assert args.part_method in ["metis", "random"], \
                "We only support 'metis' or 'random'."
        partition_graph(g, node_data, edge_data, args.graph_name,
                        args.num_parts, args.output_dir,
                        save_mapping=True, # always save mapping
                        part_method=args.part_method)
    elif args.output_format == "DGL":
        for ntype in node_data:
            for name, ndata in node_data[ntype].items():
                if isinstance(ndata, HDF5Array):
                    g.nodes[ntype].data[name] = ndata.to_tensor()
                else:
                    g.nodes[ntype].data[name] = th.tensor(ndata)
        for etype in edge_data:
            for name, edata in edge_data[etype].items():
                if isinstance(edata, HDF5Array):
                    g.edges[etype].data[name] = edata.to_tensor()
                else:
                    g.edges[etype].data[name] = th.tensor(edata)
        dgl.save_graphs(os.path.join(args.output_dir, args.graph_name + ".dgl"), [g])
    else:
        raise ValueError('Unknown output format: {}'.format(args.output_format))
    for ntype in node_id_map:
        map_file = os.path.join(args.output_dir, ntype + "_id_remap.parquet")
        if node_id_map[ntype].save(map_file):
            logging.info("Graph construction generates new node IDs for '%s'. " + \
                    "The ID map is saved in %s.", ntype, map_file)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Preprocess graphs")
    argparser.add_argument("--conf-file", type=str, required=True,
                           help="The configuration file.")
    argparser.add_argument("--output-conf-file", type=str,
                           help="The output file with the updated configurations.")
    argparser.add_argument("--num-processes", type=int, default=1,
                           help="The number of processes to process the data simulteneously.")
    argparser.add_argument("--num-processes-for-nodes", type=int,
                           help="The number of processes to process node data simulteneously.")
    argparser.add_argument("--num-processes-for-edges", type=int,
                           help="The number of processes to process edge data simulteneously.")
    argparser.add_argument("--output-dir", type=str, required=True,
                           help="The path of the output data folder.")
    argparser.add_argument("--graph-name", type=str, required=True,
                           help="The graph name")
    argparser.add_argument("--remap-node-id", action='store_true',
                           help="Whether or not to remap node IDs.")
    argparser.add_argument("--add-reverse-edges", action='store_true',
                           help="Add reverse edges.")
    argparser.add_argument("--output-format", type=str, default="DistDGL",
                           help="The output format of the constructed graph.")
    argparser.add_argument("--num-parts", type=int, default=1,
                           help="The number of graph partitions. " + \
                                   "This is only valid if the output format is DistDGL.")
    argparser.add_argument("--part-method", type=str, default='metis',
                           help="The partition method. Currently, we support 'metis' and 'random'.")
    argparser.add_argument("--skip-nonexist-edges", action='store_true',
                           help="Skip edges that whose endpoint nodes don't exist.")
    argparser.add_argument("--ext-mem-workspace", type=str,
                           help="The directory where we can store data during graph construction.")
    argparser.add_argument("--ext-mem-feat-size", type=int, default=64,
                           help="The minimal number of feature dimensions that features " + \
                                   "can be stored in external memory.")
    argparser.add_argument("--logging-level", type=str, default="info",
                           help="The logging level. The possible values: debug, info, warning, \
                                   error. The default value is info.")
    process_graph(argparser.parse_args())
