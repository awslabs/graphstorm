""" Preprocess the input data """
from multiprocessing import Process
import multiprocessing
import glob
import os
import dgl
import json
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from functools import partial
import argparse

from transformers import BertTokenizer
import torch as th

##################### The I/O functions ####################

def read_data_parquet(data_file):
    """ Read data from the parquet file.

    A row of a multi-dimension data is stored as an object in Parquet.
    We need to stack them to form a tensor.

    Parameters
    ----------
    data_file : str
        The parquet file that contains the data

    Returns
    -------
    dict : map from data name to data.
    """
    table = pq.read_table(data_file)
    pd = table.to_pandas()
    data = {}
    for key in pd:
        d = np.array(pd[key])
        # For multi-dimension arrays, we split them by rows and
        # save them as objects in parquet. We need to merge them
        # together and store them in a tensor.
        if d.dtype.hasobject:
            d = [d[i] for i in range(len(d))]
            d = np.stack(d)
        data[key] = d
    return data

def write_data_parquet(data, data_file):
    """ Write data in parquet files.

    Normally, Parquet cannot support multi-dimension arrays.
    This function splits a multi-dimensiion array into N arrays
    (each row is an array) and store the arrays as objects in the parquet file.

    Parameters
    ----------
    data : dict
        The data to be saved to the Parquet file.
    data_file : str
        The file name of the Parquet file.
    """
    df = {}
    for key in data:
        arr = data[key]
        assert len(arr.shape) == 1 or len(arr.shape) == 2
        if len(arr.shape) == 1:
            df[key] = arr
        else:
            df[key] = [arr[i] for i in range(len(arr))]
    table = pa.Table.from_arrays(list(df.values()), names=list(df.keys()))
    pq.write_table(table, data_file)

def parse_file_format(fmt):
    """ Parse the file format blob

    Parameters
    ----------
    fmt : dict
        Describe the file format.
    """
    assert 'name' in fmt, "'name' field must be defined in the format."
    if fmt["name"] == "parquet":
        return read_data_parquet
    else:
        raise ValueError('Unknown file format: {}'.format(fmt['name']))

############## The functions for parsing configurations #############

class Tokenizer:
    """ A wrapper to a tokenizer.

    It is defined to process multiple strings.

    Parameters
    ----------
    tokenizer : a tokenizer
    max_seq_length : int
        The maximal length of the tokenization results.
    """
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, strs):
        """ Tokenization function.

        Parameters
        ----------
        strs : list of strings.
            The text data to be tokenized.

        Returns
        -------
        a dict of tokenization results.
        """
        tokens = []
        att_masks = []
        type_ids = []
        for s in strs:
            t = self.tokenizer(s, max_length=self.max_seq_length,
                               truncation=True, padding='max_length', return_tensors='pt')
            tokens.append(t['input_ids'])
            att_masks.append(t['attention_mask'])
            type_ids.append(t['token_type_ids'])
        return {'token_ids': th.cat(tokens, dim=0),
                'attention_mask': th.cat(att_masks, dim=0),
                'token_type_ids': th.cat(type_ids, dim=0)}

def parse_tokenize(op):
    """ Parse the tokenization configuration

    The parser returns a function that tokenizes text with HuggingFace tokenizer.
    The tokenization function returns a dict of three Pytorch tensors.

    Parameters
    ----------
    op : dict
        The configuration for the operation.

    Returns
    -------
    callable : a function to process the data.
    """
    tokenizer = BertTokenizer.from_pretrained(op['bert_model'])
    max_seq_length = int(op['max_seq_length'])
    return Tokenizer(tokenizer, max_seq_length)

def parse_feat_ops(confs):
    """ Parse the configurations for processing the features

    The feature transformation:
    {
        "feature_col":  ["<column name>", ...],
        "feature_name": "<feature name>",
        "data_type":    "<feature data type>",
        "transform":    {"name": "<operator name>", ...}
    }

    Parameters
    ----------
    confs : list
        A list of feature transformations.

    Returns
    -------
    list of tuple : The operations
    """
    ops = []
    for feat in confs:
        # TODO(zhengda) we will support data type in the future.
        dtype = None
        if 'transform' not in feat:
            transform = None
        else:
            transform = feat['transform']
            assert 'name' in transform, \
                    "'name' must be defined in the transformation field."
            if transform['name'] == 'tokenize_hf':
                transform = parse_tokenize(transform)
            else:
                raise ValueError('Unknown operation: {}'.format(transform['name']))
        feat_name = feat['feature_name'] if 'feature_name' in feat else None
        assert 'feature_col' in feat, \
                "'feature_col' must be defined in a feature field."
        ops.append((feat['feature_col'], feat_name, dtype, transform))
    return ops

#################### The main function for processing #################

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

def process_labels(data, label_confs):
    """ Process labels

    Parameters
    ----------
    data : dict
        The data stored as a dict.
    label_conf : list of dict
        The list of configs to construct labels.
    """
    assert len(label_confs) == 1
    label_conf = label_confs[0]
    assert 'label_col' in label_conf, "'label_col' must be defined in the label field."
    label_col = label_conf['label_col']
    label = data[label_conf['label_col']]
    assert 'task_type' in label_conf, "'task_type' must be defined in the label field."
    if label_conf['task_type'] == 'classification':
        label = np.int32(label)
    if 'split_type' in label_conf:
        train_split, val_split, test_split = label_conf['split_type']
        rand_idx = np.random.permutation(len(label))
        train_idx = rand_idx[0:int(len(label) * train_split)]
        val_idx = rand_idx[int(len(label) * train_split):int(len(label) * (train_split + val_split))]
        test_idx = rand_idx[int(len(label) * (train_split + val_split)):]
        train_mask = np.zeros((len(label),), dtype=np.int8)
        val_mask = np.zeros((len(label),), dtype=np.int8)
        test_mask = np.zeros((len(label),), dtype=np.int8)
        train_mask[train_idx] = 1
        val_mask[val_idx] = 1
        test_mask[test_idx] = 1
    return {label_col: label,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask}

################### The functions for multiprocessing ###############

def wait_process(q, max_proc):
    """ Wait for a process

    Parameters
    ----------
    q : list of process
        The list of processes
    max_proc : int
        The maximal number of processes to process the data together.
    """
    if len(q) < max_proc:
        return
    q[0].join()
    q.pop(0)
    
def wait_all(q):
    """ Wait for all processes

    Parameters
    ----------
    q : list of processes
        The list of processes
    """
    for p in q:
        p.join()
        
def get_in_files(in_files):
    """ Get the input files.

    The input file string may contains a wildcard. This function
    gets all files that meet the requirement.

    Parameters
    ----------
    in_files : a str or a list of str
        The input files.

    Returns
    -------
    a list of str : the full name of input files.
    """
    if '*' in in_files:
        in_files = glob.glob(in_files)
    elif not isinstance(in_files, list):
        in_files = [in_files]
    in_files.sort()
    return in_files

def parse_node_data(file_idx, in_file, feat_ops, node_id_col, label_conf,
                    read_file, return_dict):
    """ Parse node data.

    The function parses a node file that contains node IDs, features and labels
    The node file is parsed according to users' configuration
    and performs some feature transformation and save the result in
    `return_dict`.

    Parameters
    ----------
    file_idx : int
        The index of the node file among all node files.
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
    return_dict : dict
        The dictionary that is shared among all processes and saves the parsed node data.
    """
    data = read_file(in_file)
    feat_data = process_features(data, feat_ops) if feat_ops is not None else {}
    if label_conf is not None:
        label_data = process_labels(data, label_conf)
        for key, val in label_data.items():
            feat_data[key] = val
    return_dict[file_idx] = (data[node_id_col], feat_data)

def parse_edge_data(file_idx, in_file, feat_ops, src_id_col, dst_id_col, edge_type,
                    node_id_map, label_conf, read_file, return_dict):
    """ Parse edge data.

    The function parses an edge file that contains the source and destination node
    IDs, edge features and potentially edge labels. The edge file is parsed
    according to users' configuration and performs some feature transformation
    and save the result in `return_dict`.

    Parameters
    ----------
    file_idx : int
        The index of the edge file among all node files.
    in_file : str
        The path of the input edge file.
    feat_ops : dict
        The operations run on the edge features of the edge file.
    src_id_col : str
        The column name that contains the source node ID.
    dst_id_col : str
        The column name that contains the destination node ID.
    edge_type : tuple
        The tuple that contains source node type, relation type and destination node type.
    node_id_map : dict
        Contains the ID mapping for every node type.
    label_conf : dict
        The configuration of labels.
    read_file : callable
        The function to read the node file
    return_dict : dict
        The dictionary that is shared among all processes and saves the parsed edge data.
    """
    data = read_file(in_file)
    feat_data = process_features(data, feat_ops) if feat_ops is not None else {}
    if label_conf is not None:
        label_data = process_labels(data, label_conf)
        for key, val in label_data.items():
            feat_data[key] = val
    src_ids = data[src_id_col]
    dst_ids = data[dst_id_col]
    assert node_id_map is not None
    src_type, _, dst_type = edge_type
    if src_type in node_id_map:
        src_ids = np.array([node_id_map[src_type][sid] for sid in src_ids])
    else:
        assert np.issubdtype(src_ids.dtype, np.integer), \
                "The source node Ids have to be integer."
    if dst_type in node_id_map:
        dst_ids = np.array([node_id_map[dst_type][did] for did in dst_ids])
    else:
        assert np.issubdtype(dst_ids.dtype, np.integer), \
                "The destination node Ids have to be integer."
    return_dict[file_idx] = (src_ids, dst_ids, feat_data)

def create_id_map(ids):
    """ Create ID map

    This creates an ID map for the input IDs.

    Parameters
    ----------
    ids : Numpy array
        The input IDs

    Returns
    -------
    dict : the key is the original ID and the value is the new ID.
    """
    return {id1: i for i, id1 in enumerate(ids)}

def process_node_data(process_confs, remap_id):
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
    remap_id: bool
        Whether or not to remap node IDs

    Returns
    -------
    dict: node ID map
    dict: node features.
    """
    node_data = {}
    node_id_map = {}
    for process_conf in process_confs:
        assert 'node_id_col' in process_conf, \
                "'node_id_col' must be defined for a node type."
        node_id_col = process_conf['node_id_col']
        assert 'node_type' in process_conf, \
                "'node_type' must be defined for a node type"
        node_type = process_conf['node_type']
        assert 'format' in process_conf, \
                "'format' must be defined for a node type"
        read_file = parse_file_format(process_conf['format'])
        assert 'files' in process_conf, \
                "'files' must be defined for a node type"
        in_files = get_in_files(process_conf['files'])
        feat_ops = parse_feat_ops(process_conf['features']) \
                if 'features' in process_conf else None
        label_conf = process_conf['labels'] if 'labels' in process_conf else None
        q = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        for i, in_file in enumerate(in_files):
            p = Process(target=parse_node_data, args=(i, in_file, feat_ops, node_id_col,
                                                      label_conf, read_file, return_dict))
            p.start()
            q.append(p)
            wait_process(q, num_processes)
        wait_all(q)

        type_node_id_map = [None] * len(return_dict)
        type_node_data = {}
        for i, (node_ids, data) in return_dict.items():
            for feat_name in data:
                if feat_name not in type_node_data:
                    type_node_data[feat_name] = [None] * len(return_dict)
                type_node_data[feat_name][i] = data[feat_name]
            type_node_id_map[i] = node_ids

        assert type_node_id_map[0] is not None
        type_node_id_map = np.concatenate(type_node_id_map)
        # We don't need to create ID map if the node IDs are integers,
        # all node Ids are in sequence start from 0 and
        # the user doesn't force to remap node IDs.
        if np.issubdtype(type_node_id_map.dtype, np.integer) \
                and np.all(type_node_id_map == np.arange(len(type_node_id_map))) \
                and not remap_id:
            num_nodes = len(type_node_id_map)
            type_node_id_map = None
        else:
            num_nodes = len(type_node_id_map)
            type_node_id_map = create_id_map(type_node_id_map)

        for feat_name in type_node_data:
            type_node_data[feat_name] = np.concatenate(type_node_data[feat_name])
            assert len(type_node_data[feat_name]) == num_nodes

        # Some node types don't have data.
        if len(type_node_data) > 0:
            node_data[node_type] = type_node_data
        if type_node_id_map is not None:
            node_id_map[node_type] = type_node_id_map

    return (node_id_map, node_data)

def process_edge_data(process_confs, node_id_map):
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
        The configurations to process edge data.
    node_id_map: dict
        The node ID map.

    Returns
    -------
    dict: edge features.
    """
    edges = {}
    edge_data = {}

    for process_conf in process_confs:
        assert 'source_id_col' in process_conf, \
                "'source_id_col' is not defined for an edge type."
        src_id_col = process_conf['source_id_col']
        assert 'dest_id_col' in process_conf, \
                "'dest_id_col' is not defined for an edge type."
        dst_id_col = process_conf['dest_id_col']
        assert 'relation' in process_conf, \
                "'relation' is not defined for an edge type."
        edge_type = process_conf['relation']
        assert 'format' in process_conf, \
                "'format' is not defined for an edge type."
        read_file = parse_file_format(process_conf['format'])
        assert 'files' in process_conf, \
                "'files' is not defined for an edge type."
        in_files = get_in_files(process_conf['files'])
        feat_ops = parse_feat_ops(process_conf['features']) \
                if 'features' in process_conf else None
        label_conf = process_conf['labels'] if 'labels' in process_conf else None
        q = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        for i, in_file in enumerate(in_files):
            p = Process(target=parse_edge_data, args=(i, in_file, feat_ops,
                                                      src_id_col, dst_id_col, edge_type,
                                                      node_id_map, label_conf,
                                                      read_file, return_dict))
            p.start()
            q.append(p)
            wait_process(q, num_processes)
        wait_all(q)

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

        type_src_ids = np.concatenate(type_src_ids)
        type_dst_ids = np.concatenate(type_dst_ids)
        assert len(type_src_ids) == len(type_dst_ids)

        for feat_name in type_edge_data:
            type_edge_data[feat_name] = np.concatenate(type_edge_data[feat_name])
            assert len(type_edge_data[feat_name]) == len(type_src_ids)

        edge_type = tuple(edge_type)
        edges[edge_type] = (type_src_ids, type_dst_ids)
        # Some edge types don't have edge data.
        if len(type_edge_data) > 0:
            edge_data[edge_type] = type_edge_data

    return edges, edge_data

def save_dist_graph(g, graph_name, output_dir):
    """ Save the constructed graph in DistDGL format.

    This function first converts the DGLGraph format to DistDGL graph format
    in DGL v1.0 and save the result in the output directory.

    To save memory consumption, it converts the format manually instead of
    calling the graph partitioning function in DGL. It first saves the node
    features and edge features in the DGL format and free the memory used
    to store node/edge features. It then converts the DGL graph object
    to the homogeneous graph format and adds the necessary node/edge field
    to mimic the DistGraph object.

    Parameters
    ----------
    g : DGLGraph
        The constructed graph
    graph_name : str
        The graph name.
    output_dir : str
        The path of the output directory
    """
    node_data = {}
    part_path = os.path.join(output_dir, 'part0')
    os.makedirs(part_path, mode=0o775, exist_ok=True)
    for ntype in g.ntypes:
        for name in g.nodes[ntype].data:
            node_data[ntype + '/' + name] = g.nodes[ntype].data[name]
            # We should delete node data from the DGL graph object
            del g.nodes[ntype].data[name]
    dgl.data.utils.save_tensors(os.path.join(part_path, 'node_feat.dgl'), node_feats)
    node_feats = None   # This will trigger GC to free memory for storing node features.
    for etype in g.canonical_etypes:
        for name in g.edges[etype].data:
            edge_data[_etype_tuple_to_str(etype) + '/' + name] = g.edges[etype].data[name]
            # We should delete edge data from the DGL graph object
            del g.edges[etype].data[name]
    dgl.data.utils.save_tensors(os.path.join(part_path, 'edge_feat.dgl'), edge_feats)
    edge_feats = None   # This will trigger GC to free memory for storing edge features.

    ntypes = {ntype:g.get_ntype_id(ntype) for ntype in g.ntypes}
    etypes = {etype:g.get_etype_id(etype) for etype in g.canonical_etypes}
    node_map_val = {}
    edge_map_val = {}
    num_nodes = 0
    num_edges = 0
    for ntype in g.ntypes:
        node_map_val[ntype] = [num_nodes, num_nodes + g.number_of_nodes(ntype)]
        num_nodes += g.number_of_nodes(ntype)
    for etype in g.canonical_etypes:
        edge_map_val[etype] = [num_edges, num_edges + g.number_of_edges(etype)]
        num_edges += g.number_of_edges(etype)
    # We store the graph structure in the homogeneous graph format.
    g = dgl.to_homogeneous(g)
    g.ndata['inner_node'] = th.ones(g.number_of_nodes(), dtype=th.uint8)
    g.edata['inner_edge'] = th.ones(g.number_of_edges(), dtype=th.uint8)
    g.ndata[dgl.NID] = th.arange(g.number_of_nodes(), dtype=th.int64)
    g.edata[dgl.EID] = th.arange(g.number_of_edges(), dtype=th.int64)
    dgl.save_graphs(os.path.join(part_path, "graph.dgl"), [g])

    part_metadata = {'graph_name': graph_name,
                     'num_nodes': g.number_of_nodes(),
                     'num_edges': g.number_of_edges(),
                     'part_method': "None",
                     'num_parts': 1,
                     'halo_hops': 0,
                     'node_map': node_map_val,
                     'edge_map': edge_map_val,
                     'ntypes': ntypes,
                     'etypes': etypes}
    with open(os.path.join(output_dir, graph_name + '.json'), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Preprocess graphs")
    argparser.add_argument("--conf_file", type=str, required=True,
            help="The configuration file.")
    argparser.add_argument("--num_processes", type=int, default=1,
            help="The number of processes to process the data simulteneously.")
    argparser.add_argument("--output_dir", type=str, required=True,
            help="The path of the output data folder.")
    argparser.add_argument("--graph_name", type=str, required=True,
            help="The graph name")
    argparser.add_argument("--remap_node_id", type=bool, default=False,
            help="Whether or not to remap node IDs.")
    argparser.add_argument("--output_format", type=str, default="DistDGL",
            help="The output format of the constructed graph.")
    args = argparser.parse_args()
    num_processes = args.num_processes
    process_confs = json.load(open(args.conf_file, 'r'))

    node_id_map, node_data = process_node_data(process_confs['node'], args.remap_node_id)
    edges, edge_data = process_edge_data(process_confs['edge'], node_id_map)
    num_nodes = {}
    for ntype in set(list(node_data.keys()) + list(node_id_map.keys())):
        # If a node type has Id map.
        if ntype in node_id_map:
            num_nodes[ntype] = len(node_id_map[ntype])
        # If a node type has node data.
        elif ntype in node_data:
            for feat_name in node_data[ntype]:
                # Here we only need to look at the length of the first node features
                # to get the number of nodes for the node type.
                num_nodes[ntype] = len(node_data[ntype][feat_name])
                break
        else:
            # A node type must have either ID map or node data.
            raise ValueError('Node type {} must have either ID map or node data'.format(ntype))
    g = dgl.heterograph(edges, num_nodes_dict=num_nodes)
    for ntype in node_data:
        for name, data in node_data[ntype].items():
            g.nodes[ntype].data[name] = th.tensor(data)
    for etype in edge_data:
        for name, data in edge_data[etype].items():
            g.edges[etype].data[name] = th.tensor(data)

    if args.output_format == "DistDGL":
        dgl.save_graphs(os.path.join(args.output_dir, args.graph_name + ".dgl"), [g])
    elif args.output_format == "DGL":
        save_dist_graph(g, args.graph_name, args.output_dir)
    else:
        raise ValueError('Unknown output format: {}'.format(args.output_format))
    for ntype in node_id_map:
        map_data = {}
        map_data["orig"] = np.array(list(node_id_map[ntype].keys()))
        map_data["new"] = np.array(list(node_id_map[ntype].values()))
        write_data_parquet(map_data, os.path.join(args.output_dir, ntype + "_id_remap.parquet"))
