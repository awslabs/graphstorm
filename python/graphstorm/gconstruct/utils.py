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
import os
import queue
import gc
import logging

import numpy as np
import dgl
import torch as th
from torch import multiprocessing
from torch.multiprocessing import Process

from ..utils import sys_tracker
from .file_io import HDF5Array

SHARED_MEM_OBJECT_THRESHOLD = 1.9 * 1024 * 1024 * 1024 # must < 2GB
SHARED_MEMORY_CROSS_PROCESS_STORAGE = "shared_memory"
PICKLE_CROSS_PROCESS_STORAGE = "pickle"

def _to_shared_memory(data):
    """ Move all tensor objects into torch shared memory

    Parameters
    ----------
    data: dict/tuple/list of tensors
        Data returned by user_parser
    """
    if th.is_tensor(data):
        return data.share_memory_()
    elif isinstance(data, np.ndarray):
        assert data.dtype is not np.object_, \
            "Numpy array of python objects can not be handled by graph construction"

        # only handle data that can be converted to torch tensor
        if data.dtype in [np.float64, np.float32, np.float16,
                          np.complex64, np.complex128, np.int64,
                          np.int32, np.int16, np.int8,
                          np.uint8]:
            return th.tensor(data).share_memory_()
        return data
    elif isinstance(data, dict):
        new_data = {}
        for name, val in data.items():
            new_data[name] = _to_shared_memory(val)
        return new_data
    elif isinstance(data, list):
        new_data = []
        for val in data:
            new_data.append(_to_shared_memory(val))
        return new_data
    elif isinstance(data, tuple):
        new_data = []
        for val in list(data):
            new_data.append(_to_shared_memory(val))
        return tuple(new_data)

    # ignore other types
    return data

def _to_numpy_array(data):
    """ Move all data objects back to numpy array

    Parameters
    ----------
    data: dict/tuple/list of tensors
        Data returned by user_parser
    """
    if th.is_tensor(data):
        return data.numpy()
    elif isinstance(data, np.ndarray):
        return data # do nothing
    elif isinstance(data, dict):
        new_data = {}
        for name, val in data.items():
            new_data[name] = _to_numpy_array(val)
        return new_data
    elif isinstance(data, list):
        new_data = []
        for val in data:
            new_data.append(_to_numpy_array(val))
        return new_data
    elif isinstance(data, tuple):
        new_data = []
        for val in list(data):
            new_data.append(_to_numpy_array(val))
        return tuple(new_data)

    # ignore other types
    return data

def _estimate_sizeof(data):
    """ Estimate the size of a data.
        We assume the most memory consuming objects are tensors.

    Parameters
    ----------
    data: dict/tuple/list of tensors
        Data returned by user_parser
    """
    if th.is_tensor(data):
        data_size = data.element_size() * data.nelement()
    elif isinstance(data, np.ndarray):
        assert data.dtype is not np.object_, \
            "Numpy array of python objects can not be handled by graph construction"

        data_size = data.size * data.itemsize
    elif isinstance(data, dict):
        data_size = 0
        for _, val in data.items():
            data_size += _estimate_sizeof(val)
    elif isinstance(data, list):
        data_size = 0
        for val in data:
            data_size += _estimate_sizeof(val)
    elif isinstance(data, tuple):
        data_size = 0
        for val in list(data):
            data_size += _estimate_sizeof(val)
    else:
        # for other types like primitives
        # ignore their size.
        data_size = 0

    return data_size

def worker_fn(worker_id, task_queue, res_queue, user_parser):
    """ The worker function in the worker pool

    Parameters
    ----------
    worker_id : int
        The worker ID, starting from 0.
    task_queue : Queue
        The queue that contains all tasks
    res_queue : Queue
        The queue that contains the processed data. This is used for
        communication between the worker processes and the master process.
    user_parser : callable
        The user-defined function to read and process the data files.
    """
    # We need to set a GPU device for each worker process in case that
    # some transformations (e.g., computing BERT embeddings) require GPU computation.
    if th.cuda.is_available():
        num_gpus = th.cuda.device_count()
        gpu = worker_id % num_gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        if worker_id >= num_gpus:
            logging.warning("There are more than 1 processes are attachd to GPU %d.", gpu)
    try:
        while True:
            # If the queue is empty, it will raise the Empty exception.
            i, in_file = task_queue.get_nowait()
            logging.debug("%d Processing %s", worker_id, in_file)
            data = user_parser(in_file)
            size = _estimate_sizeof(data)
            # Max pickle obj size is 2 GByte
            if size > SHARED_MEM_OBJECT_THRESHOLD:
                # Use torch shared memory as a workaround
                # This will consume shared memory and cause an additional
                # data copy, i.e., general memory to torch shared memory.
                data = (SHARED_MEMORY_CROSS_PROCESS_STORAGE, _to_shared_memory(data))
            else:
                data = (PICKLE_CROSS_PROCESS_STORAGE, data)
            res_queue.put((i, data))
            gc.collect()
    except queue.Empty:
        pass

def update_two_phase_feat_ops(phase_one_info, ops):
    """ Update the ops for the second phase feat processing

    Parameters
    ----------
    phase_one_info: dict
        A dict mapping file index to node/edge features info corresponding to ops.
    ops: dict of FeatTransform
        The operations run on the node/edge features of the node/edge files.
    """
    feat_info = {}
    for _, finfo in phase_one_info.items():
        for feat_name, info in finfo.items():
            if feat_name not in feat_info:
                feat_info[feat_name] = [info]
            else:
                feat_info[feat_name].append(info)
    for op in ops:
        # It is possible that there is no information from phase one.
        if op.feat_name in feat_info:
            op.update_info(feat_info[op.feat_name])

def multiprocessing_data_read(in_files, num_processes, user_parser):
    """ Read data from multiple files with multiprocessing.

    It creates a set of worker processes, each of which runs a worker function.
    It first adds all tasks in a queue and each worker gets a task from the queue
    at a time until the workers process all tasks. It maintains a result queue
    and each worker adds the processed results in the result queue.
    To ensure the order of the data, each data file is associated with
    a number and the processed data are ordered by that number.

    If there are only one input file, it reads the data from the input file in the main process.

    Parameters
    ----------
    in_files : list of str
        The input data files.
    num_processes : int
        The number of processes that run in parallel.
    user_parser : callable
        The user-defined function to read and process the data files.

    Returns
    -------
    a dict : key is the file index, the value is processed data.
    """
    if num_processes > 0 and len(in_files) > 1:
        processes = []
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        res_queue = manager.Queue(8 if num_processes < 8 else num_processes)
        num_files = len(in_files)
        for i, in_file in enumerate(in_files):
            task_queue.put((i, in_file))
        for i in range(num_processes):
            proc = Process(target=worker_fn, args=(i, task_queue, res_queue, user_parser))
            proc.start()
            processes.append(proc)

        return_dict = {}
        while len(return_dict) < num_files:
            file_idx, vals= res_queue.get()
            # If the size of `vals`` is larger than utils.SHARED_MEM_OBJECT_THRESHOLD
            # we will automatically convert tensors in `vals` into torch tensor
            # and copy the tensor into shared memory.
            # This helps avoid the pickle max obj size issue.
            storage_type, vals = vals
            if storage_type == SHARED_MEMORY_CROSS_PROCESS_STORAGE:
                vals = _to_numpy_array(vals)
            return_dict[file_idx] = vals
            sys_tracker.check(f'process data file: {file_idx}')
            gc.collect()

        for proc in processes:
            proc.join()

        return return_dict
    else:
        return_dict = {}
        for i, in_file in enumerate(in_files):
            return_dict[i] = user_parser(in_file)
        return return_dict

def _get_tot_shape(arrs):
    """ Get the shape after merging the arrays.

    Parameters
    ----------
    arrs : list of arrays

    Returns
    -------
    tuple : the shape of the merged array.
    """
    num_rows = 0
    shape1 = arrs[0].shape[1:]
    for arr in arrs:
        num_rows += arr.shape[0]
        assert shape1 == arr.shape[1:]
    shape = [num_rows] + list(shape1)
    return tuple(shape)

def _merge_arrs(arrs, tensor_path):
    """ Merge the arrays.

    The merged array may be stored in a file specified by the path.

    Parameters
    ----------
    arrs : list of arrays.
        The input arrays.
    tensor_path : str
        The path where the Numpy array is stored.

    Returns
    -------
    Numpy array : the merged array.
    """
    assert isinstance(arrs, list)
    shape = _get_tot_shape(arrs)
    dtype = arrs[0].dtype
    if tensor_path is not None:
        out_arr = np.memmap(tensor_path, dtype, mode="w+", shape=shape)
        row_idx = 0
        for arr in arrs:
            out_arr[row_idx:(row_idx + arr.shape[0])] = arr[:]
            row_idx += arr.shape[0]
        return out_arr
    elif isinstance(arrs[0], HDF5Array):
        arrs = [arr.to_numpy() for arr in arrs]
        return np.concatenate(arrs)
    else:
        return np.concatenate(arrs)

class ExtMemArrayMerger:
    """ Merge multiple Numpy arrays.

    The merged array may be stored on disks.

    Parameters
    ----------
    ext_mem_workspace : str
        The path of the directory where the array will be stored.
    ext_mem_feat_size : int
        The threshold of the feature size that triggers storing data on disks.
    """
    def __init__(self, ext_mem_workspace, ext_mem_feat_size):
        self._ext_mem_workspace = ext_mem_workspace
        self._ext_mem_feat_size = ext_mem_feat_size
        self._tensor_files = []

    def __del__(self):
        for tensor_file in self._tensor_files:
            os.remove(tensor_file)

    def __call__(self, arrs, name):
        """ Merge multiple Numpy array.

        Parameters
        ----------
        arrs : list of arrays.
            The input arrays.
        name : str
            The name of the external memory array.

        Returns
        -------
        Numpy array : an array stored in external memory.
        """
        assert isinstance(arrs, list)
        shape = _get_tot_shape(arrs)
        # If external memory workspace is not initialized or the feature size is smaller
        # than a threshold, we don't do anything.
        if self._ext_mem_workspace is None or np.prod(shape[1:]) < self._ext_mem_feat_size:
            if len(arrs) == 1 and isinstance(arrs[0], HDF5Array):
                return arrs[0].to_numpy()
            elif len(arrs) == 1:
                return arrs[0]
            else:
                return _merge_arrs(arrs, None)

        # We need to create the workspace directory if it doesn't exist.
        os.makedirs(self._ext_mem_workspace, exist_ok=True)
        tensor_path = os.path.join(self._ext_mem_workspace, name + ".npy")
        self._tensor_files.append(tensor_path)
        if len(arrs) > 1:
            return _merge_arrs(arrs, tensor_path)
        else:
            arr = arrs[0]
            em_arr = np.memmap(tensor_path, arr.dtype, mode="w+", shape=shape)
            em_arr[:] = arr[:]
            return em_arr

def save_maps(output_dir, fname, map_data):
    """ Save node id mapping or edge id mapping

    Parameters
    ----------
    output_dir : str
        The directory where we will save the partitioned results.
    fname: str
        Mapping file name
    map_data: dict of tensors
        ID mapping
    """
    map_file = f"{fname}.pt"
    map_file = os.path.join(output_dir, map_file)
    # Use torch save as tensors are torch tensors
    th.save(map_data, map_file)

def partition_graph(g, node_data, edge_data, graph_name, num_partitions, output_dir,
                    part_method=None, save_mapping=True):
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
    save_mapping: bool
        Whether to store the mappings for the edges and nodes after partition.
        Default: True
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

    mapping = \
        dgl.distributed.partition_graph(g, graph_name, num_partitions, output_dir,
                                        part_method=part_method,
                                        # TODO(zhengda) we need to enable balancing node types.
                                        balance_ntypes=None,
                                        balance_edges=True,
                                        return_mapping=save_mapping)
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

    if save_mapping:
        new_node_mapping, new_edge_mapping = mapping

        # the new_node_mapping contains per entity type on the ith row
        # the original node id for the ith node.
        save_maps(output_dir, "node_mapping", new_node_mapping)
        # the new_edge_mapping contains per edge type on the ith row
        # the original edge id for the ith edge.
        save_maps(output_dir, "edge_mapping", new_edge_mapping)
