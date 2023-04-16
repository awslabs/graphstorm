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
import multiprocessing
from multiprocessing import Process
import queue
import gc

import numpy as np
import dgl
import torch as th

from ..utils import sys_tracker

def worker_fn(task_queue, res_queue, user_parser):
    """ The worker function in the worker pool

    Parameters
    ----------
    task_queue : Queue
        The queue that contains all tasks
    res_queue : Queue
        The queue that contains the processed data. This is used for
        communication between the worker processes and the master process.
    user_parser : callable
        The user-defined function to read and process the data files.
    """
    try:
        while True:
            # If the queue is empty, it will raise the Empty exception.
            i, in_file = task_queue.get_nowait()
            data = user_parser(in_file)
            res_queue.put((i, data))
            gc.collect()
    except queue.Empty:
        pass

class WorkerPool:
    """ A worker process pool

    This worker process pool is specialized to process node/edge data.
    It creates a set of worker processes, each of which runs a worker function.
    It first adds all tasks in a queue and each worker gets a task from the queue
    at a time until the workers process all tasks. It maintains a result queue
    and each worker adds the processed results in the result queue.
    To ensure the order of the data, each data file is associated with
    a number and the processed data are ordered by that number.

    Parameters
    ----------
    name : str or tuple of str
        The name of the worker pool.
    in_files : list of str
        The input data files.
    num_processes : int
        The number of processes that run in parallel.
    user_parser : callable
        The user-defined function to read and process the data files.
    """
    def __init__(self, name, in_files, num_processes, user_parser):
        self.name = name
        self.processes = []
        manager = multiprocessing.Manager()
        self.task_queue = manager.Queue()
        self.res_queue = manager.Queue(8)
        self.num_files = len(in_files)
        for i, in_file in enumerate(in_files):
            self.task_queue.put((i, in_file))
        for _ in range(num_processes):
            proc = Process(target=worker_fn, args=(self.task_queue, self.res_queue, user_parser))
            proc.start()
            self.processes.append(proc)

    def get_data(self):
        """ Get the processed data.

        Returns
        -------
        a dict : key is the file index, the value is processed data.
        """
        return_dict = {}
        while len(return_dict) < self.num_files:
            file_idx, vals= self.res_queue.get()
            return_dict[file_idx] = vals
            sys_tracker.check(f'process {self.name} data file: {file_idx}')
            gc.collect()
        return return_dict

    def close(self):
        """ Stop the process pool.
        """
        for proc in self.processes:
            proc.join()

class ExtMemArrayConverter:
    """ Convert a Numpy array to an external-memory Numpy array.

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

    def __call__(self, arr, name):
        """ Convert a Numpy array.

        Parameters
        ----------
        arr : Numpy array
            The input array.
        name : str
            The name of the external memory array.

        Returns
        -------
        Numpy array : the Numpy array stored in external memory.
        """
        # If external memory workspace is not initialized or the feature size is smaller
        # than a threshold, we don't do anything.
        if self._ext_mem_workspace is None or np.prod(arr.shape[1:]) < self._ext_mem_feat_size:
            return arr

        # We need to create the workspace directory if it doesn't exist.
        os.makedirs(self._ext_mem_workspace, exist_ok=True)
        tensor_path = os.path.join(self._ext_mem_workspace, name + ".npy")
        self._tensor_files.append(tensor_path)
        em_arr = np.memmap(tensor_path, arr.dtype, mode="w+", shape=arr.shape)
        em_arr[:] = arr[:]
        return em_arr

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
