'''
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

    This file contains some general utility functions for the framework.
'''
import os
import json
import time
import resource
import psutil

import dgl
import torch as th

def get_rank():
    """ Get rank of a process
    """
    return th.distributed.get_rank()

def estimate_mem_train(root, task):
    ''' Estimate the memory consumption per machine during training.

    Parameters
    ----------
    root : str
        The path to the partitioned graph folder.
    task : str
        It's either an 'edge' task or a 'node' task.

    Returns
    -------
    a tuple of max memory size and shared memory size.
    '''
    mem_list = []
    shared_mem_list = []
    parts = []
    # Find the partition IDs from the folder.
    for f in os.listdir(root):
        if os.path.isdir(os.path.join(root, f)):
            parts.append(int(f[4:]))
    parts.sort()
    for i in parts:
        part_path = os.path.join(root, f'part{i}')
        if os.path.isdir(part_path):
            g = dgl.load_graphs(os.path.join(part_path, 'graph.dgl'))[0][0]
            num_nodes = g.number_of_nodes()
            num_edges = g.number_of_edges()
            # The memory consumption of the graph structure.
            # This includes the coo format (16), edge ID (8), inner edge (1),
            # original edge ID (8), edge type (8)
            struct_size = (num_edges * (16 + 8 + 1 + 8 + 8)
                    # this includes inner node (1), node ID (8), original node ID (8),
                    # node type (8)
                    + num_nodes * (1 + 8 + 8 + 8)
                    + (num_edges * 16 + num_nodes * 8) # This is to store the CSC format.
                    ) / 1024/1024/1024
            node_feats = os.path.getsize(os.path.join(part_path, 'node_feat.dgl')) / 1024/1024/1024
            edge_feats = os.path.getsize(os.path.join(part_path, 'edge_feat.dgl')) / 1024/1024/1024
            # The memory usage when after the server runs.
            # At this point, all data are stored in the shared memory.
            shared_mem = stable_serv_mem = struct_size + node_feats + edge_feats
            # The peak memory usage
            # Here we assume that the shared memory is pre-allocated.
            # If we need to allocate regular memory, we need additional memory from the system.
            max_serv_mem = max([shared_mem + struct_size, # when loading the graph structure.
                shared_mem + node_feats,                  # when loading the node features.
                shared_mem + edge_feats])                 # when loading the edge features.
            # The memory usage of all trainers in a machine.
            max_cli_mem = num_edges * 8 * 2 if task == 'edge' else num_nodes * 8 * 2
            # It's bit hard to estimate the trainer memory. Let's be more conservative.
            max_cli_mem *= 1.5
            max_cli_mem = max_cli_mem / 1024/1024/1024
            mem_list.append(max(max_serv_mem, stable_serv_mem + max_cli_mem))
            shared_mem_list.append(shared_mem)
            print('part{i}, N={num_nodes}, E={num_edges}, peak serv mem: {max_serv_mem:.3f} GB, '\
                    'stable serv mem: {stable_serv_mem:.3f} GB, '\
                    'shared mem: {shared_mem_list[-1]:.3f} GB, cli mem: {max_cli_mem:.3f} GB')
    return max(mem_list), max(shared_mem_list)

def estimate_mem_infer(root, graph_name, hidden_size, num_layers):
    ''' Estimate the memory consumption for inference.

    Parameters
    ----------
    root : str
        The path to the partitioned graph folder.
    graph_name : str
        The graph name.
    hidden_size : int
        The hidden size for the GNN embeddings.
    num_layers : int
        The number of GNN layers.

    Returns
    -------
    a tuple of max memory size and shared memory size.
    '''
    mem_list = []
    shared_mem_list = []
    parts = []
    # Find the partition IDs from the folder.
    for f in os.listdir(root):
        if os.path.isdir(os.path.join(root, f)):
            parts.append(int(f[4:]))
    with open(os.path.join(root, graph_name + '.json'), 'r', encoding='utf-8') as f:
        schema = json.load(f)
    parts.sort()
    for i in parts:
        part_path = os.path.join(root, f'part{i}')
        if os.path.isdir(part_path):
            # number of nodes in the partition.
            ntypes = list(schema['node_map'].keys())
            num_part_nodes = []
            for ntype in ntypes:
                r = schema['node_map'][ntype][i]
                num_part_nodes.append(r[1] - r[0])
            num_part_nodes = sum(num_part_nodes)

            g = dgl.load_graphs(os.path.join(part_path, 'graph.dgl'))[0][0]
            num_nodes = g.number_of_nodes()
            num_edges = g.number_of_edges()
            # The memory size for the graph structure. The calculation is the same as above.
            struct_size = (num_edges * (16 + 8 + 1 + 8 + 8) + num_nodes * (1 + 8 + 8 + 8)
                    + (num_edges * 16 + num_nodes * 8)) / 1024/1024/1024
            # The memory size for the node features.
            node_feats = os.path.getsize(os.path.join(part_path, 'node_feat.dgl')) / 1024/1024/1024
            # The memory size for the edge features.
            edge_feats = os.path.getsize(os.path.join(part_path, 'edge_feat.dgl')) / 1024/1024/1024
            # The shared memory stores the graph structure, the node features, edge features
            # as well as the embeddings of the input layer and each GNN layer.
            shared_mem = (struct_size + node_feats + edge_feats
                    + num_part_nodes * hidden_size * 4 * (num_layers + 1) / 1024/1024/1024)
            # The memory usage when after the server runs.
            # Majority data is stored in shared memory. When saving the GNN embeddings to the disk,
            # we need to extract the GNN node embeddings, which is stored
            # in the local Pytorch tensor.
            stable_serv_mem = shared_mem + num_part_nodes * hidden_size * 4 / 1024/1024/1024
            # The peak memory usage
            max_serv_mem = max([struct_size + shared_mem, shared_mem + node_feats,
                shared_mem + edge_feats, stable_serv_mem])
            # The memory usage of all trainers in a machine.
            max_cli_mem = num_nodes * 8 * 2
            # It's bit hard to estimate the trainer memory. Let's be more conservative.
            max_cli_mem *= 1.5
            max_cli_mem = max_cli_mem / 1024/1024/1024
            mem_list.append(max(max_serv_mem, stable_serv_mem + max_cli_mem))
            shared_mem_list.append(shared_mem)
            print(f'part {i}, N={num_nodes}, E={num_edges}, peak serv mem: {max_serv_mem:.3f} GB, '\
                    'stable serv mem: {stable_serv_mem:.3f} GB, '\
                    'shared mem: {shared_mem_list[-1]:.3f} GB, cli mem: {max_cli_mem:.3f} GB')
    return max(mem_list), max(shared_mem_list)

class SysTracker:
    """ This tracks the system performance.

    It tracks the runtime and memory consumption.
    """
    def __init__(self, verbose=True):
        self._checkpoints = []
        self._rank = dgl.distributed.rpc.get_rank()
        self._verbose = verbose

    # This is to create only one instance.
    _instance = None

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        """ Only create one instance.
        """
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)

        return cls._instance

    def set_rank(self, rank):
        """ Manually set rank.

        This can be used if the system is not initialized correctly.
        """
        self._rank = rank

    def check(self, name):
        """ Check the system metrics.
        """
        mem_info = psutil.Process(os.getpid()).memory_info()
        gmem_info = psutil.virtual_memory()
        self._checkpoints.append((name, time.time(), mem_info.rss, mem_info.shared,
                                  resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                  gmem_info.used, gmem_info.shared))
        # We need to get the right rank
        if self._rank < 0:
            self._rank = dgl.distributed.rpc.get_rank()
        if len(self._checkpoints) >= 2 and self._verbose and self._rank == 0:
            checkpoint1 = self._checkpoints[-2]
            checkpoint2 = self._checkpoints[-1]
            print("{}: elapsed time: {:.3f}, mem (curr: {:.3f}, peak: {:.3f}, shared: {:.3f}, \
                    global curr: {:.3f}, global shared: {:.3f}) GB".format(
                name, checkpoint2[1] - checkpoint1[1],
                checkpoint2[2]/1024/1024/1024, checkpoint2[4]/1024/1024,
                checkpoint2[3]/1024/1024/1024, checkpoint2[5]/1024/1024/1024,
                checkpoint2[6]/1024/1024/1024))

sys_tracker = SysTracker()
