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
import logging
import psutil

import pandas as pd
import dgl
import torch as th
import numpy as np

TORCH_MAJOR_VER = int(th.__version__.split('.', maxsplit=1)[0])
USE_WHOLEGRAPH = False
GS_DEVICE = th.device('cpu')

def get_graph_name(part_config):
    """ Get graph name from graph partition config file

    Parameter
    ---------
    part_config: str
        Path to graph partition config file

    Return
    ------
        graph_name
    """
    with open(part_config, "r", encoding='utf-8') as f:
        config = json.load(f)
    return config["graph_name"]

def setup_device(local_rank):
    r"""Setup computation device.

    Parameters
    -----------
    local_rank: int
        Rank of the current process in a distributed environment.

    Returns
    -------
    str: device where the model runs.
    """
    if th.cuda.is_available():
        assert local_rank < th.cuda.device_count(), \
                f"local rank={local_rank} but there are {th.cuda.device_count()} GPUs."
        device = 'cuda:%d' % local_rank
        th.cuda.set_device(device)
    else:
        device = 'cpu'

    global GS_DEVICE
    GS_DEVICE = th.device(device)
    return device

def get_device():
    """ Get the torch device to run model forward and backward

    Return
    th.device: device where the model runs
    """
    return GS_DEVICE

def is_distributed():
    """ Test whether the process runs in a distributed mode.
    """
    return th.distributed.is_initialized()

def get_backend():
    """ Get the backend of a process group.
    """
    assert is_distributed(), "get_backend() is valid only when is_distributed() is True."
    return th.distributed.get_backend()

def get_rank():
    """ Get rank of a process
    """
    if is_distributed():
        return th.distributed.get_rank()
    return 0

def get_world_size():
    """ Get the world size.
    """
    if is_distributed():
        return th.distributed.get_world_size()
    return 1

def barrier():
    """ Run barrier among trainers.
    """
    if is_distributed():
        th.distributed.barrier()

def use_wholegraph(part_config):
    """ Use wholegraph for feature fetching if 'wholegraph' folder exists
    """
    global USE_WHOLEGRAPH
    USE_WHOLEGRAPH = bool(part_config is not None and os.path.isdir(os.path.join( \
        os.path.dirname(part_config), 'wholegraph')))
    return USE_WHOLEGRAPH

def is_wholegraph():
    """ Check whether global USE_WHOLEGRAPH is true
    """
    return USE_WHOLEGRAPH

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
        if os.path.isdir(os.path.join(root, f)) and f.startswith("part"):
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
            logging.info('part%d, N=%d, E=%d, peak serv mem: %.3f GB, ' + \
                    'stable serv mem: %.3f GB, shared mem: %.3f GB, cli mem: %.3f GB',
                         i, num_nodes, num_edges, max_serv_mem,
                         stable_serv_mem, shared_mem_list[-1], max_cli_mem)
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
        if os.path.isdir(os.path.join(root, f)) and f.startswith("part"):
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
            logging.info('part%d, N=%d, E=%d, peak serv mem: %.3f GB, ' + \
                    'stable serv mem: %.3f GB, shared mem: %.3f GB, cli mem: %.3f GB',
                         i, num_nodes, num_edges, max_serv_mem,
                         stable_serv_mem, shared_mem_list[-1], max_cli_mem)
    return max(mem_list), max(shared_mem_list)

def print_mem(device):
    """ Print memory consumption
    """
    if th.cuda.is_available():
        logging.info("Peak GPU Mem alloc: %.4f MB",
                     th.cuda.max_memory_allocated(device) / 1024 / 1024)
    else:
        logging.info("Peak RAM Mem alloc: %.4f MB",
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

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

def create_dist_tensor(shape, dtype, name=None, part_policy=None, persistent=False):
    """ A wrapper function to create a distributed tensor.
    """
    tensor = dgl.distributed.DistTensor(shape, dtype, name=name,
                                        part_policy=part_policy, persistent=persistent)
    logging.debug("Create DistTensor of %s with shape of %s", name, str(tensor.shape))
    return tensor

def get_lm_ntypes(lm_configs):
    """ Get the node types with text features.
    """
    if lm_configs is None:
        return None

    ntypes = []
    for config in lm_configs:
        ntypes.extend(config['node_types'])
    return ntypes

class SysTracker:
    """ This tracks the system performance.

    It tracks the runtime and memory consumption.
    """
    def __init__(self):
        self._checkpoints = []
        self._rank = -1
        self._verbose = False

    # This is to create only one instance.
    _instance = None

    def init(self, verbose=False, rank=None):
        """ Initialize the profiler.

            Set tracking level.
            TODO: Only have verbose=True or False.
            Need to support different logging level later.

        Parameter
        ---------
        verbose: bool
            Verbose for print out more info
        rank: int
            Current trainer rank
        """
        self._verbose = verbose
        if rank is None:
            self._rank = get_rank()
        else:
            self._rank = rank

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
        if self._verbose is False:
            return

        mem_info = psutil.Process(os.getpid()).memory_info()
        gmem_info = psutil.virtual_memory()
        self._checkpoints.append((name, time.time(), mem_info.rss, mem_info.shared,
                                  resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                  gmem_info.used, gmem_info.shared))
        # We need to get the right rank
        if self._rank < 0:
            self._rank = get_rank()
        if len(self._checkpoints) >= 2 and self._rank == 0:
            checkpoint1 = self._checkpoints[-2]
            checkpoint2 = self._checkpoints[-1]
            logging.debug("{}: elapsed time: {:.3f}, mem (curr: {:.3f}, peak: {:.3f}, \
                    shared: {:.3f}, global curr: {:.3f}, global shared: {:.3f}) GB".format(
                name, checkpoint2[1] - checkpoint1[1],
                checkpoint2[2]/1024/1024/1024, checkpoint2[4]/1024/1024,
                checkpoint2[3]/1024/1024/1024, checkpoint2[5]/1024/1024/1024,
                checkpoint2[6]/1024/1024/1024))

class RuntimeProfiler:
    """ This profiles the runtime performance.

    It tracks the runtime.
    """
    def __init__(self, profile_path=None):
        self._checkpoints = []
        self._runtime = {}
        self._profile_path = profile_path
        self._rank = -1

    # This is to create only one instance.
    _instance = None

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        """ Only create one instance.
        """
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)

        return cls._instance

    def init(self, path, rank=None):
        """ Initialize the profiler.

        Setting the profile path enables profiling.
        """
        self._profile_path = path
        if rank is None:
            self._rank = get_rank()
        else:
            self._rank = rank

    def start_record(self):
        """ Start recording.

        This records the first time for the following operations.
        """
        if self._profile_path is None:
            return
        self._checkpoints.append(("", time.time()))
        # We put a barrier here so that the next operation starts
        # at the same time.
        th.distributed.barrier()

    def record(self, name):
        """ Record the computation step.

        It basically adds a checkpoint in the place where it is called.

        Parameters
        ----------
        name : str
            The name of the check point.
        """
        if self._profile_path is None:
            return

        self._checkpoints.append((name, time.time()))
        if len(self._checkpoints) >= 2:
            checkpoint1 = self._checkpoints[-2]
            checkpoint2 = self._checkpoints[-1]
            runtime = checkpoint2[1] - checkpoint1[1]
            name = checkpoint2[0]
            if name not in self._runtime:
                self._runtime[name] = [runtime]
            else:
                self._runtime[name].append(runtime)
        # We put a barrier here so that the next operation starts
        # at the same time.
        th.distributed.barrier()

    def print_stats(self):
        """ Print the statistics
        """
        if self._rank == 0 and self._profile_path is not None:
            for name, runtimes in self._runtime.items():
                logging.info("%s %.3f seconds", name, sum(runtimes) / len(runtimes))

    def save_profile(self):
        """ Save the profiling result to a file.
        """
        if self._profile_path is not None:
            runtime = {}
            for name in self._runtime:
                runtime[name] = np.array(self._runtime[name])
            profile_path = os.path.join(self._profile_path, f"{self._rank}.csv")
            data_frame = pd.DataFrame(runtime)
            data_frame.to_csv(profile_path, float_format='%.3f', index=False)
            logging.info("Save profiling in %s.", profile_path)

sys_tracker = SysTracker()
rt_profiler = RuntimeProfiler()
