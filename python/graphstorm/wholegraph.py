"""This module provides functionality for working with the WholeGraph"""
import os
import json
from typing import Optional
from dataclasses import dataclass

import torch as th
from .utils import get_rank, get_world_size

try:
    import pylibwholegraph
    import pylibwholegraph.torch as wgth
except ImportError:
    wgth = None

def init_wholegraph():
    """ Initialize Wholegraph"""
    if wgth is None:
        raise ImportError("WholeGraph is not installed")
    from dgl.distributed import role
    import pylibwholegraph.binding.wholememory_binding as wmb
    @dataclass
    class Options: # pylint: disable=missing-class-docstring
        pass
    Options.launch_agent = 'pytorch'
    Options.launch_env_name_world_rank = 'RANK'
    Options.launch_env_name_world_size = 'WORLD_SIZE'
    Options.launch_env_name_local_rank = 'LOCAL_RANK'
    Options.launch_env_name_local_size = 'LOCAL_WORLD_SIZE'
    Options.launch_env_name_master_addr = 'MASTER_ADDR'
    Options.launch_env_name_master_port = 'MASTER_PORT'
    Options.local_rank = get_rank() % role.get_num_trainers()
    Options.local_size = role.get_num_trainers()

    wgth.distributed_launch(Options, lambda: None)
    wmb.init(0)
    wgth.comm.set_world_info(get_rank(), get_world_size(), Options.local_rank,
                            Options.local_size)

def load_wg_feat(part_config_path, num_parts, type_name, name):
    """Load features from wholegraph memory

    Parameters
    ----------
    part_config_path : str
        The path of the partition configuration file.
    num_parts : int
        The number of partitions of the dataset
    type_name: str
        The type of node or edge for which to fetch features or labels for.
    name: str
        The name of the features to load
    """
    if wgth is None:
        raise ImportError("WholeGraph is not installed")
    global_comm = wgth.comm.get_global_communicator()
    feature_comm = global_comm
    embedding_wholememory_type = 'distributed'
    embedding_wholememory_location = 'cpu'
    cache_policy = wgth.create_builtin_cache_policy(
        "none", # cache type
        embedding_wholememory_type,
        embedding_wholememory_location,
        "readonly", # access type
        0.0, # cache ratio
    )
    metadata_file = os.path.join(os.path.dirname(part_config_path),
                                'wholegraph/metadata.json')
    with open(metadata_file, encoding="utf8") as f:
        wg_metadata = json.load(f)
    data_shape = wg_metadata[type_name + '/' + name]['shape']
    feat_wm_embedding = wgth.create_embedding(
        feature_comm,
        embedding_wholememory_type,
        embedding_wholememory_location,
        getattr(th, wg_metadata[type_name + '/' + name]['dtype'].split('.')[1]),
        [data_shape[0], 1] if len(data_shape) == 1 else data_shape,
        optimizer=None,
        cache_policy=cache_policy,
    )
    feat_path = os.path.join(os.path.dirname(part_config_path), 'wholegraph', \
                                            type_name + '~' + name)
    feat_wm_embedding.get_embedding_tensor().from_file_prefix(feat_path,
                                                                part_count=num_parts)
    return feat_wm_embedding

def create_wholememory_optimizer(
    optimizer_type: str, param_dict: dict
) -> wgth.WholeMemoryOptimizer:
    """Create a wholegraph sparse optimizer.

    If we use wholegraph to store sparse embeddings, for future update, a joint
    wholegraph sparse optimizer has to be created ahead of time, and then attach
    to the (wholegraph)sparse embedding.

    Parameters
    ----------
    optimizer_type: str
        optimizer types: [ "sgd" | "adam" | "adagrad" | "rmsprop" ]
    param_dict: dict
        parameters of the optimizer

    Returns
    -------
    WholeMemoryOptimizer : WholeGraph native optimizer (wgth.WholeMemoryOptimizer)
    """
    if wgth is None:
        raise ImportError("WholeGraph is not installed")
    return wgth.create_wholememory_optimizer(optimizer_type, param_dict)

def create_wg_sparse_params(
    nnodes: int,
    embedding_dim: int,
    optimizer: Optional[wgth.WholeMemoryOptimizer] = None,
    location: str = "cpu",
) -> wgth.WholeMemoryEmbeddingModule:
    """Create a wholegraph sparse embedding module.

    This is to use wholegraph distributed host/device memory to store sparse embs.
    To enable trainable embeddings, the created wholegraph embedding has to be wrapped
    in a WholeMemoryEmbeddingModule to attach gradients during each pass.

    Parameters
    ----------
    nnodes : int
        Number of nodes of the embedding, i.e., embedding_tensor.shape[0]
    embedding_dim: int
        The dimension of each embedding entry, i.e., embedding_tensor.shape[1]
    optimizer : WholeMemoryOptimizer, optional
        The attached wholegraph sparse optimizer
    location : str
        The desired location to store the embedding [ "cpu" | "cuda" ]

    Returns
    -------
    WholeMemoryEmbeddingModule : The wrapped nn module including
    the embedding table as its parameters.
    """
    if wgth is None:
        raise ImportError("WholeGraph is not installed")
    global_comm = wgth.comm.get_global_communicator()
    embedding_wholememory_type = "distributed"
    embedding_wholememory_location = location
    # Here the  initializer is different. DistDGL uses init_emb (uniform_),
    # while wg uses torch.nn.init.xavier_uniform_(local_tensor) to initialize
    dist_embedding = wgth.create_embedding(
        global_comm,
        embedding_wholememory_type,
        embedding_wholememory_location,
        # to consistent with distDGL:
        # github:dgl/blob/master/python/dgl/distributed/nn/pytorch/sparse_emb.py#L79
        th.float32,
        [nnodes, embedding_dim],
        optimizer=optimizer,
        cache_policy=None,  # disable cache for now
        random_init=True,
    )
    # wrap over emb into wg nn module to trace grad/update embed
    return wgth.WholeMemoryEmbeddingModule(dist_embedding)

class WholeGraphSparseEmbedding:
    """
    (Distributed) WholeGraph Embedding Interface for Sparse Update

    Parameters
    ----------
    nnodes : int
        Number of nodes in the graph.
    embedding_dim : int
        Dimension of the node embeddings.
    name : str
        Name of the wholegraph embeddings.
    optimizer : Optional[wgth.WholeMemoryOptimizer], optional
        Optimizer for the embeddings, by default None.
    location : str, optional
        Location of the embeddings (e.g., "cpu" or "cuda"), by default "cpu".

    """
    def __init__(
        self,
        nnodes: int,
        embedding_dim: int,
        name: str,
        optimizer: Optional[wgth.WholeMemoryOptimizer] = None,
        location: str = "cpu",
    ):
        self._module = create_wg_sparse_params(nnodes, embedding_dim, optimizer, location)
        self._optimizer = optimizer
        self._name = name
        self._num_embeddings = nnodes
        self._embedding_dim = embedding_dim

    def __call__(self, idx: th.Tensor, device: th.device = th.device("cpu")) -> th.Tensor:
        """
        Get the embeddings for the specified node indices.

        Parameters
        ----------
        idx : torch.Tensor
            Index of the embeddings to collect.
        device : torch.device, optional
            Target device to put the collected embeddings, by default torch.device("cpu").

        Returns
        -------
        torch.Tensor
            The requested node embeddings.
        """
        idx = idx.cuda()
        emb = self._module(idx).to(device, non_blocking=True)
        return emb

    def save_to_file(
        self,
        path: str,
        file_prefix: str,
    ) -> None:
        """
        Save the embedding tensor to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the file will be saved.
        file_prefix : str
            The prefix of the file.

        Returns
        -------
        None
        """
        file_prefix = os.path.join(path, file_prefix)
        self._module.wm_embedding.get_embedding_tensor().to_file_prefix(file_prefix)

    def load_from_file(
        self,
        path: str,
        file_prefix: str,
        num_files: int,
    ) -> None:
        """
        Load the embedding tensor from files.

        Parameters
        ----------
        path : str
            The path to the directory where the file is located.
        file_prefix : str
            The prefix of the file.
        num_files : int
            The number of files to load.

        Returns
        -------
        None
        """
        file_prefix = os.path.join(path, file_prefix)
        self._module.wm_embedding.get_embedding_tensor().from_file_prefix(
            file_prefix, part_count=num_files
        )

    @property
    def name(self):
        """
        Return the name of the wholegraph embeddings.

        Returns
        -------
        str
            The name of the embeddings.
        """
        return self._name

    @property
    def num_embeddings(self):
        """
        Return the number of embeddings.

        Returns
        -------
        int
            The number of embeddings.
        """
        return self._num_embeddings

    @property
    def embedding_dim(self):
        """
        Return the dimension of embeddings.

        Returns
        -------
        int
            The dimension of embeddings.
        """
        return self._embedding_dim

    @property
    def weight(self):
        """
        Return the weight tensor in distributed WholeMemoryTensor.

        Returns
        -------
        wgth.WholeMemoryTensor
            The weight tensor of the embeddings.
        """
        return self._module.wm_embedding.get_embedding_tensor()

    @property
    def optimizer(self):
        """
        Return the assoicated WholeGraph sparse optimizer

        Returns
        -------
        wgth.WholeMemoryOptimizer
            The sparse optimizer attached to the embeddings.
        """
        return self._optimizer
