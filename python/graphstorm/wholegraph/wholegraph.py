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

    Functions/classes for integrating WholeGraph into GraphStorm
"""

import os

import json
import gc
import logging
import re

import torch as th
import dgl
from dataclasses import dataclass

from ..utils import get_rank, get_world_size

try:
    import pylibwholegraph.torch as wgth
except ImportError:
    wgth = None

WHOLEGRAPH_INIT = False

def init_wholegraph():
    """ Initialize Wholegraph"""
    if wgth is None:
        raise ImportError("WholeGraph is not installed")
    from dgl.distributed import role
    import pylibwholegraph.binding.wholememory_binding as wmb
    global WHOLEGRAPH_INIT

    @dataclass
    class Options:  # pylint: disable=missing-class-docstring
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
    WHOLEGRAPH_INIT = True

def is_wholegraph_init():
    """ Query if WholeGraph is initialized """
    return WHOLEGRAPH_INIT

def wholegraph_processing(
    whole_feat_tensor, metadata, feat, wg_folder, num_parts
):
    """Convert DGL tensors to wholememory tensor

    Parameters
    ----------
    whole_feat_tensor : Tensor
        The concatenated feature tensor of different partitions
    metadata : Tensor
        Metadata of the feature tensor
    feat : str
        Name of the feature to be converted
    wg_folder : str
        Name of the folder to store the converted files
    num_parts : int
        Number of partitions of the input features
    """
    metadata[feat] = {
        "shape": list(whole_feat_tensor.shape),
        "dtype": str(whole_feat_tensor.dtype),
    }
    local_comm = wgth.comm.get_local_device_communicator()
    # Round up the integer division to match WholeGraph partitioning scheme
    subpart_size = -(whole_feat_tensor.shape[0] // -num_parts)

    for part_num in range(num_parts):
        st = part_num * subpart_size
        end = (part_num + 1) * subpart_size \
            if part_num != (num_parts - 1) \
            else whole_feat_tensor.shape[0]

        wg_tensor = wgth.create_wholememory_tensor(
            local_comm,
            "continuous",
            "cpu",
            (end - st, *whole_feat_tensor.shape[1:]),
            whole_feat_tensor.dtype,
            None,
        )
        local_tensor, _ = wg_tensor.get_local_tensor(host_view=True)
        local_tensor.copy_(whole_feat_tensor[st:end])
        filename = wgth.utils.get_part_file_name(
            feat.replace("/", "~"), part_num, num_parts
        )
        wg_tensor.local_to_file(os.path.join(wg_folder, filename))
        wgth.destroy_wholememory_tensor(wg_tensor)


def trim_feat_files(trimmed_feats, folder, file_name, part):
    """Save new truncated distDGL tensors
    Parameters
    ----------
    trimmed_feats : list of tensors
        distDGL tensors after trimming out the processed features
    folder : str
        Name of the folder of the input feature files
    file_name : str
        Name of the feature file, either node_feat.dgl or edge_feat.dgl
    part : int
        Partition number of the input feature files

    """
    dgl.data.utils.save_tensors(
        os.path.join(folder, f"part{part}", "new_" + file_name), trimmed_feats[part]
    )
    os.rename(
        os.path.join(folder, f"part{part}", file_name),
        os.path.join(folder, f"part{part}", file_name + ".bak"),
    )
    os.rename(
        os.path.join(folder, f"part{part}", "new_" + file_name),
        os.path.join(folder, f"part{part}", file_name),
    )


def convert_feat_to_wholegraph(fname_dict, file_name, metadata, folder, use_low_mem):
    """Convert features from distDGL tensor format to WholeGraph format

    Parameters
    ----------
    fname_dict: dict of list
        Dict of names of the edge features of different edge types
    file_name:
        Name of the feature file, either node_feat.dgl or edge_feat.dgl
    metadata : Tensor
        Metadata of the feature tensor
    folder: str
        Name of the folder of the input feature files
    use_low_mem: bool
        Whether to use low memory version for conversion
    """
    wg_folder = os.path.join(folder, "wholegraph")
    folder_pattern = re.compile(r"^part[0-9]+$")
    part_files = [
        f
        for f in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, f)) and folder_pattern.match(f)
    ]
    part_files = sorted(part_files, key=lambda x: int(x.split("part")[1]))
    feats_data = []

    # When 'use_low_mem' is not enabled, this code loads and appends features from individual
    # partitions. Then features are concatenated and converted into the WholeGraph format one
    # by one. The minimum memory requirement for this approach is 2X the size of the input
    # nodes or edges features in the graph.
    if not use_low_mem:
        # Read features from file
        for path in (os.path.join(folder, name) for name in part_files):
            feats_data.append(dgl.data.utils.load_tensors(f"{path}/{file_name}"))
        num_parts = len(feats_data)
        for type_name, feats in fname_dict.items():
            for feat in feats:
                feat = type_name + "/" + feat
                if feat not in feats_data[0]:
                    raise RuntimeError(
                        f"Error: Unknown feature '{feat}'. Files contain \
                                       the following features: {feats_data[0].keys()}."
                    )
                logging.info("Processing %s features...", feat)
                whole_feat_tensor = th.concat(
                    tuple(t[feat] for t in feats_data), dim=0
                )
                # Delete processed feature from memory
                for t in feats_data:
                    del t[feat]
                wholegraph_processing(
                    whole_feat_tensor, metadata, feat, wg_folder, num_parts
                )
        # Trim the original distDGL tensors
        for part in range(num_parts):
            trim_feat_files(feats_data, folder, file_name, part)

    # This low-memory version loads one partition at a time. It processes features one by one,
    # iterating through all the partitions and appending only the current feature, converting
    # it to a WholeGraph. The minimum memory requirement for this approach is 2X the size of
    # the largest node or edge feature in the graph.
    else:  # low-mem
        for ntype, feats in fname_dict.items():
            for feat in feats:
                feat = ntype + "/" + feat
                node_feats_data = None
                num_parts = 0
                # Read features from file
                for path in (os.path.join(folder, name) for name in part_files):
                    nfeat = dgl.data.utils.load_tensors(f"{path}/{file_name}")
                    if feat not in nfeat:
                        raise RuntimeError(
                            f"Error: Unknown feature '{feat}'. Files contain \
                                       the following features: {nfeat.keys()}."
                        )
                    if node_feats_data is None:
                        node_feats_data = nfeat[feat]
                    else:
                        node_feats_data = th.concat((node_feats_data, nfeat[feat]), dim=0)
                    num_parts += 1
                del nfeat
                gc.collect()
                wholegraph_processing(
                    node_feats_data,
                    metadata,
                    feat,
                    wg_folder,
                    num_parts,
                )
        num_parts = 0
        for path in (os.path.join(folder, name) for name in part_files):
            feats_data = dgl.data.utils.load_tensors(f"{path}/{file_name}")
            for type_name, feats in fname_dict.items():
                for feat in feats:
                    feat = type_name + "/" + feat
                    # Delete processed feature from memory
                    del feats_data[feat]
            num_parts += 1
            trim_feat_files(feats_data, folder, file_name, num_parts)


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
):
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
    optimizer,  # Optional[wgth.WholeMemoryOptimizer, None]
    location: str = "cpu",
):
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
        optimizer,  # Optional[wgth.WholeMemoryOptimizer, None]
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
