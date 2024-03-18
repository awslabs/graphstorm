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
    import pylibwholegraph
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
    global WHOLEGRAPH_INIT
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
    if not is_wholegraph_init():
        raise ImportError("WholeGraph is not initialized yet.")
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
    if not is_wholegraph_init():
        raise ImportError("WholeGraph is not initialized yet.")
    return wgth.create_wholememory_optimizer(optimizer_type, param_dict)


def create_wg_dist_tensor(
    shape: tuple,
    dtype: th.dtype,
    location: str = "cpu",
    optimizer = None,
):
    """Create a WholeGraph-managed distributed tensor.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor. It has to be a two-dimensional tensor for now.
        The first dimension typically is the number of nodes.
        The second dimension is the feature/embedding dimension.
    dtype : th.dtype
        The dtype of the tensor. The data type has to be the one in the deep learning framework.
    location : str, optional
        The desired location to store the embedding [ "cpu" | "cuda" ]
    optimizer : WholeMemoryOptimizer, optional
        The attached wholegraph sparse optimizer. If None, the tensor is not trainable.
    """
    global_comm = wgth.comm.get_global_communicator()
    embedding_wholememory_type = 'distributed'
    embedding_wholememory_location = location
    assert len(shape) == 2, "The shape of the tensor must be 2D."
    wm_embedding = wgth.create_embedding(
        global_comm,
        embedding_wholememory_type,
        embedding_wholememory_location,
        dtype,
        [shape[0], shape[1]],
        optimizer=optimizer,
        cache_policy=None, # disable cache for now
        random_init=False if optimizer is None else True,
    )
    return wm_embedding


class WholeGraphDistTensor:
    """
    WholeGraph Embedding Interface for using distribute tensor in GraphStorm
    Parameters
    ----------
    shape : tuple
        The shape of the tensor. It has to be a two-dimensional tensor for now.
        The first dimension typically is the number of nodes.
        The second dimension is the feature/embedding dimension.
    dtype : th.dtype
        The dtype of the tensor. The data type has to be the one in the deep learning framework.
    location : str, optional
        The desired location to store the embedding [ "cpu" | "cuda" ]
    use_wg_optimizer : bool, optional
        Whether to use WholeGraph sparse optimizer to track/trace the gradients for WG embeddings.
        If so, defer the creation of WG tensor until the WG optimizer is created/attached.
    """
    def __init__(
        self,
        shape: tuple,
        dtype: th.dtype,
        name: str,
        location: str = "cpu",
        use_wg_optimizer: bool = False,
    ):
        self._nnodes = shape[0]
        self._embedding_dim = shape[1]
        self._name = name
        self._dtype = dtype
        self._location = location
        self._use_wg_optimizer = use_wg_optimizer
        # Need the pylibwholegraph be at least 23.12.00 to support _tensor.scatter API.
        assert pylibwholegraph.__version__ >= "23.12.00", \
            "Please upgrade to WholeGraph 23.12.00 or higher."
        self._tensor = None
        self._module = None
        self._optimizer = None
        # When _use_wg_optimizer, we have _module -> _tensor -> optimizer (-> means "depends on")
        # So, optimizer has to be created first, then tensor, then module.
        if not self._use_wg_optimizer:
            # Otherwise, we can initialize _tensor here
            self._tensor = create_wg_dist_tensor(shape, dtype, location)

    def attach_wg_optimizer(self, wg_optimizer=None):
        """
        Attach a WholeGraph sparse optimizer to the WholeGraph embedding.
        This is needed for trainable embeddings

        Parameters
        ----------
        wg_optimizer : WholeMemoryOptimizer
            The WholeGraph sparse optimizer to be attached to the WholeGraph embedding.

        Returns
        -------
        None

        """
        assert self._use_wg_optimizer, \
            "Please create WholeGraphDistTensor tensor with use_wg_optimizer=True."
        if self.optimizer == wg_optimizer and self._module is not None:
            # no-op if the optimizer is the same
            return
        assert self.optimizer is None and self._module is None, \
            "Make sure WholeGraphDistTensor attaches to only one/unique optimizer."
        # When attach an optimizer, we need to reset _tensor/_module/_optimizer.
        self._reset_storage()
        # WG sparse optimizer has to be created before WG distTensor.
        # This is because WG embedding depends on WG sparse optimizer to track/trace
        # the gradients for embeddings.
        self._optimizer = wg_optimizer
        shape = (self._nnodes, self._embedding_dim)
        self._tensor = create_wg_dist_tensor(shape, self.dtype,
                                            self._location, optimizer=wg_optimizer)
        self._module = wgth.WholeMemoryEmbeddingModule(self._tensor)

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
        assert self._tensor is not None, \
            "Please create WholeGraph tensor by either initializing WholeGraphDistTensor" \
            "with use_wg_optimizer=False or "\
            "with use_wg_optimizer=True followed by attach_wg_optimizer()."
        file_prefix = os.path.join(path, file_prefix)
        # save optimizer stats is not supported yet
        self._tensor.get_embedding_tensor().to_file_prefix(file_prefix)

    def load_from_file(
        self,
        path: str,
        file_prefix: str,
        num_files: int,
        wg_optimizer = None
    ) -> None:
        """
        Load the embedding tensor from files and attach to a wg_optimizer if presented.

        Parameters
        ----------
        path : str
            The path to the directory where the file is located.
        file_prefix : str
            The prefix of the file.
        num_files : int
            The number of files to load.
        wg_optimizer : WholeMemoryOptimizer or None
            The WholeGraph sparse optimizer to be attached to the WholeGraph embedding.
        Returns
        -------
        None
        """
        if wg_optimizer is not None:
            assert self._use_wg_optimizer, \
                "Please create WholeGraphDistTensor tensor with use_wg_optimizer=True."
            # attach to an optimizer
            self.attach_wg_optimizer(wg_optimizer)
        else:
            if self.use_wg_optimizer:
                assert self.optimizer is not None, \
                    "Need either self.optimizer or wg_optimizer to be available."
                assert self._module is not None and self._tensor is not None, \
                    "Please create WholeGraphDistTensor tensor with attach_wg_optimizer()."
            else:
                assert self.optimizer is None and self._module is None, \
                    "For regular embeddings (not trainable), self.optimizer should be None."
        # replace the existing _tensor by loading from file
        file_prefix = os.path.join(path, file_prefix)
        self._tensor.get_embedding_tensor().from_file_prefix(
            file_prefix, part_count=num_files
        )

    def __setitem__(self, idx: th.Tensor, val: th.Tensor):
        """
        Set the embeddings for the specified node indices.
        This call must be called by all processes.

        Parameters
        ----------
        idx : torch.Tensor
            Index of the embeddings to collect.
        val : torch.Tensor
            The requested node embeddings.
        """
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        idx = idx.cuda()
        val = val.cuda()

        if val.dtype != self.dtype:
            val = val.to(self.dtype)
        self._tensor.get_embedding_tensor().scatter(val, idx)

    def __getitem__(self, idx: th.Tensor) -> th.Tensor:
        """
        Get the embeddings for the specified node indices (remotely).
        This call must be called by all processes.

        Parameters
        ----------
        idx : torch.Tensor
            Index of the embeddings to collect.
        Returns
        -------
        torch.Tensor
            The requested node embeddings.
        """
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        idx = idx.cuda()
        output_tensor = self._tensor.gather(idx)  # output_tensor is on cuda by default
        return output_tensor

    def get_local_tensor(self):
        """
        Get the local embedding tensor and its element offset at current rank.

        Returns
        -------
        (torch.Tensor, int)
            Tuple of local torch Tensor (converted from DLPack) and its offset.
        """
        if self._location == "cuda":
            local_tensor, offset = self._tensor.get_embedding_tensor().get_local_tensor()
        else:
            local_tensor, offset = self._tensor.get_embedding_tensor().get_local_tensor(
                host_view=True
            )
        return local_tensor, offset

    def get_comm(self):
        """
        Get the communicator of the WholeGraph embedding.

        Returns
        -------
        WholeMemoryCommunicator
            The WholeGraph global communicator of the WholeGraph embedding.
        """
        return self._tensor.get_embedding_tensor().get_comm()

    def _reset_storage(self):
        """Reset the storage of the WholeGraph embedding."""
        self._tensor = None
        self._module = None
        self._optimizer = None

    @property
    def use_wg_optimizer(self):
        """
        Return whether the WholeGraph embedding is trainable.

        Returns:
        --------
        bool
            True if the WholeGraph embedding is trainable.
        """
        return self._use_wg_optimizer

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
    def module(self):
        """
        Return nn module wrapper for underlaying wholegraph embedding.

        Returns
        -------
        str
            The name of the embeddings.
        """
        return self._module

    @property
    def num_embeddings(self):
        """
        Return the number of embeddings.

        Returns
        -------
        int
            The number of embeddings.
        """
        return self._nnodes

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
    def dtype(self):
        """
        Return the data type of embeddings.

        Returns
        -------
        th.dtype
            The data type of embeddings.
        """
        return self._dtype

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
