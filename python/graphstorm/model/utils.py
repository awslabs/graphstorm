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

    Utility functions and classes.
"""
import os
import math
import json
import shutil
import logging

import torch as th
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import dgl

from ..config import GRAPHSTORM_LP_EMB_L2_NORMALIZATION
from ..gconstruct.file_io import stream_dist_tensors_to_hdf5
from ..utils import (
    get_rank,
    barrier,
    get_world_size,
    create_dist_tensor,
)
from ..wholegraph import WholeGraphDistTensor, create_wholememory_optimizer
from ..data.utils import alltoallv_cpu, alltoallv_nccl
from ..distributed import flush_data

# placeholder of the ntype for homogeneous graphs
NTYPE = dgl.NTYPE

def pad_file_index(file_index, width=5):
    """ Left pad file_index with zerros.

        for examaple, given 1, it will return 00001.

        Parameters
        ----------
        file_index: int
            Index of the file
        width: int
            Minimum length of resulting string; strings with length less
            than width be prepended with 0 characters.

        Return
        ------
        str: padded file_index
    """
    assert width > 1, "Width should be larger than 1"
    return str(file_index).zfill(width)

def sparse_emb_initializer(emb):
    """ Initialize sparse embedding

        Parameters
        ----------
        emb: th.Tensor
            Tensor to initialize

        Returns
        -------
        Initialized tensor: th.Tensor
    """
    th.nn.init.xavier_uniform_(emb)
    return emb

def save_model(model_path, gnn_model=None, embed_layer=None, decoder=None):
    """ A model should have three parts:
        * GNN model
        * embedding layer
        The model is only used for inference.

        Parameters
        ----------
        model_path: str
            The path of the model is saved.
        gnn_model: model
            A (distributed) model of GNN
        embed_layer: model
            A (distributed) model of embedding layers.
        decoder: model
            A (distributed) model of decoder
    """
    model_states = {}
    if gnn_model is not None and isinstance(gnn_model, nn.Module):
        model_states['gnn'] = gnn_model.state_dict()
    if embed_layer is not None and isinstance(embed_layer, nn.Module):
        model_states['embed'] = embed_layer.state_dict()
    if decoder is not None and isinstance(decoder, nn.Module):
        model_states['decoder'] = decoder.state_dict()

    os.makedirs(model_path, exist_ok=True)
    # [04/16]: Assume this method is called by rank 0 who can perform chmod
    assert get_rank() == 0, "Only can rank 0 process change folders mode."
    # mode 767 means rwx-rw-rwx:
    #     - owner of the folder can read, write, and execute;
    #     - owner' group can read, write;
    #     - others can read, write, and execute.
    os.chmod(model_path, 0o767)
    th.save(model_states, os.path.join(model_path, 'model.bin'))

def save_model_results_json(conf, test_model_performance, save_perf_results_path):
    """
    This function writes the model configuration and the test metric results to a JSON file
    Args:
        conf: the model configuration
        test_model_performance: the final best test metric

    Returns:

    """
    model_results_and_conf = {"configuration": conf,
                              "test-model-performance": test_model_performance}

    if save_perf_results_path is not None:
        json_path = os.path.join(save_perf_results_path, 'performance_results.json')
        if not os.path.isdir(save_perf_results_path):
            os.makedirs(save_perf_results_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(model_results_and_conf, f, ensure_ascii=False, indent=4)

def _get_sparse_emb_range(num_embs, rank, world_size):
    """ Provide a deterministic method to split trainable sparse embeddings
        during saveing and loading according local rank and world size.

        Parameters
        ----------
        num_embs:
            Size of a sparse embedding
        rank: int
            Rank of the current process in a distributed environment.
        world_size : int
            World size in a distributed environment. This tells the size of a distributed cluster
            (How many processes in a cluster).
    """
    assert rank < world_size, \
        "local rank {rank} shold be smaller than world size {world_size}"
    # Get corresponding data range
    if num_embs < world_size:
        start = rank if rank < num_embs else num_embs
        end = rank + 1 if rank < num_embs else num_embs
    else:
        start = rank * math.ceil(num_embs / world_size)
        end = (rank + 1) * math.ceil(num_embs / world_size)
        end = num_embs if rank + 1 == world_size else end
    return start, end

def save_sparse_emb(model_path, sparse_emb, ntype):
    """ Save sparse emb `sparse_emb` into `model_path`

        Sparse embeddings are stored as:
            $model_path/$ntype/sparse_emb_0.pt
            $model_path/$ntype/sparse_emb_1.pt
            ...

        Example:
        --------
        Save sparse embeddings of an input embed_layer by calling ``save_sparse_emb``
        iterating all the sparse embeddings of the embed_layer

        .. code::

            # embed_layer is the input embed_layer
            embed_layer = embed_layer.module \
                if isinstance(embed_layer, DistributedDataParallel) \
                else embed_layer

            for ntype, sparse_emb in embed_layer.sparse_embeds.items():
                save_sparse_emb(model_path, sparse_emb, ntype)

        Parameters
        ----------
        model_path: str
            The path of the model is saved.
        sparse_emb: dgl.distributed.DistEmbedding or wholegraph.WholeGraphDistTensor
            A Distributed node embedding.
        ntype: str
            The node type the embedding belongs to.
    """
    rank = get_rank()
    world_size = get_world_size()
    num_embs = sparse_emb.num_embeddings
    start, end = _get_sparse_emb_range(num_embs, rank, world_size)
    # collect sparse_emb in a iterative way

    # In distributed mode where uses an NFS folder, directly call this makedirs method to
    # create spare embedding path will cause folder permission error that prevents
    # non-rank 0 process from saving embeddings. Therefore, need rank 0 process to call
    # the create_sparse_embeds_path() method first before calling save_sparse_embeds().
    emb_path = os.path.join(model_path, ntype)
    os.makedirs(emb_path, exist_ok=True)

    if isinstance(sparse_emb, WholeGraphDistTensor):
        (local_tensor, _) = sparse_emb.get_local_tensor()
        # Using WholeGraph will save sparse emb in binary format (evenly distributed)
        # Example: wg_sparse_emb_part_1_of_2, wg_sparse_emb_part_2_of_2
        assert (
            local_tensor.shape[0] == end - start
        ), "WholeGraph tensor local boundary has invalid dimensions."
        sparse_emb.save_to_file(emb_path, file_prefix="wg_sparse_emb")
    else:
        emb_file_path = os.path.join(emb_path, f"sparse_emb_{pad_file_index(rank)}.pt")
        embs = []
        batch_size = 10240
        # TODO: dgl.distributed.DistEmbedding should provide emb.shape

        idxs = th.split(th.arange(start=start, end=end), batch_size, dim=0)
        for idx in idxs:
            # TODO: dgl.distributed.DistEmbedding should allow some basic tensor ops
            embs.append(sparse_emb._tensor[idx])

        embs = th.cat(embs, dim=0)
        th.save(embs, emb_file_path)


def save_sparse_embeds(model_path, embed_layer):
    """ save sparse embeddings if embed_layer has any

        Sparse embeddings are stored as:
        $model_path/ntype0/sparse_emb_0.pt
                           ...
                           sparse_emb_N.pt
        $model_path/ntype1/sparse_emb_0.pt
                           ...
                           sparse_emb_N.pt
        ...

        Parameters
        ----------
        model_path: str
            The path of the model is saved.
        embed_layer: model
            A (distributed) model of embedding layers.
    """
    if embed_layer is None:
        return
    embed_layer = embed_layer.module \
        if isinstance(embed_layer, DistributedDataParallel) else embed_layer

    if len(embed_layer.sparse_embeds) > 0:
        for ntype, sparse_emb in embed_layer.sparse_embeds.items():
            save_sparse_emb(model_path, sparse_emb, ntype)

def save_opt_state(model_path, dense_opts, lm_opts, sparse_opts):
    """ Save the states of the optimizers.

        Parameters
        ----------
        model_path : str
            The path of the folder where the model is saved.
            We save the optimizer states with the model.
        dense_opts : list of optimizer
            The optimizers for dense model parameters.
        lm_opts: list of optimizer
            The optimizers for language model parameters.
        emb_opts : list of optimizer
            The optimizers for sparse embedding layer.
    """
    opt_states = {}

    assert len(dense_opts) <= 1, "We can only support one dense optimizer now."
    assert len(lm_opts) <= 1, "We can only support one language model optimizer now."

    if len(dense_opts) == 1:
        opt_states['dense'] = dense_opts[0].state_dict()
    if len(lm_opts) == 1:
        opt_states['lm'] = lm_opts[0].state_dict()
    # TODO(zhengda) we need to change DGL to make it work.
    if len(sparse_opts) > 0:
        # TODO(xiangsx) Further discussion of whether we need to save the state of
        #               sparse optimizer is needed.
        logging.warning("We do not export the state of sparse optimizer")
    os.makedirs(model_path, exist_ok=True)
    th.save(opt_states, os.path.join(model_path, 'optimizers.bin'))

def save_relation_embeddings(emb_path, decoder):
    """ Save relation embeddings

        This function is called only when decoder has relation embeds

        Parameters
        ----------
        emb_path: str
            The path to save embedding
        decoder: Decoder
            Link predicition decoder
    """
    assert hasattr(decoder, "get_relembs"), \
        "Decoder must implement get_relembs()"
    relembs, et2id_map = decoder.get_relembs()
    relembs = relembs.detach().cpu()
    with open(os.path.join(emb_path, 'relation2id_map.json'), "w", encoding='utf-8') as f:
        et2id_map = {str(key): val for key, val in et2id_map.items()}
        json.dump(et2id_map, f, ensure_ascii=False, indent=4)
    th.save(relembs, os.path.join(emb_path, "rel_emb.pt"))

def get_data_range(rank, world_size, num_embs):
    """ save_embeddings will evenly split node embeddings across all
        the workers to save. This function returns the data range according
        to the current worker rank and the total number of nodes (embeddings).

        Parameters
        ----------
        rank: int
            Current worker rank
        world_size: int
            Total number of workers
        num_embs: int
            Number of node embeddings.

        Return
        ------
        start: int
            Starting node idx of the current embedding data range
        end: int
            Ending node idx of the current embedding data range.
    """
    assert rank < world_size, \
        f"Rank {rank} of a worker should be smaller than the cluster size {world_size}"

    # Get corresponding data range
    start = rank * (num_embs // world_size)
    end = (rank + 1) * (num_embs // world_size)
    end = num_embs if rank + 1 == world_size else end
    return start, end

def _exchange_node_id_mapping(rank, world_size, device,
    node_id_mapping, num_embs):
    """ Rank0 loads node_id_mappings and spreads it to other ranks.
        Each rank will get a sub-range of node_id_mappings.
        We use alltoall_v to send sub-node_id_mappings to each rank.

        Parameters
        ----------
        rank: int
            Rank of the current process in a distributed environment.
        world_size : int
            World size in a distributed environment. This tells the size of a distributed cluster
            (How many processes in a cluster).
        device: torch device
            Device used for all_to_allv data exchange. For gloo backend
            we store data in CPU, For nccl backend, we need to store
            data in GPU.
        node_id_mapping_file: str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.

        Return:
        Tensor: sub node_id_mappings corresponding to `rank`
    """
    backend = th.distributed.get_backend()
    device = th.device('cpu') if backend == "gloo" else device

    if rank == 0:
        data_tensors = []
        for i in range(world_size):
            start_idx, end_idx = get_data_range(i, world_size, num_embs)
            data_tensors.append(
                node_id_mapping[start_idx:end_idx].to(device))
    else:
        data_tensors = [th.empty((0,),
                                    dtype=th.long,
                                    device=device) \
            for _ in range(world_size)]

    start_idx, end_idx = get_data_range(rank, world_size, num_embs)
    gather_list = \
        [th.empty((end_idx-start_idx,),
                    dtype=th.long,
                    device=device) \
            if i == 0 else th.empty((0,),
                                    dtype=th.long,
                                    device=device) \
            for i in range(world_size)]
    if backend == "gloo":
        alltoallv_cpu(rank, world_size, gather_list, data_tensors)
    else: # backend == "nccl"
        alltoallv_nccl(gather_list, data_tensors)
    # move mapping into CPU
    return gather_list[0].to(th.device("cpu"))

def _load_dist_nid_map(node_id_mapping_file, ntypes):
    """ Load id mapping files in dist partition format.
    """
    # node_id_mapping_file it is actually a directory
    # <node_id_mapping_file>/part0, <node_id_mapping_file>/part1, ...
    part_dirs = [part_path for part_path in os.listdir(node_id_mapping_file) \
                if part_path.startswith("part")]

    # we need the mapping chunks are ordered like part0, part1, ...
    id_mappings = {ntype:[] for ntype in ntypes}
    for i in range(len(part_dirs)):
        id_mapping_part = dgl.data.utils.load_tensors(
            os.path.join(node_id_mapping_file, f"part{i}", "orig_nids.dgl"))
        for ntype in ntypes:
            id_mappings[ntype].append(id_mapping_part[ntype])
    id_mappings = {
        ntype: th.cat(mappings) for ntype, mappings in id_mappings.items()
    }

    return id_mappings

def distribute_nid_map(embeddings, rank, world_size,
    node_id_mapping_file, device=th.device('cpu')):
    """ Distribute nid_map to all workers.

        Parameters
        ----------
        embeddings : DistTensor
            Embeddings to save
        rank : int
            Local rank
        world_size : int
            World size in a distributed env.
        node_id_mapping_file: str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
        device: torch device
            Device used for all_to_allv data exchange. For gloo backend
            we store data in CPU, For nccl backend, we need to store
            data in GPU.

        Returns
        _______
        Dict of list: Embeddings index from original order.
    """
    assert node_id_mapping_file is not None
    if isinstance(embeddings, (dgl.distributed.DistTensor, LazyDistTensor)):
        # only host 0 will load node id mapping from disk
        if rank == 0:
            if node_id_mapping_file.endswith("pt"):
                ori_node_id_mapping = th.load(node_id_mapping_file)
            else:
                # Homogeneous graph
                # node id mapping file from dgl tools/distpartitioning/convert_partition.py.
                ori_node_id_mapping = _load_dist_nid_map(node_id_mapping_file, ["_N"])["_N"]
            _, node_id_mapping = th.sort(ori_node_id_mapping)
        else:
            node_id_mapping = None

        nid_mapping = _exchange_node_id_mapping(
            rank, world_size, device, node_id_mapping, len(embeddings))
    elif isinstance(embeddings, dict):
        nid_mapping = {}
        # only host 0 will load node id mapping from disk
        if rank == 0:
            if node_id_mapping_file.endswith("pt"):
                node_id_mappings = th.load(node_id_mapping_file)
            else:
                # node id mapping file from dgl tools/distpartitioning/convert_partition.py.
                node_id_mappings = _load_dist_nid_map(node_id_mapping_file,
                                                      list(embeddings.keys()))
        else:
            node_id_mappings = None

        for name, emb in embeddings.items():
            if rank == 0:
                assert name in node_id_mappings, \
                    f"node id mapping for ntype {name} should exists"
                # new mapping back index
                ori_node_id_mapping = node_id_mappings[name]
                _, node_id_mapping = th.sort(ori_node_id_mapping)
            else:
                node_id_mapping = None

            nid_mapping[name] = _exchange_node_id_mapping(
                rank, world_size, device, node_id_mapping, len(emb))
    else:
        nid_mapping = None
    return nid_mapping

def remap_embeddings(embeddings, rank, world_size,
    node_id_mapping_file, device=th.device('cpu')):
    """ Remap embeddings by nid_map without writing to disk.

        Parameters
        ----------
        embeddings : DistTensor
            Embeddings to save
        rank : int
            Local rank
        world_size : int
            World size in a distributed env.
        node_id_mapping_file: str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
        device: torch device
            Device used for all_to_allv data exchange. For gloo backend
            we store data in CPU, For nccl backend, we need to store
            data in GPU.

        Returns
        _______
        DistTensor : remapped DistTensor
    """
    assert node_id_mapping_file is not None

    # TODO: handle when node_id_mapping_file is None.
    nid_mapping = distribute_nid_map(embeddings, rank, world_size,
            node_id_mapping_file, device)

    if isinstance(embeddings, (dgl.distributed.DistTensor, LazyDistTensor)):
        start, end = get_data_range(rank, world_size, len(embeddings))
        embeddings[list(range(start, end))] = embeddings[nid_mapping]
    elif isinstance(embeddings, dict):
        # We need to duplicate the dict so that the input argument is not changed.
        embeddings = dict(embeddings.items())
        for name, emb in embeddings.items():
            if isinstance(emb, (dgl.distributed.DistTensor, LazyDistTensor)):
                # this is the same window as nid_mapping
                start, end = get_data_range(rank, world_size, len(emb))
                # we need to keep emb to be dist tensor unchanged
                emb[th.arange(start, end)] = emb[nid_mapping[name]]
            barrier()

    return embeddings

def save_pytorch_embedding(emb_path, embedding, rank, world_size):
    """ Save Dist embedding tensor in Pytorch format.

        Parameters
        ----------
        emb_path : str
            The path of the save embedding files.
        embedding : DistTensor
            The Dist tensor to save.
        rank : int
            Rank of the current process in a distributed environment.
        world_size : int
            World size in a distributed env.
    """
    os.makedirs(emb_path, exist_ok=True)
    # [04/16]: Only rank 0 can chmod to let all other ranks to write files.
    if rank == 0:
        # mode 767 means rwx-rw-rwx:
        #     - owner of the folder can read, write, and execute;
        #     - owner' group can read, write;
        #     - others can read, write, and execute.
        os.chmod(emb_path, 0o767)

    # make sure the emb_path permission is changed before other process start to save
    barrier()

    assert rank < world_size, \
        f"Process rank {rank} must be smaller than the distributed cluster size {world_size}"

    assert isinstance(embedding, (dgl.distributed.DistTensor, LazyDistTensor)), \
        "Input embedding must be a dgl.distributed.DistTensor or a LazyDistTensor"

    start, end = get_data_range(rank, world_size, len(embedding))
    embedding = embedding[start:end]
    th.save(embedding, os.path.join(emb_path, f'embed-{pad_file_index(rank)}.pt'))

def save_wholegraph_embedding(emb_path, embedding, rank, world_size, fmt="binary"):
    """ Save Dist embedding tensor in binary format for WholeGraph.

        Parameters
        ----------
        emb_path : str
            The path of the save embedding files.
        embedding : WholeGraphDistTensor
            The WholeGraph dist tensor to save.
        rank : int
            Rank of the current process in a distributed environment.
        world_size : int
            World size in a distributed env.
        fmt : str
            The format of the saved embeddings. Currently only support "binary" and "pytorch".
    """
    assert fmt in ["binary", "pytorch"], \
        "Using WholeGraph, the supported formats of the saved embeddings " + \
        "are 'binary' and 'pytorch'."
    os.makedirs(emb_path, exist_ok=True)
    # [04/16]: Only rank 0 can chmod to let all other ranks to write files.
    if rank == 0:
        # mode 767 means rwx-rw-rwx:
        #     - owner of the folder can read, write, and execute;
        #     - owner' group can read, write;
        #     - others can read, write, and execute.
        os.chmod(emb_path, 0o767)

    # make sure the emb_path permission is changed before other process start to save
    barrier()

    assert rank < world_size, \
        f"Process rank {rank} must be smaller than the distributed cluster size {world_size}"

    assert isinstance(embedding, WholeGraphDistTensor), \
        "Input embedding must be a WholeGraphDistTensor."

    emb_num = embedding.num_embeddings
    emb_dim = embedding.embedding_dim
    emb_dtype = embedding.dtype
    emb_name = embedding.name
    emb_fmt = "wholegraph-" + fmt
    emb_info = {
        "format": emb_fmt,
        "emb_num": str(emb_num),
        "emb_dim": str(emb_dim),
        "emb_dtype": str(emb_dtype),
        "emb_name": emb_name,
        "world_size": world_size
    }
    if fmt == "binary":
        # use binary format to save the embedding (supported by native WholeGraph APIs)
        # Example: wg-embed_part_0_of_2, wg-embed_part_1_of_2
        # Pros: WholeGraph's natvie API to load the embedding directly.
        #       no RAM duplication; support save/load with different world_size.
        embedding.save_to_file(emb_path, file_prefix="wg-embed")
    elif fmt == "pytorch":
        # use pytorch format to save the embedding (dump local tensor to pt file)
        # Example: embed-00000.pt, embed-00001.pt
        # Pros: Compatible with the format when WholeGraph is not enabled,
        #       but still follows wholegraph's even partition policy and duplicate RAM when load.
        emb = embedding.get_local_tensor()[0]
        wg_rank = embedding.get_comm().get_rank()
        th.save(emb, os.path.join(emb_path, f'embed-{pad_file_index(wg_rank)}.pt'))

    if rank == 0:
        with open(os.path.join(emb_path, "emb_info.json"), 'w', encoding='utf-8') as f:
            json.dump(emb_info, f, indent=4)

def load_wholegraph_embedding(emb_path, name):
    """ Load embedding tensor in binary format for WholeGraph.

    Parameters
    ----------
    emb_path : str
        The path of the save embedding files.
    part_policy : dgl.distributed.PartitionPolicy
        The partitioning policy
    name : str
        The name of the created distributed tensor.

    Returns
    -------
    WholeGraphDistTensor : the loaded embeddings in WholeGraph.
    """
    file_path = os.path.join(emb_path, "emb_info.json")
    assert os.path.exists(file_path), \
        f"Embedding JSON file: {file_path} not found. " + \
        "This file is needed for storing embedding with WholeGraph. It's generated when " + \
        "you save embeddings with '--use-wholegraph-embed' flag."
    with open(file_path, 'r', encoding='utf-8') as f:
        emb_info = json.load(f)

    emb_fmt = emb_info['format']
    assert emb_fmt.startswith("wholegraph-"), \
        "The format of the saved embeddings should be started with 'wholegraph-'."
    emb_fmt = emb_fmt.split("-")[1]
    emb_num = int(emb_info['emb_num'])
    emb_dim = int(emb_info['emb_dim'])
    world_size_in_save = int(emb_info['world_size'])
    supported_dtypes = {
        'torch.half': th.half,
        'torch.float16': th.float16,
        'torch.float32': th.float32,
        'torch.float': th.float,
        'torch.int64': th.int64,
        'torch.int32': th.int32
    }
    emb_dtype = supported_dtypes[emb_info['emb_dtype']]
    dist_emb = WholeGraphDistTensor((emb_num, emb_dim), emb_dtype, name=name)
    if emb_fmt == "pytorch":
        assert dist_emb.get_comm().get_size() == world_size_in_save, \
            "World_size when save the embedding is different than the current world_size. " \
            "Please switch to the binary format."
        wg_rank = dist_emb.get_comm().get_rank()
        file_path = os.path.join(emb_path, f'embed-{pad_file_index(wg_rank)}.pt')
        assert os.path.exists(file_path), f"Embedding file {file_path} of \
            my rank {wg_rank} doesn't exist."
        emb = th.load(file_path)
        local_emb = dist_emb.get_local_tensor()[0]
        assert emb.shape[0] == local_emb.shape[0] and emb.shape[1] == local_emb.shape[1], \
            f"Embedding shape of {name} does not match! " + \
            f"Expect {emb.shape}, but get {local_emb.shape}"

        assert emb.dtype == local_emb.dtype, "Embedding datatype do not match!"
        local_emb.copy_(emb)
    elif emb_fmt == "binary":
        files = os.listdir(emb_path)
        filtered_files = [file for file in files if file.startswith("wg-embed")]
        num_files = len(filtered_files)
        assert num_files > 0, "No WholeGraph embedding files found."
        assert world_size_in_save == num_files, \
            f"World_size when save the embedding  {world_size_in_save} \
            doesn't match the number of files {num_files}."
        dist_emb.load_from_file(emb_path, "wg-embed", num_files)

    barrier()
    return dist_emb

def load_pytorch_embedding(emb_path, part_policy, name):
    """ Load embedding tensor in Pytorch format.

    Parameters
    ----------
    emb_path : str
        The path of the save embedding files.
    part_policy : dgl.distributed.PartitionPolicy
        The partitioning policy
    name : str
        The name of the created distributed tensor.

    Returns
    -------
    DistTensor : the loaded embeddings.
    """
    rank = get_rank()
    world_size = get_world_size()
    emb = th.load(os.path.join(emb_path, f'embed-{pad_file_index(rank)}.pt'))
    dist_emb = create_dist_tensor((part_policy.get_size(), emb.shape[1]), emb.dtype,
            name=name, part_policy=part_policy)
    start, end = get_data_range(rank, world_size, len(dist_emb))
    assert end - start == emb.shape[0], \
            f"The loaded BERT embeddings have {emb.shape[0]} rows, " + \
            f"doesn't match the required data range [{start}, {end}]."
    dist_emb[start:end] = emb
    barrier()
    return dist_emb

def save_pytorch_embeddings(emb_path, embeddings, rank, world_size,
    device=th.device('cpu'), node_id_mapping_file=None):
    """ Save node embeddings as pytorch tensors in a distributed way.

        The input node `embeddings` are stored in Partition Node ID space.
        When `node_id_mapping_file` is provided (GraphStorm graph processing
        pipeline automatically generate node id mapping files by default),
        `save_pytorch_embeddings` will shuffle the order of
        node embeddings so that they are stored in Graph Node ID space.

        The node embeddings are stored into multiple pytorch files.

        Example:
        --------
        The saved node embeddings looks like:

        .. code::

            PATH_TO_EMB:
                |- emb_info.json
                |- ntype0_embed-00000.pt
                |- ...
                |- ntype1_embed-00000.pt
                |- ...

        The emb.info.json contains three information:
            * "format", how data are stored, e.g., "pytorch".
            * "world_size", the total number of file parts. 0 means there is no partition.
            * "emb_name", a list of node types that have embeddings saved.

        Example:
        --------
        .. code::

            {
                "format": "pytorch",
                "world_size": 8,
                "emb_name": ["movie", "user"]
            }

        The order of embeddings are sorted according to the node IDs in
        Graph Node ID space.

        Example:
        --------

        .. code::

            Graph Node ID   |   embeddings
            0               |   0.112,0.123,-0.011,...
            1               |   0.872,0.321,-0.901,...
            2               |   0.472,0.432,-0.732,...
            ...

        An alternative way to save node embeddings is calling `save_full_node_embeddings`
        which is recommended as it is more efficient. Please refer to `save_full_node_embeddings`
        for more details.

        Parameters
        ----------
        emb_path : str
            The path of the folder where the embeddings are saved.
        embeddings : DistTensor or dict of DistTensor
            Embeddings to save
        rank : int
            Rank of the current process in a distributed environment.
        world_size : int
            World size in a distributed env.
        device: torch device
            Device used for all_to_allv data exchange. For gloo backend
            we store data in CPU, For nccl backend, we need to store
            data in GPU.
        node_id_mapping_file: str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
    """
    # [04/16]: Only rank 0 can chmod to let all other ranks to write files.
    if rank == 0:
        # mode 767 means rwx-rw-rwx:
        #     - owner of the folder can read, write, and execute;
        #     - owner' group can read, write;
        #     - others can read, write, and execute.
        os.chmod(emb_path, 0o767)

    # make sure the emb_path permission is changed before other process start to save
    barrier()

    assert rank < world_size
    # Node ID mapping won't be very large if number of nodes is
    # less than 10 billion. An ID mapping of 10 billion nodes
    # will take around 80 GByte.
    if node_id_mapping_file is not None:
        nid_mapping = distribute_nid_map(embeddings, rank, world_size,
            node_id_mapping_file, device)
    else:
        nid_mapping = None

    if isinstance(embeddings, (dgl.distributed.DistTensor, LazyDistTensor)):
        if nid_mapping is None:
            start, end = get_data_range(rank, world_size, len(embeddings))
            embeddings = embeddings[start:end]
        else:
            embeddings = embeddings[nid_mapping]
    elif isinstance(embeddings, dict):
        # We need to duplicate the dict so that the input argument is not changed.
        embeddings = dict(embeddings.items())
        for name, emb in embeddings.items():
            if isinstance(emb, (dgl.distributed.DistTensor, LazyDistTensor)):
                if nid_mapping is None:
                    start, end = get_data_range(rank, world_size, len(emb))
                    emb = emb[start:end]
                else:
                    emb = emb[nid_mapping[name]]
                embeddings[name] = emb

    emb_info = {
        "format": "pytorch",
        "emb_name":[], # This is telling how many node types have node embeddings
        "world_size": world_size
    }

    if isinstance(embeddings, dict):
        # embedding per node type
        for name, emb in embeddings.items():
            os.makedirs(os.path.join(emb_path, name), exist_ok=True)
            th.save(emb, os.path.join(os.path.join(emb_path, name),
                                      f'embed-{pad_file_index(rank)}.pt'))
            emb_info["emb_name"].append(name)
    else:
        os.makedirs(os.path.join(emb_path, NTYPE), exist_ok=True)
        # There is no ntype for the embedding
        # use NTYPE
        th.save(embeddings, os.path.join(os.path.join(emb_path, NTYPE),
                                         f'embed-{pad_file_index(rank)}.pt'))
        emb_info["emb_name"] = NTYPE

    if rank == 0:
        with open(os.path.join(emb_path, "emb_info.json"), 'w', encoding='utf-8') as f:
            json.dump(emb_info, f, indent=4)

def save_hdf5_embeddings(emb_path, embeddings, rank, world_size,
    device=th.device('cpu'), node_id_mapping_file=None):
    """ Save embeddings through hdf5 into a single file.

        Parameters
        ----------
        emb_path : str
            The path of the folder where the embeddings are saved.
        embeddings : DistTensor
            Embeddings to save
        rank : int
            Rank of the current process in a distributed environment.
        world_size : int
            World size in a distributed env.
        device: torch device
            Device used for all_to_allv data exchange. For gloo backend
            we store data in CPU, For nccl backend, we need to store
            data in GPU.
        node_id_mapping_file: str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
    """
    mapped_embeds = remap_embeddings(embeddings, rank, world_size, node_id_mapping_file, device)
    if rank == 0:
        stream_dist_tensors_to_hdf5(mapped_embeds, os.path.join(emb_path, "embed_dict.hdf5"))
        emb_info = {
            "format": "hdf5",
            "world_size":0
        }
        with open(os.path.join(emb_path, "emb_info.json"), 'w', encoding='utf-8') as f:
            json.dump(emb_info, f, indent=4)

def save_shuffled_node_embeddings(shuffled_embs, save_embed_path, save_embed_format="pytorch"):
    """ Save node embeddings that have corresponding node IDs shuffled into Graph
        Node ID space.

        For each node embeddings, two tensors are required and should be
        provided as a tuple: (embedding tensor, node ID tensor).
        The node ID tensor stores node IDs in Graph Node ID space.

        Parameters
        ----------
        shuffled_embs: dict of tuple of tensors
            Embeddings and their associated node ids to be saved
        save_embed_path: str
            Path to save the embeddings
        save_embed_format : str
            The format of saved embeddings.
            Currently support ["pytorch"].
    """
    os.makedirs(save_embed_path, exist_ok=True)
    assert save_embed_format == "pytorch", \
        "save_shuffled_node_embeddings only supports pytorch format now."
    rank = get_rank()
    world_size = get_world_size()
    # [04/16]: Only rank 0 can chmod to let all other ranks to write files.
    if rank == 0:
        # mode 767 means rwx-rw-rwx:
        #     - owner of the folder can read, write, and execute;
        #     - owner' group can read, write;
        #     - others can read, write, and execute.
        os.chmod(save_embed_path, 0o767)
        logging.info("Writing GNN embeddings to "\
            "%s in pytorch format.", save_embed_path)

    # make sure the save_embed_path permission is changed before other process start to save
    barrier()

    emb_info = {
        "format": "pytorch",
        "emb_name":[],
        "world_size":world_size
    }

    for ntype, (embs, nids) in shuffled_embs.items():
        os.makedirs(os.path.join(save_embed_path, ntype), exist_ok=True)
        assert len(nids) == len(embs), \
            f"The embeding length {len(embs)} does not match the node id length {len(nids)}"
        th.save(embs, os.path.join(os.path.join(save_embed_path, ntype),
                                  f'embed-{pad_file_index(rank)}.pt'))
        th.save(nids, os.path.join(os.path.join(save_embed_path, ntype),
                                  f'embed_nids-{pad_file_index(rank)}.pt'))
        emb_info["emb_name"].append(ntype)

    if rank == 0:
        with open(os.path.join(save_embed_path, "emb_info.json"), 'w', encoding='utf-8') as f:
            json.dump(emb_info, f, indent=4)

def save_full_node_embeddings(g, save_embed_path,
                              embeddings,
                              node_id_mapping_file,
                              save_embed_format="pytorch"):
    """ Save all node embeddings with node IDs in Graph Node ID space.

        The input node `embeddings` are stored in Partition Node ID space.
        By default, `save_full_node_embeddings` will translate the node IDs
        from Partition Node ID space into their counterparts in Graph Node
        ID space.

        `save_full_node_embeddings` will save two information of an
        embedding: 1) the embedding and 2) its corresponding node ID
        in Graph Node ID space.
        It assumes the input `embeddings` are stored in Partition Node
        ID space and the IDs start from 0 to N. It will call NodeIDShuffler
        to shuffle the node IDs from Partition Node ID space into Graph
        Node ID space.

        The saved node embeddings are in the following format:

        Example
        --------
        # embedddings:
        #   ntype0:
        #     embed_nids-00000.pt
        #     embed_nids-00001.pt
        #     ...
        #     embed-00000.pt
        #     embed-00001.pt
        #     ...
        #   ntype1:
        #     embed_nids-00000.pt
        #     embed_nids-00001.pt
        #     ...
        #     embed-00000.pt
        #     embed-00001.pt
        #     ...

        The content of embed_nids- files and embed- files looks like:

        Example:
        --------

        .. code::

            embed_nids-00000.pt    |   embed-00000.pt
                                 |
            Graph Node ID        |   embeddings
            10                   |   0.112,0.123,-0.011,...
            1                    |   0.872,0.321,-0.901,...
            23                   |   0.472,0.432,-0.732,...
            ...

        Note: `save_pytorch_embeddings` (called by `save_embeddings`) is different from
        `save_full_node_embeddings`. In `save_pytorch_embeddings`, it will shuffle the
        order of node embeddings so that the node embeddings are shuffled according to
        node IDs in Graph Node ID space. While `save_full_node_embeddings`
        shuffles node IDs instead of node embeddings, which is more efficient.

        Note: Users need to call graphstorm.gcostruct.remap_result to remap the output
        of `save_full_node_embeddings` from Graph Node ID space to Raw Node ID space.
        GraphStorm's launch scripts will automatically call remap_result by default.

        Parameters
        ----------
        g: DGLGraph
            The graph
        save_embed_path : str
            The path of the folder where the embeddings are saved.
        embeddings : DistTensor or dict of DistTensor
            Embeddings to save
        node_id_mapping_file : str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
        save_embed_format : str
            The format of saved embeddings.
            Currently support ["pytorch"].
    """
    assert save_embed_format in ["pytorch"], \
        "Only support save embeddings in the format of ['pytorch']"
    ntypes = list(embeddings.keys())
    nid_shuffler = NodeIDShuffler(g, node_id_mapping_file, ntypes) \
                if node_id_mapping_file else None

    pb = g.get_partition_book()
    shuffled_embs = {}
    for ntype in ntypes:
        if get_rank() == 0:
            logging.info("save embeddings of %s to %s", ntype, save_embed_path)

        # only save embeddings of target_nidx
        assert ntype in embeddings, \
            f"{ntype} is not in the set of evaluation ntypes {ntypes}"
        emb_nids = \
            dgl.distributed.node_split(th.full((g.num_nodes(ntype),), True, dtype=th.bool),
                                       pb, ntype=ntype, force_even=True)
        emb = embeddings[ntype][emb_nids]
        if nid_shuffler is not None:
            emb_nids = nid_shuffler.shuffle_nids(ntype, emb_nids)
        shuffled_embs[ntype] = (emb, emb_nids)

    save_shuffled_node_embeddings(shuffled_embs, save_embed_path, save_embed_format)


def save_embeddings(emb_path, embeddings, rank, world_size,
    device=th.device('cpu'), node_id_mapping_file=None,
    save_embed_format="pytorch"):
    """ Save embeddings.

        Parameters
        ----------
        emb_path : str
            The path of the folder where the embeddings are saved.
        embeddings : DistTensor or dict of DistTensor
            Embeddings to save
        rank : int
            Rank of the current process in a distributed environment.
        world_size : int
            World size in a distributed env.
        device: torch device
            Device used for all_to_allv data exchange. For gloo backend
            we store data in CPU, For nccl backend, we need to store
            data in GPU.
        node_id_mapping_file : str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
        save_embed_format : str
            The format of saved embeddings.
            Currently support ["pytorch", "hdf5"].
    """
    os.makedirs(emb_path, exist_ok=True)
    if save_embed_format == "pytorch":
        if rank == 0:
            logging.info("Writing GNN embeddings to %s in pytorch format.",
                    emb_path)
        save_pytorch_embeddings(emb_path, embeddings, rank, world_size,
            device, node_id_mapping_file)
    elif save_embed_format == "hdf5":
        if rank == 0:
            logging.info("Writing GNN embeddings to %s in hdf5 format.", \
                os.path.join(emb_path, 'embed_dict.hdf5'))
        save_hdf5_embeddings(emb_path, embeddings, rank, world_size,
            device, node_id_mapping_file)
    else:
        raise ValueError(f"{emb_path} is not supported for save_embed_format")

def shuffle_predict(predictions, id_mapping_file, pred_type,
                    rank, world_size, device):
    """ Shuffle prediction result according to id_mapping

        Parameters
        ----------
        predictions: dgl DistTensor
            prediction results
        id_mapping_file: str
            Path to the file storing node id mapping or edge id mapping generated by the
            graph partition algorithm.
        pred_type: str or tuple
            Node type or edge type of the prediction target.
        rank: int
            Rank of the current process in a distributed environment.
        world_size : int
            World size in a distributed environment. This tells the size of a distributed cluster
            (How many processes in a cluster).
        device : torch device
            Device used to do data shuffling.
    """
    # In most of cases, id_mapping is a dict for heterogeneous graph.
    # For homogeneous graph, it is just a tensor.
    if rank == 0:
        id_mappings = th.load(id_mapping_file)
        ori_id_mapping = id_mappings[pred_type] if isinstance(id_mappings, dict) else id_mappings
        # new mapping back index
        _, id_mapping = th.sort(ori_id_mapping)
    else:
        id_mapping = None

    local_id_mapping = _exchange_node_id_mapping(
                rank, world_size, device, id_mapping,
                len(predictions)).cpu() # predictions are stored in CPU
    return predictions[local_id_mapping]

class NodeIDShuffler():
    """ Shuffle node ids into the Graph Node ID space according to node_id_mappings

        Parameters
        ----------
        g: Dist DGLGraph
            Graph.
        node_id_mapping_file:
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
        ntypes: list of str
            The node types that will have node ids shuffled.
    """
    def __init__(self, g, node_id_mapping_file, ntypes=None):
        self._g = g
        assert node_id_mapping_file is not None \
            and os.path.exists(node_id_mapping_file), \
            f"{node_id_mapping_file} must exist."

        ntypes = ntypes if ntypes is not None else g.ntypes
        assert isinstance(ntypes, list) and len(ntypes) > 0, \
            f"ntypes is not a list or is an empty list {ntypes}"

        if node_id_mapping_file.endswith("pt"):
            # node id mapping file from gconstruct.
            id_mappings = th.load(node_id_mapping_file) if get_rank() == 0 else None
        else:
            # node id mapping file from dgl tools/distpartitioning/convert_partition.py.
            id_mappings = _load_dist_nid_map(node_id_mapping_file, ntypes) \
                if get_rank() == 0 else None

        self._id_mapping_info = {
                ntype: self._load_id_mapping(g, ntype, id_mappings) \
                    for ntype in ntypes
            }

    def _load_id_mapping(self, g, ntype, id_mappings):
        """load id mapping of ntype"""
        num_nodes = g.num_nodes(ntype)
        id_mapping_info = create_dist_tensor((num_nodes,), dtype=th.int64,
                                             name=f"mapping-{ntype}",
                                             part_policy=g.get_node_partition_policy(ntype))
        if get_rank() == 0:
            # the id_mapping stores the mapping from shuffled node ID to original ID
            # For example, the id_mapping [1, 0 ,2] means:
            # Original node ID: 1, 0, 2
            # Shuffled node ID: 0, 1, 2
            id_mapping = id_mappings[ntype] if isinstance(id_mappings, dict) else id_mappings
            assert id_mapping.shape[0] == num_nodes, \
                "Id mapping should have the same size of num_nodes." \
                f"Expect {id_mapping.shape[0]}, but get {num_nodes}"
            # Save ID mapping into dist tensor
            id_mapping_info[th.arange(num_nodes)] = id_mapping
        flush_data()
        return id_mapping_info

    def shuffle_nids(self, ntype, nids):
        """ Shuffle node ids of nype into their Graph Node ID space.

            Parameters
            ----------
            ntype: str
                Node type of nids.
            nids: torch.Tensor
                Node ids.
        """
        assert ntype in self._id_mapping_info, \
            f"The id mapping of {ntype} is not loaded, please provide ntypes" \
            "when initializing NodeIDShuffler"

        return self._id_mapping_info[ntype][nids]

def save_edge_prediction_result(predictions, src_nids, dst_nids,
                               prediction_path, rank):
    """ Save edge predictions to the given path, i.e., prediction_path.

        The function will save three tensors: 1) predictions, which stores
        the prediction results; 2) src_nids, which stores the source
        node ids of target edges and 3) dst_nids, which stores the
        destination node ids of target edges.
        The (src_nid, dst_nid) pairs can be used to identify the target
        edges.

        Parameters
        ----------
        prediction_path: tensor
            The tensor of predictions.
        src_nids: tensor
            The tensor of src node ids.
        dst_nids: tensor
            The tensor of dst node ids.
        prediction_path: str
            The path of the prediction is saved.
        rank: int
            Rank of the current process in a distributed environment.
    """
    os.makedirs(prediction_path, exist_ok=True)
    # [04/16]: Only rank 0 can chmod to let all other ranks to write files.
    if rank == 0:
        # mode 767 means rwx-rw-rwx:
        #     - owner of the folder can read, write, and execute;
        #     - owner' group can read, write;
        #     - others can read, write, and execute.
        os.chmod(prediction_path, 0o767)
    # make sure the prediction_path permission is changed before other process start to save
    barrier()
    th.save(predictions, os.path.join(prediction_path, f"predict-{pad_file_index(rank)}.pt"))
    th.save(src_nids, os.path.join(prediction_path, f"src_nids-{pad_file_index(rank)}.pt"))
    th.save(dst_nids, os.path.join(prediction_path, f"dst_nids-{pad_file_index(rank)}.pt"))

def save_node_prediction_result(predictions, nids,
                               prediction_path, rank):
    """ Save node predictions to the given path, i.e., prediction_path.

        The function will save two tensors: 1) predictions, which stores
        the prediction results; 2) nides, which stores the node ids of the
        target nodes.

        Parameters
        ----------
        prediction_path: tensor
            The tensor of predictions.
        nids: tensor
            The tensor of target node ids.
        prediction_path: str
            The path of the prediction is saved.
        rank: int
            Rank of the current process in a distributed environment.
    """
    os.makedirs(prediction_path, exist_ok=True)
    # [04/16]: Only rank 0 can chmod to let all other ranks to write files.
    if rank == 0:
        # mode 767 means rwx-rw-rwx:
        #     - owner of the folder can read, write, and execute;
        #     - owner' group can read, write;
        #     - others can read, write, and execute.
        os.chmod(prediction_path, 0o767)
    # make sure the prediction_path permission is changed before other process start to save
    barrier()
    th.save(predictions, os.path.join(prediction_path, f"predict-{pad_file_index(rank)}.pt"))
    th.save(nids, os.path.join(prediction_path, f"predict_nids-{pad_file_index(rank)}.pt"))

def save_prediction_results(predictions, prediction_path, rank):
    """ Save node predictions to the given path

        Parameters
        ----------
        prediction_path: tensor
            The tensor of predictions
        prediction_path: str
            The path of the prediction is saved.
        rank: int
            Rank of the current process in a distributed environment.
    """
    os.makedirs(prediction_path, exist_ok=True)
    # [04/16]: Only rank 0 can chmod to let all other ranks to write files.
    if rank == 0:
        # mode 767 means rwx-rw-rwx:
        #     - owner of the folder can read, write, and execute;
        #     - owner' group can read, write;
        #     - others can read, write, and execute.
        os.chmod(prediction_path, 0o767)
    # make sure the prediction_path permission is changed before other process start to save
    barrier()

    th.save(predictions, os.path.join(prediction_path, f"predict-{pad_file_index(rank)}.pt"))

def save_node_prediction_results(predictions, prediction_path):
    """ Save node predictions to the given path

        The saved node prediction results looks like:

        Example:
        --------
        .. code::

            PATH_TO_RESULTS:
            |- result_info.json
            |- ntype0
                |- predict-00000.pt
                |- predict-00001.pt
                |- ...
                |- predict_nids-00000.pt
                |- predict_nids-00001.pt
                |- ...
            |- ntype1
                |- ...

        The result_info.json contains three information:
           * "format", how data are stored, e.g., "pytorch".
           * "world_size", the total number of file parts. 0 means there is no partition.
           * "ntypes", a list of node types that have prediction results.

        Example:
        --------
        .. code::

            {
                "format": "pytorch",
                "world_size": 8,
                "ntypes": ["movie", "user"]
            }

        .. note::

            The saved prediction results are in Graph Node ID space.
            You need to remap them into Raw Node ID space.

        Parameters
        ----------
        predictions: dict of tuple of tensors
            The dict of tuple of tensors of predict results and the corresponding nids
        prediction_path: str
            The path of the prediction is saved.
    """
    rank = get_rank()
    world_size = get_world_size()
    for ntype, (pred, nids) in predictions.items():
        save_node_prediction_result(pred, nids,
                                    os.path.join(prediction_path, ntype),
                                    rank)
    if rank == 0:
        meta_fname = os.path.join(prediction_path, "result_info.json")
        meta_info = {
            "format": "pytorch",
            "world_size": world_size,
            "ntypes": list(predictions.keys())
        }
        with open(meta_fname, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, indent=4)

def save_edge_prediction_results(predictions, prediction_path):
    """ Save edge predictions to the given path

        Example:
        --------
        The saved node prediction results looks like:

        .. code::

            PATH_TO_RESULTS:
            |- result_info.json
            |- etype0
                |- predict-00000.pt
                |- predict-00001.pt
                |- ...
                |- src_nids-00000.pt
                |- src_nids-00001.pt
                |- ...
                |- dst_nids-00000.pt
                |- dst_nids-00001.pt
                |- ...
            |- etype1
                |- ...

        The result_info.json contains three information:
           * "format", how data are stored, e.g., "pytorch".
           * "world_size", the total number of file parts. 0 means there is no partition.
           * "etypes", a list of edge types that have prediction results.

        Example:
        --------
        .. code::

            {
                "format": "pytorch",
                "world_size": 8,
                "etypes": [("movie","rated-by","user"), ("user","watched","movie")]
            }

        .. note::

            The saved prediction results are in Graph Node ID space.
            You need to remap them into Raw Node ID space.

        Parameters
        ----------
        prediction: dict of tensor
            The dict of tensors of predictions.
        prediction_path: str
            The path of the prediction is saved.
    """
    rank = get_rank()
    world_size = get_world_size()
    for etype, pred in predictions.items():
        pred_val, src_nid, dst_nid = pred
        save_edge_prediction_result(pred_val, src_nid, dst_nid,
                                     os.path.join(prediction_path, "_".join(etype)), rank)

    if rank == 0:
        meta_fname = os.path.join(prediction_path, "result_info.json")
        meta_info = {
            "format": "pytorch",
            "world_size": world_size,
            "etypes": list(predictions.keys())
        }
        with open(meta_fname, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, indent=4)

def load_model(model_path, gnn_model=None, embed_layer=None, decoder=None):
    """ Load a complete gnn model.
        A user needs to provide the correct model architectures first.

        Parameters
        ----------
        model_path : str
            The path of the folder where the model is saved.
        gnn_model: model
            GNN model to load
        embed_layer: model
            Embed layer model to load
        decoder: model
            Decoder to load
    """
    gnn_model = gnn_model.module \
        if isinstance(gnn_model, DistributedDataParallel) else gnn_model
    embed_layer = embed_layer.module \
        if isinstance(embed_layer, DistributedDataParallel) else embed_layer
    decoder = decoder.module \
        if isinstance(decoder, DistributedDataParallel) else decoder

    if th.__version__ < "1.13.0":
        logging.warning("torch.load() uses pickle module implicitly, " \
                "which is known to be insecure. It is possible to construct " \
                "malicious pickle data which will execute arbitrary code " \
                "during unpickling. Only load data you trust or " \
                "update torch to 1.13.0+")
        checkpoint = th.load(os.path.join(model_path, 'model.bin'), map_location='cpu')
    else:
        checkpoint = th.load(os.path.join(model_path, 'model.bin'),
                             map_location='cpu',
                             weights_only=True)
    if 'gnn' in checkpoint and gnn_model is not None:
        gnn_model.load_state_dict(checkpoint['gnn'])
    if 'embed' in checkpoint and embed_layer is not None:
        embed_layer.load_state_dict(checkpoint['embed'], strict=False)
    if 'decoder' in checkpoint and decoder is not None:
        decoder.load_state_dict(checkpoint['decoder'])

def load_sparse_emb(target_sparse_emb, ntype_emb_path):
    """load sparse embeddings from ntype_emb_path

        Sparse embeddings are stored as:
            $model_path/ntype0/sparse_emb_0.pt
                               ...
                               sparse_emb_N.pt
            $model_path/ntype1/sparse_emb_0.pt
                               ...
                               sparse_emb_N.pt
            ...

        Example:
        --------
        Load sparse embeddings of an input embed_layer by calling ``load_sparse_emb``
        iterating all the sparse embeddings of the embed_layer

        .. code::

            # embed_layer is the input embed_layer
            # model_path is where the sparse embeddings are stored.
            for ntype, sparse_emb in embed_layer.sparse_embeds.items():
                load_sparse_emb(sparse_emb, os.path.join(model_path, ntype))

        Parameters
        ----------
        target_sparse_emb: dgl.distributed.DistEmbedding
            A Distributed node embedding object where the loaded spare embeddings are stored.
        ntype_emb_path: str
            The path where the node embedding are stored (To be loaded).
    """
    rank = get_rank()
    world_size = get_world_size()
    num_files = len(os.listdir(ntype_emb_path))
    num_embs = target_sparse_emb.num_embeddings

    if isinstance(target_sparse_emb, WholeGraphDistTensor):
        # Using WholeGraph will load sparse emb in binary format, let's assume
        # the sparse emb is saved by WholeGraphDistTensor.save_to_file(), i.e.,
        # the meta info remains the same.
        # Example: wg_sparse_emb_part_0_of_2, wg_sparse_emb_part_1_of_2
        target_sparse_emb.load_from_file(ntype_emb_path, "wg_sparse_emb", num_files)
    else:
        # Suppose a sparse embedding is trained and saved using N trainers (e.g., GPUs).
        # We are going to use K trainers/infers to load it.
        # The code handles the following cases:
        # 1. N == K
        # 2. N > K, some trainers/infers need to load more than one files
        # 3. N < K, some trainers/infers do not need to load any files
        for i in range(math.ceil(num_files/world_size)):
            file_idx = i * world_size + rank
            if file_idx < num_files:
                emb = th.load(os.path.join(ntype_emb_path,
                                           f'sparse_emb_{pad_file_index(file_idx)}.pt'))

                # Get the target idx range for sparse_emb_{rank}.pt
                start, end = _get_sparse_emb_range(num_embs,
                                                   rank=file_idx,
                                                   world_size=num_files)
                # write sparse_emb back in an iterative way
                batch_size = 10240
                idxs = th.split(th.arange(end - start), batch_size, dim=0)
                for idx in idxs:
                    # TODO: dgl.distributed.DistEmbedding should allow some basic tensor ops
                    target_sparse_emb._tensor[start+idx] = emb[idx]

def load_sparse_embeds(model_path, embed_layer):
    """load sparse embeddings if any

        Sparse embeddings are stored as:
        $model_path/ntype0/sparse_emb_0.pt
                           ...
                           sparse_emb_N.pt
        $model_path/ntype1/sparse_emb_0.pt
                           ...
                           sparse_emb_N.pt
        ...

        Parameters
        ----------
        model_path: str
            The path of the model to be saved.
        embed_layer: model
            A (distributed) model of embedding layers.
    """
    if embed_layer is None:
        return
    embed_layer = embed_layer.module \
        if isinstance(embed_layer, DistributedDataParallel) else embed_layer

    if len(embed_layer.sparse_embeds) > 0:
        for ntype, sparse_emb in embed_layer.sparse_embeds.items():
            if th.__version__ < "1.13.0":
                logging.warning("torch.load() uses pickle module implicitly, " \
                        "which is known to be insecure. It is possible to construct " \
                        "malicious pickle data which will execute arbitrary code " \
                        "during unpickling. Only load data you trust or " \
                        "update torch to 1.13.0+")
            emb_path = os.path.join(model_path, ntype)
            assert os.path.exists(emb_path), f"The sparse embedding file {emb_path} doesn't exist."
            load_sparse_emb(sparse_emb, emb_path)

def load_wg_sparse_embeds(model_path, embed_layer):
    """load wholegraph sparse embeddings if any

        Sparse embeddings are stored as:
        $model_path/ntype0/wg_sparse_emb_part_0_of_N
                           ...
                           wg_sparse_emb_part_N-1_of_N
        $model_path/ntype1/wg_sparse_emb_part_0_of_N
                           ...
                           wg_sparse_emb_part_N-1_of_N
        ...

        Parameters
        ----------
        model_path: str
            The path of the model to be saved.
        embed_layer: model
            A (distributed) model of embedding layers.
    """
    if embed_layer is None:
        return
    embed_layer = embed_layer.module \
        if isinstance(embed_layer, DistributedDataParallel) else embed_layer

    if len(embed_layer.sparse_embeds) > 0:
        emb_optimizer = create_wholememory_optimizer("adam", {})
        for ntype, sparse_emb in embed_layer.sparse_embeds.items():
            emb_path = os.path.join(model_path, ntype)
            assert os.path.exists(emb_path), f"The sparse embedding file {emb_path} doesn't exist."
            load_wg_sparse_emb(sparse_emb, emb_path, emb_optimizer)

def load_wg_sparse_emb(target_sparse_emb, ntype_emb_path, optimizer):
    """load wg sparse embeddings from ntype_emb_path

        Sparse embeddings are stored in binary format:
        $model_path/ntype0/wg_sparse_emb_part_0_of_N
                           ...
                           wg_sparse_emb_part_N-1_of_N
        $model_path/ntype1/wg_sparse_emb_part_0_of_N
                           ...
                           wg_sparse_emb_part_N-1_of_N
        ...
        Let's assume the sparse emb is saved by WholeGraphDistTensor.save_to_file(), i.e.,
        the meta info remains the same.

        Example:
        --------
        Load sparse embeddings of an input embed_layer by calling ``load_sparse_emb``
        iterating all the sparse embeddings of the embed_layer

        Parameters
        ----------
        target_sparse_emb: dgl.distributed.DistEmbedding
            A Distributed node embedding object where the loaded spare embeddings are stored.
        ntype_emb_path: str
            The path where the node embedding are stored (To be loaded).
        optimizer: WholeMemoryOptimizer
            The WholeGraph sparse optimizer to be attached to the WholeGraph embedding.
    """
    num_files = len(os.listdir(ntype_emb_path))
    assert isinstance(target_sparse_emb, WholeGraphDistTensor), \
        "The target_sparse_emb must be a WholeGraphDistTensor when loading WholeGraph sparse emb."
    target_sparse_emb.load_from_file(ntype_emb_path, "wg_sparse_emb", num_files, optimizer)


def load_opt_state(model_path, dense_opts, lm_opts, sparse_opts):
    """ Load the optimizer states and resotre the optimizers.

        Parameters
        ----------
        model_path: str
            The path of the model is saved.
        dense_opts: list of optimizers
            Optimzer for dense layers
        lm_opts: list of optimizers
            Optimzer for language models layers
        sparse_opts: list of optimizers
            Optimizer for sparse emb layer
    """
    if th.__version__ < "1.13.0":
        logging.warning("torch.load() uses pickle module implicitly, " \
                "which is known to be insecure. It is possible to construct " \
                "malicious pickle data which will execute arbitrary code " \
                "during unpickling. Only load data you trust")
    checkpoint = th.load(os.path.join(model_path, 'optimizers.bin'), map_location='cpu')

    assert len(dense_opts) <= 1, "We can only support one dense optimizer now."
    assert len(lm_opts) <= 1, "We can only support one language model optimizer now."

    # Load general dense models like gnn and input projection matrix
    if "dense" in checkpoint:
        assert len(dense_opts) == 1, "General dense parameters must exists in the model"
        dense_opts[0].load_state_dict(checkpoint["dense"])
    # Load language models.
    if "lm" in checkpoint:
        assert len(lm_opts) == 1, "Language model parameters must exists in the model"
        lm_opts[0].load_state_dict(checkpoint["lm"])

    # TODO(zhengda) we need to change DGL to make it work.
    if 'sparse' in checkpoint and len(sparse_opts) > 0:
        raise NotImplementedError('We cannot load the state of sparse optimizer')


def remove_saved_models(model_path):
    """ For only save the Top k best performaned models to save disk spaces, need this function to
        removed previously saved model files.

        Parameters
        ----------
        model_path: str
            The path of the model to be removed.

        Returns
        ----------
        status: int
            The remove status.
            0 : successful;
            -1: error occurs for reasons that will be printed.
    """
    assert os.path.exists(model_path), f'The {model_path} does not exists!'

    try:
        shutil.rmtree(model_path)
    except OSError:
        logging.error('Something wrong with deleting contents of %s!', model_path)
        return -1

    return 0

class LazyDistTensor:
    '''Lazy distributed tensor

        When slicing a distributed tensor, we can postpone the operation.
        A user will get the actual data when he slices data from the tensor.
        This behavior is similar to DGL's DistTensor.

        Parameters
        ----------
        dist_tensor : DistTensor
            The distributed tensor
        slice_idx : tensor
            The index to slice the tensor
    '''
    def __init__(self, dist_tensor, slice_idx):
        self.dist_tensor = dist_tensor
        self.slice_idx = slice_idx

    def __len__(self):
        return len(self.slice_idx)

    def __getitem__(self, idx):
        return self.dist_tensor[self.slice_idx[idx]]

    def __setitem__(self, idx, val):
        self.dist_tensor[self.slice_idx[idx]] = val

    @property
    def shape(self):
        """ Shape of lazy tensor
        """
        s = list(self.dist_tensor.shape)
        s[0] = len(self.slice_idx)
        return tuple(s)

    @property
    def dtype(self):
        """ Dtype of lazy tensor
        """
        return self.dist_tensor.dtype

    @property
    def part_policy(self):
        """ Part policy of the underlying dist tensor
        """
        return self.dist_tensor.part_policy

    @property
    def name(self):
        """ Return the name of the underlying dist tensor
        """
        return self.dist_tensor.name

def all_gather(tensor):
    """ Run all_gather on arbitrary tensor data
        Note that this can be further implemented to support arbitrary pickable data
        like list by serialize the data into byte tensor.

        Parameters
        ----------
            data: th.Tensor
                data to collect

        Returns:
        --------
        list of data gathered from each rank: list[th.Tensor]
    """
    # data = data.cpu()
    tensor = tensor.cpu()
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]

    # obtain Tensor size of each rank
    # this is needed to get the maximum size for padding
    # and also to remove the padding when aggregating the results
    local_size = th.LongTensor([tensor.shape[1]])
    size_list = [th.LongTensor([0]) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes, which cause the deadlock
    tensor_list = []
    placeholder_shape = list(tensor.shape)
    placeholder_shape[-1] = max_size
    for _ in size_list:
        tensor_list.append(th.Tensor(size=placeholder_shape).type(tensor.dtype))
    padding_shape = list(tensor.shape)
    padding_shape[-1] = max_size - local_size
    if local_size != max_size:
        padding = th.Tensor(size=padding_shape).type(tensor.dtype)
        tensor = th.cat((tensor, padding), dim=-1)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensr in zip(size_list, tensor_list):
        # remove the padding here by local size of each trainer
        tensr = tensr[..., :size].cpu()
        data_list.append(tensr)

    return data_list

class TopKList():
    """ Purely based on the GSF validation score rank case, which give a score's rank from a list.

    Parameter:
    ----------
        top_k: size of the list, should >= 0. If is 0, then does not keep any record, which is
               for inference only.

    """
    def __init__(self, top_k):
        assert top_k >= 0, f'The top_k argument should be larger or equal to 0, but got {top_k}.'

        self.top_k = top_k
        self.toplist = []

    def insert(self, rank, val):
        """
        Arguments:
        ---------
            rank: int, the rank of the val in the top list, should > 0
            val : the value to be stored in the top list. It could be an object
                  that has comparator method

        Returns:
        ---------
            insert_success: Boolean, if the given rank has been inserted.
                            True, if the topk list is not full or the rank is in the top k
                            False, if the topk list is full and the rank is not in the top k
            return_val: A value either is the given val, or the last top k value in the topk list.
                        If the insert_success is True, the return_val should be the given val,
                        which should be saved, or the last val in the previous topk list, which
                        should be removed;
                        If the insert_success is False, the return_val could be the given val.

        """
        if (rank - 1) >= self.top_k:                # only when list length > k will rank be > k
            insert_success = False
            return_val = val
        else:
            if len(self.toplist) == self.top_k:  # list is full
                insert_success = True
                return_val = self.toplist[-1]

                first_part = self.toplist[:(rank - 1)]
                last_part = self.toplist[(rank - 1): -1]
                self.toplist = first_part + [val] + last_part
            else:                                   # list is not full and rank <= list lenght
                insert_success = True
                return_val = val

                first_part = self.toplist[: (rank - 1)]
                last_part = self.toplist[(rank - 1):]
                self.toplist = first_part + [val] + last_part

        return insert_success, return_val

def create_sparse_emb_path(model_path, ntype):
    """ Create sparse embedding save path by the rank 0

         The folders are like:
            $model_path/ntype0/
            $model_path/ntype1/
            ...

        Example:
        --------

        Creat paths on a shared file system for saving sparse embeddings of
        an input embed_layer by calling ``create_sparse_emb_path``
        for each sparse embedding of the embed_layer

        .. code::

            # embed_layer is the input embed_layer
            embed_layer = embed_layer.module \
                if isinstance(embed_layer, DistributedDataParallel) else embed_layer

            if len(embed_layer.sparse_embeds) > 0:
                for ntype, _ in embed_layer.sparse_embeds.items():
                    create_sparse_emb_path(model_path, ntype)
            return

        Parameters
        ----------
        model_path: str
            The path of the model is saved.
        ntype: str
            The node type the sparse embedding belongs to.
    """
    # Assume this method is called by rank 0 who can perform chmod
    if get_rank() == 0:
        emb_path = os.path.join(model_path, ntype)
        os.makedirs(emb_path, exist_ok=True)

        # mode 767 means rwx-rw-rwx:
        #     - owner of the folder can read, write, and execute;
        #     - owner' group can read, write;
        #     - others can read, write, and execute.
        os.chmod(emb_path, 0o767)
    barrier()

def create_sparse_embeds_path(model_path, embed_layer):
    """ create sparse embeddings save path by the rank 0
        The folders are like:

        $model_path/ntype0/
        $model_path/ntype1/
        ...

        Parameters
        ----------
        model_path: str
            The path of the model is saved.
        embed_layer: model
            A (distributed) model of embedding layers.
    """
    if embed_layer is None:
        return

    embed_layer = embed_layer.module \
        if isinstance(embed_layer, DistributedDataParallel) else embed_layer

    if len(embed_layer.sparse_embeds) > 0:
        for ntype, _ in embed_layer.sparse_embeds.items():
            create_sparse_emb_path(model_path, ntype)

def append_to_dict(from_dict, to_dict):
    """ Append content of from_dict to to_dict

        Parameters
        ----------
        from_dict: dict of Tensor
            Dict of tensor to be added to to_dict
        to_dict: dict of Tensor
            Target dict of tensor
    """
    for k, v in from_dict.items():
        if k in to_dict:
            to_dict[k].append(v.cpu())
        else:
            to_dict[k] = [v.cpu()]

def normalize_node_embs(embs, norm_method):
    """ Do node embedding normalization

        Parameters
        ----------
        embs: dict of Tensor
            node embeddings.
        norm_method: str
            Node embedding normalization method.

        Return
        ------
        dict of Tensor: Dict of normalized embeddings.
    """
    if norm_method is None or norm_method == "":
        def norm(emb):
            return emb
        norm_func = norm
    elif norm_method == GRAPHSTORM_LP_EMB_L2_NORMALIZATION:
        def do_l2_norm(emb):
            return F.normalize(emb)
        norm_func = do_l2_norm
    else:
        raise RuntimeError(f"Unsupported embedding normalization method {norm_method}")

    embs = {key: norm_func(emb) for key, emb in embs.items()}
    return embs
