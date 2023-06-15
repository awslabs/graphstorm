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

import torch as th
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import dgl

from ..utils import get_rank
from ..data.utils import alltoallv_nccl, alltoallv_cpu

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

def _get_sparse_emb_range(num_embs, local_rank, world_size):
    """ Provide a deterministic method to split trainable sparse embeddings
        during saveing and loading according local rank and world size.

        Parameters
        ----------
        num_embs:
            Size of a sparse embedding
        local_rank : int
            Local rank
        world_size : int
            World size in a distributed env.
    """
    assert local_rank < world_size, \
        "local rank {local_rank} shold be smaller than world size {world_size}"
    # Get corresponding data range
    if num_embs < world_size:
        start = local_rank if local_rank < num_embs else num_embs
        end = local_rank + 1 if local_rank < num_embs else num_embs
    else:
        start = local_rank * math.ceil(num_embs / world_size)
        end = (local_rank + 1) * math.ceil(num_embs / world_size)
        end = num_embs if local_rank + 1 == world_size else end
    return start, end

def save_sparse_embeds(model_path, embed_layer, local_rank, world_size):
    """ save sparse embeddings if any

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
        local_rank : int
            Local rank
        world_size : int
            World size in a distributed env.
    """
    if embed_layer is None:
        return
    embed_layer = embed_layer.module \
        if isinstance(embed_layer, DistributedDataParallel) else embed_layer

    if len(embed_layer.sparse_embeds) > 0:
        assert local_rank < world_size
        for ntype, sparse_emb in embed_layer.sparse_embeds.items():
            num_embs = embed_layer.g.number_of_nodes(ntype)
            start, end = _get_sparse_emb_range(num_embs, local_rank, world_size)
            # collect sparse_emb in a iterative way
            embs = []
            batch_size = 10240
            # TODO: dgl.distributed.DistEmbedding should provide emb.shape

            idxs = th.split(th.arange(start=start, end=end), batch_size, dim=0)
            for idx in idxs:
                # TODO: dgl.distributed.DistEmbedding should allow some basic tensor ops
                embs.append(sparse_emb._tensor[idx])

            embs = th.cat(embs, dim=0)
            # In distributed mode where uses an NFS folder, directly call this makedirs method to
            # create spare embedding path will cause folder permission error that prevents
            # non-rank 0 process from saving embeddings. Therefore, need rank 0 process to call
            # the create_sparse_embeds_path() method first before calling save_sparse_embeds().
            emb_path = os.path.join(model_path, ntype)
            os.makedirs(emb_path, exist_ok=True)
            emb_file_path = os.path.join(emb_path, f'sparse_emb_{local_rank}.pt')
            th.save(embs, emb_file_path)

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
        print("WARNING: We do not export the state of sparse optimizer")
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

def _get_data_range(rank, world_size, num_embs):
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

def _exchange_node_id_mapping(local_rank, world_size, device,
    node_id_mapping, num_embs):
    """ Rank0 loads node_id_mappings and spreads it to other ranks.
        Each rank will get a sub-range of node_id_mappings.
        We use alltoall_v to send sub-node_id_mappings to each rank.

        Parameters
        ----------
        local_rank: int
            Local rank
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
    backend = th.distributed.get_backend()
    device = th.device('cpu') if backend == "gloo" else device

    if local_rank == 0:
        data_tensors = []
        for i in range(world_size):
            start_idx, end_idx = _get_data_range(i, world_size, num_embs)
            data_tensors.append(
                node_id_mapping[start_idx:end_idx].to(device))
    else:
        data_tensors = [th.empty((0,),
                                    dtype=th.long,
                                    device=device) \
            for _ in range(world_size)]

    start_idx, end_idx = _get_data_range(local_rank, world_size, num_embs)
    gather_list = \
        [th.empty((end_idx-start_idx,),
                    dtype=th.long,
                    device=device) \
            if i == 0 else th.empty((0,),
                                    dtype=th.long,
                                    device=device) \
            for i in range(world_size)]
    if backend == "gloo":
        alltoallv_cpu(local_rank, world_size, gather_list, data_tensors)
    else: # backend == "nccl"
        alltoallv_nccl(local_rank, world_size, gather_list, data_tensors)
    return gather_list[0]

def save_embeddings(model_path, embeddings, local_rank, world_size,
    device=th.device('cpu'), node_id_mapping_file=None):
    """ Save embeddings in a distributed way

        Parameters
        ----------
        model_path : str
            The path of the folder where the model is saved.
        embeddings : DistTensor
            Embeddings to save
        local_rank : int
            Local rank
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
    os.makedirs(model_path, exist_ok=True)
    # [04/16]: Only rank 0 can chmod to let all other ranks to write files.
    if local_rank == 0:
        # mode 767 means rwx-rw-rwx:
        #     - owner of the folder can read, write, and execute;
        #     - owner' group can read, write;
        #     - others can read, write, and execute.
        os.chmod(model_path, 0o767)

    # make sure the model_path permission is changed before other process start to save
    th.distributed.barrier()

    assert local_rank < world_size
    # Node ID mapping won't be very large if number of nodes is
    # less than 10 billion. An ID mapping of 10 billion nodes
    # will take around 80 GByte.
    if node_id_mapping_file is not None:
        if isinstance(embeddings, (dgl.distributed.DistTensor, LazyDistTensor)):
            # only host 0 will load node id mapping from disk
            node_id_mapping = th.load(node_id_mapping_file) \
                if local_rank == 0 else None

            nid_mapping = _exchange_node_id_mapping(
                local_rank, world_size, device, node_id_mapping, len(embeddings))
        elif isinstance(embeddings, dict):
            nid_mapping = {}
            # only host 0 will load node id mapping from disk
            node_id_mappings = th.load(node_id_mapping_file) \
                if local_rank == 0 else None

            for name, emb in embeddings.items():
                if local_rank == 0:
                    assert name in node_id_mappings, \
                        f"node id mapping for ntype {name} should exists"
                    node_id_mapping = node_id_mappings[name]
                else:
                    node_id_mapping = None

                nid_mapping[name] = _exchange_node_id_mapping(
                    local_rank, world_size, device, node_id_mapping, len(emb))
        else:
            nid_mapping = None
    else:
        nid_mapping = None

    if isinstance(embeddings, (dgl.distributed.DistTensor, LazyDistTensor)):
        if nid_mapping is None:
            start, end = _get_data_range(local_rank, world_size, len(embeddings))
            embeddings = embeddings[start:end]
        else:
            embeddings = embeddings[nid_mapping]
    elif isinstance(embeddings, dict):
        # We need to duplicate the dict so that the input argument is not changed.
        embeddings = dict(embeddings.items())
        for name, emb in embeddings.items():
            if isinstance(emb, (dgl.distributed.DistTensor, LazyDistTensor)):
                if nid_mapping is None:
                    start, end = _get_data_range(local_rank, world_size, len(emb))
                    emb = emb[start:end]
                else:
                    emb = emb[nid_mapping[name]]
                embeddings[name] = emb

    emb_info = {
        "emb_name":[],
        "world_size":world_size
    }

    if isinstance(embeddings, dict):
        # embedding per node type
        for name, emb in embeddings.items():
            th.save(emb, os.path.join(model_path, f'{name}_emb.part{local_rank}.bin'))
            emb_info["emb_name"].append(name)
    else:
        th.save(embeddings, os.path.join(model_path, f'emb.part{local_rank}.bin'))
        emb_info["emb_name"] = None

    if local_rank == 0:
        with open(os.path.join(model_path, "emb_info.json"), 'w', encoding='utf-8') as f:
            f.write(json.dumps(emb_info))

def shuffle_predict(predictions, id_mapping_file, pred_type,
                    local_rank, world_size, device):
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
        local_rank : int
            Local rank
        world_size : int
            World size in a distributed env.
        device : torch device
            Device used to do data shuffling.
    """
    id_mapping = th.load(id_mapping_file) if local_rank == 0 else None
    # In most of cases, id_mapping is a dict for heterogeneous graph.
    # For homogeneous graph, it is just a tensor.
    id_mapping = id_mapping[pred_type] if isinstance(id_mapping, dict) else id_mapping
    local_id_mapping = _exchange_node_id_mapping(
                local_rank, world_size, device, id_mapping,
                len(predictions)).cpu() # predictions are stored in CPU
    return predictions[local_id_mapping]

def save_prediction_results(predictions, prediction_path, local_rank):
    """ Save node and edge predictions to the given path

        Parameters
        ----------
        prediction_path: tensor
            The tensor of predictions
        prediction_path: str
            The path of the prediction is saved.
        local_rank : int
            Local rank
    """
    os.makedirs(prediction_path, exist_ok=True)
    # [04/16]: Only rank 0 can chmod to let all other ranks to write files.
    if local_rank == 0:
        # mode 767 means rwx-rw-rwx:
        #     - owner of the folder can read, write, and execute;
        #     - owner' group can read, write;
        #     - others can read, write, and execute.
        os.chmod(prediction_path, 0o767)
    # make sure the prediction_path permission is changed before other process start to save
    th.distributed.barrier()

    th.save(predictions, os.path.join(prediction_path, "predict-{}.pt".format(local_rank)))

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
        print("WARNING: torch.load() uses pickle module implicitly, " \
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
        embed_layer.load_state_dict(checkpoint['embed'])
    if 'decoder' in checkpoint and decoder is not None:
        decoder.load_state_dict(checkpoint['decoder'])

def load_sparse_embeds(model_path, embed_layer, local_rank, world_size):
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
            The path of the model is saved.
        embed_layer: model
            A (distributed) model of embedding layers.
        local_rank : int
            Local rank
        world_size : int
            World size in a distributed env.
    """
    if embed_layer is None:
        return
    embed_layer = embed_layer.module \
        if isinstance(embed_layer, DistributedDataParallel) else embed_layer

    if len(embed_layer.sparse_embeds) > 0:
        assert local_rank >= 0
        assert world_size > 0
        def load_sparse_emb(num_embs, ntype_path):
            num_files = len(os.listdir(ntype_path))
            # Suppose a sparse embedding is trained and saved using N trainers (GPUs).
            # We are going to use K trainers/infers to load it.
            # The code handles the following cases:
            # 1. N == K
            # 2. N > K, some trainers/infers need to load more than one files
            # 3. N < K, some trainers/infers do not need to load any files
            for i in range(math.ceil(num_files/world_size)):
                file_idx = i * world_size + local_rank
                if file_idx < num_files:
                    emb = th.load(os.path.join(ntype_path, f'sparse_emb_{file_idx}.pt'))

                    # Get the target idx range for sparse_emb_{local_rank}.pt
                    start, end = _get_sparse_emb_range(num_embs,
                                                       local_rank=file_idx,
                                                       world_size=num_files)
                    # write sparse_emb back in an iterative way
                    batch_size = 10240
                    idxs = th.split(th.arange(end - start), batch_size, dim=0)
                    for idx in idxs:
                        # TODO: dgl.distributed.DistEmbedding should allow some basic tensor ops
                        sparse_emb._tensor[start+idx] = emb[idx]

        for ntype, sparse_emb in embed_layer.sparse_embeds.items():
            num_embs = embed_layer.g.number_of_nodes(ntype)
            if th.__version__ < "1.13.0":
                print("WARNING: torch.load() uses pickle module implicitly, " \
                    "which is known to be insecure. It is possible to construct " \
                    "malicious pickle data which will execute arbitrary code " \
                    "during unpickling. Only load data you trust or " \
                    "update torch to 1.13.0+")
                load_sparse_emb(num_embs, os.path.join(model_path, ntype))
            else:
                load_sparse_emb(num_embs, os.path.join(model_path, ntype))

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
        print("WARNING: torch.load() uses pickle module implicitly, " \
            "which is known to be insecure. It is possible to construct " \
            "malicious pickle data which will execute arbitrary code " \
            "during unpickling. Only load data you trust")
    checkpoint = th.load(os.path.join(model_path, 'optimizers.bin'))

    assert len(dense_opts) <= 1, "We can only support one dense optimizer now."
    assert len(lm_opts) <= 1, "We can only support one language model optimizer now."

    # Load general dense models like gnn and input projection matrix
    if "dense" in checkpoint:
        assert len(dense_opts) == 1, "General dense parameters must exists in the model"
        dense_opts[0].load_state_dict(checkpoint["dense"])
    # Load language models.
    if "lm" in checkpoint:
        assert len(lm_opts) == 1, "Language model parameters must exists in the model"
        dense_opts[0].load_state_dict(checkpoint["lm"])

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
        print(f'WARNING: Something wrong with deleting contents of {model_path}!')
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
            if len(self.toplist) == self.top_k: # list is full
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
            emb_path = os.path.join(model_path, ntype)
            os.makedirs(emb_path, exist_ok=True)
            # [04/16]: Assume this method is called by rank 0 who can perform chmod
            assert get_rank() == 0, "Only can the rank 0 process can change folders mode."
            # mode 767 means rwx-rw-rwx:
            #     - owner of the folder can read, write, and execute;
            #     - owner' group can read, write;
            #     - others can read, write, and execute.
            os.chmod(emb_path, 0o767)
