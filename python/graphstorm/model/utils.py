""" Utils """
import os
import json
import time
import shutil

import torch as th
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import dgl

from .extract_node_embeddings import extract_all_embeddings_dist, prepare_batch_input

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

def save_embeds(embed_path, node_embed, relation_embs):
    ''' Save the generated node embeddings and relation embedding

        Parameters
        ----------
        embed_path: str
            Path to save the embeddings
        node_embed: th.Tensor
            Node embedding
        relation_embs:
            relation embedding
    '''
    emb_states = {
        'node_embed' : node_embed
    }
    if relation_embs is not None:
        emb_states['relation_embed'] = relation_embs
    th.save(emb_states, embed_path)

def load_embeds(embed_path):
    """ Load embedding from disk

        Parameters
        ----------
        embed_path: str
            Path to load the embeddings

        Returns
        -------
        node embedding: th.Tensor
        relation embedding: th.Tensor
    """
    emb_states = th.load(embed_path)
    node_embed = emb_states['node_embed']
    relation_embs = emb_states['relation_embed'] \
        if 'relation_embed' in emb_states else None
    return node_embed, relation_embs

def save_model(conf, model_path, gnn_model=None, embed_layer=None, decoder=None):
    """ A model should have three parts:
        * GNN model
        * embedding layer
        The model is only used for inference.

        Parameters
        ----------
        conf: dict
            The configuration of the model architecture.
        model_path: str
            The path of the model is saved.
        gnn_model: model
            A (distributed) model of GNN
        embed_layer: model
            A (distributed) model of embedding layers.
        decoder: model
            A (distributed) model of decoder
    """

    if gnn_model is not None:
        gnn_model = gnn_model.module \
            if isinstance(gnn_model, DistributedDataParallel) else gnn_model

    if embed_layer is not None:
        embed_layer = embed_layer.module \
            if isinstance(embed_layer, DistributedDataParallel) else embed_layer

    if decoder is not None:
        decoder = decoder.module \
            if isinstance(decoder, DistributedDataParallel) else decoder

    model_states = {}
    if gnn_model is not None:
        model_states['gnn'] = gnn_model.state_dict()
    if embed_layer is not None:
        model_states['embed'] = embed_layer.state_dict()
    if decoder is not None:
        model_states['decoder'] = decoder.state_dict()

    os.makedirs(model_path, exist_ok=True)
    th.save(model_states, os.path.join(model_path, 'model.bin'))

    with open(os.path.join(model_path, 'model_conf.json'), 'w', encoding='utf-8') as f:
        json.dump(conf, f, ensure_ascii=False, indent=4)

def save_sparse_embeds(model_path, embed_layer):
    """ save sparse embeddings if any

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
            # collect sparse_emb in a iterative way
            embs = []
            batch_size = 10240
            # TODO: dgl.distributed.DistEmbedding should provide emb.shape
            num_embs = embed_layer.g.number_of_nodes(ntype)
            idxs = th.split(th.arange(num_embs), batch_size, dim=0)
            for idx in idxs:
                # TODO: dgl.distributed.DistEmbedding should allow some basic tensor ops
                # TODO(xiang): Fix the scalablity problem here
                embs.append(sparse_emb._tensor[idx])

            embs = th.cat(embs, dim=0)

            th.save(embs, os.path.join(model_path, f'{ntype}_sparse_emb.pt'))

def save_opt_state(model_path, dense_opt, emb_opt):
    """ Save the states of the optimizers.

        There are usually three optimizers:
        * for the dense model parameters.
        * for the sparse embedding layers.

        Parameters
        ----------
        model_path : str
            The path of the folder where the model is saved.
            We save the optimizer states with the model.
        dense_opt : optimizer
            The optimizer for dense model parameters.
        emb_opt : optimizer
            The optimizer for sparse embedding layer.
    """
    opt_states = {}
    if dense_opt is not None:
        opt_states['dense'] = dense_opt.state_dict()
    # TODO(zhengda) we need to change DGL to make it work.
    if emb_opt is not None:
        # TODO(xiangsx) Further discussion of whether we need to save the state of
        #               sparse optimizer is needed.
        print("WARNING: We do not export the state of sparse optimizer")
    #    opt_states['emb'] = emb_opt.state_dict()
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
        json.dump(et2id_map, f, ensure_ascii=False, indent=4)
    th.save(relembs, os.path.join(emb_path, "rel_emb.pt"))

def save_embeddings(model_path, embeddings, local_rank, world_size):
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
    """
    os.makedirs(model_path, exist_ok=True)
    assert local_rank < world_size
    def get_data_range(num_embs):
        # Get corresponding data range
        start = local_rank * (num_embs // world_size)
        end = (local_rank + 1) * (num_embs // world_size)
        end = num_embs if local_rank + 1 == world_size else end
        return start, end

    if isinstance(embeddings, (dgl.distributed.DistTensor, LazyDistTensor)):
        start, end = get_data_range(len(embeddings))
        embeddings = embeddings[start:end]
    elif isinstance(embeddings, dict):
        for name, emb in embeddings.items():
            if isinstance(emb, (dgl.distributed.DistTensor, LazyDistTensor)):
                start, end = get_data_range(len(emb))
                emb = emb[start:end]
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

    checkpoint = th.load(os.path.join(model_path, 'model.bin'), map_location='cpu')
    if 'gnn' in checkpoint and gnn_model is not None:
        print("Loading gnn model")
        gnn_model.load_state_dict(checkpoint['gnn'])
    if 'embed' in checkpoint and embed_layer is not None:
        print("Loading embedding model")
        embed_layer.load_state_dict(checkpoint['embed'])
    if 'decoder' in checkpoint and decoder is not None:
        print("Loading decoder model")
        decoder.load_state_dict(checkpoint['decoder'])
    if 'decoder' in checkpoint and decoder is not None:
        decoder.load_state_dict(checkpoint['decoder'])

def load_sparse_embeds(model_path, embed_layer):
    """load sparse embeddings if any

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
            emb = th.load(os.path.join(model_path, f'{ntype}_sparse_emb.pt'))
            # write sparse_emb back in a iterative way
            batch_size = 10240
            # TODO: dgl.distributed.DistEmbedding should provide emb.shape
            num_embs = embed_layer.g.number_of_nodes(ntype)
            idxs = th.split(th.arange(num_embs), batch_size, dim=0)
            for idx in idxs:
                # TODO: dgl.distributed.DistEmbedding should allow some basic tensor ops
                sparse_emb._tensor[idx] = emb[idx]

def load_opt_state(model_path, dense_opt, emb_opt):
    """ Load the optimizer states and resotre the optimizers.

        Parameters
        ----------
        model_path: str
            The path of the model is saved.
        dense_opt: optimizer
            Optimzer for dense layers
        emb_opt: optimizer
            Optimizer for emb layer
    """
    checkpoint = th.load(os.path.join(model_path, 'optimizers.bin'))
    dense_opt.load_state_dict(checkpoint['dense'])
    # TODO(zhengda) we need to change DGL to make it work.
    if 'emb' in checkpoint and emb_opt is not None:
        raise NotImplementedError('We cannot load the state of sparse optimizer')
    #    emb_opt.load_state_dict(checkpoint['emb'])

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

# pylint: disable=invalid-name
def do_mini_batch_inference(model, embed_layer, device,
                            target_nidx, g, pb, n_hidden,
                            fanout, eval_batch_size,
                            task_tracker=None, feat_field='feat'):
    """ Do mini batch inference

        Parameters
        ----------
        model: torch model
            GNN model
        embed_layer: torch model
            GNN input embedding layer
        device: th.device
            Device
        target_nidx: th.Tensor
            Target node idices
        g: DistDGLGraph
            DGL graph
        pb: DGL partition book
            The partition book
        n_hidden: int
            GNN hidden size
        fanout: int
            Inference fanout
        eval_batch_size: int
            The batch size
        task_tracker: GSTaskTrackerAbc
            Task tracker
        feat_field: str
            field to extract features

        Returns
        -------
        Node embeddings: dict of str to th.Tensor
    """
    t0 = time.time()
    # train sampler
    target_idxs_dist = {}
    embeddings = {}
    # this mapping will hold the mapping among the rows of
    # the embedding matrix to the original target ids
    for key in target_nidx:
        # Note: The space overhead here, i.e., using a global target_mask,
        # is O(N), N is number of nodes.
        # Use int8 as all_reduce does not work well with bool
        # TODO(zhengda) we need to reduce the memory complexity described above.
        target_mask = th.full((g.num_nodes(key),), 0, dtype=th.int8)
        target_mask[target_nidx[key]] = 1

        # As each trainer may only focus on its own val/test set,
        # i.e., the val/test sets only contain local nodes or edges.
        # we need to get the full node or edge list before node_split
        # Here we use all_reduce to sync the target node/edge mask.
        # TODO(xiangsx): make it work with NCCL
        # TODO(xiangsx): make it more efficient
        th.distributed.all_reduce(target_mask,
            op=th.distributed.ReduceOp.MAX)
        print("do allreduce")

        node_trainer_ids=g.nodes[key].data['trainer_id'] \
            if 'trainer_id' in g.nodes[key].data else None
        target_idx_dist = dgl.distributed.node_split(
                target_mask.bool(),
                pb, ntype=key, force_even=False,
                node_trainer_ids=node_trainer_ids)
        target_idxs_dist[key] = target_idx_dist
        embeddings[key] = dgl.distributed.DistTensor(
            (g.number_of_nodes(key), n_hidden),
            dtype=th.float32, name='output_embeddings',
            part_policy=g.get_node_partition_policy(key),
            persistent=True)

    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
    loader = dgl.dataloading.DistNodeDataLoader(g, target_idxs_dist, sampler,
                                                batch_size=eval_batch_size,
                                                shuffle=False, num_workers=0)
    g.barrier()
    if model is not None:
        model.eval()
    if embed_layer is not None:
        embed_layer.eval()
    th.cuda.empty_cache()
    with th.no_grad():
        for iter_l, (input_nodes, seeds, blocks) in enumerate(loader):
            if task_tracker is not None:
                task_tracker.keep_alive(iter_l)

            # in the case of a graph with a single node type the returned seeds will not be
            # a dictionary but a tensor of integers this is a possible bug in the DGL code.
            # Otherwise we will select the seeds that correspond to the category node type
            if isinstance(input_nodes, dict) is False:
                # This happens on a homogeneous graph.
                assert len(g.ntypes) == 1
                input_nodes = {g.ntypes[0]: input_nodes}

            if isinstance(seeds, dict) is False:
                # This happens on a homogeneous graph.
                assert len(g.ntypes) == 1
                seeds = {g.ntypes[0]: seeds}

            blocks = [blk.to(device) for blk in blocks]
            inputs = prepare_batch_input(g,
                                         input_nodes,
                                         dev=device,
                                         feat_field=feat_field)

            input_nodes = {ntype: inodes.long().to(device) \
                    for ntype, inodes in input_nodes.items()}
            emb = embed_layer(inputs, input_nodes=input_nodes) \
                    if embed_layer is not None else inputs
            final_embs = model(emb, blocks) \
                    if model is not None else {ntype: nemb.to(device) \
                        for ntype, nemb in emb.items()}
            for key in seeds:
                # we need to go over the keys in the seed dictionary and not the final_embs.
                # The reason is that our model
                # will always return a zero tensor if there are no nodes of a certain type.
                if len(seeds[key]) > 0:
                    # it might be the case that one key has a zero tensor this will cause a failure.
                    embeddings[key][seeds[key]] = final_embs[key].cpu()

    if model is not None:
        model.train()
    if embed_layer is not None:
        embed_layer.train()
    g.barrier()
    t1 = time.time()

    if g.rank() == 0:
        print(f'Computing GNN embeddings: {(t1 - t0):.4f} seconds')

    if target_nidx is not None:
        embeddings = {ntype: LazyDistTensor(embeddings[ntype], target_nidx[ntype]) \
            for ntype in target_nidx.keys()}
    return embeddings

def do_fullgraph_infer(g, model, embed_layer, device, eval_fanout_list,
                       eval_batch_size=None,task_tracker=None,
                       feat_field='feat'):
    """ Do fullgraph inference

        Parameters
        ----------
        g: DistDGLGraph
            DGLGraph
        model: torch model
            GNN model
        embed_layer: torch model
            GNN input embedding layer
        device: th.device
            Device
        eval_fanout_list: list
            The evaluation fanout list
        eval_batch_size: int
            The batch size
        task_tracker: GSTaskTrackerAbc
            Task tracker
        feat_field: str
            Field to extract features

        Returns
        -------
        Node embeddings: dict of str to th.Tensor
    """
    node_embed = extract_all_embeddings_dist(g,
                                             # TODO(zhengda) the batch size should be configurable.
                                             1024,
                                             embed_layer,
                                             dev=device,
                                             task_tracker=task_tracker,
                                             feat_field=feat_field)
    t1 = time.time() # pylint: disable=invalid-name
    # full graph evaluation
    g.barrier()
    model.eval()
    if not isinstance(model, DistributedDataParallel):
        embeddings = model.dist_inference(g, eval_batch_size, device, 0,
            node_embed, eval_fanout_list, task_tracker=task_tracker)
    else:
        embeddings = model.module.dist_inference(g, eval_batch_size,
            device, 0, node_embed, eval_fanout_list, task_tracker=task_tracker)
    if g.rank() == 0:
        print(f"computing GNN embeddings: {time.time() - t1:.4f} seconds")
    model.train()
    return embeddings

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
