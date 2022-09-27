""" Utils """
import os
import json
import time

import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import dgl

from ..data.constants import TOKEN_IDX
from .extract_node_embeddings import extract_all_embeddings_dist, prepare_batch_input
from .hbert import extract_bert_embed
from ..data.utils import alltoallv_nccl, alltoallv_cpu

def modify_fanout_for_target_etype(g, fanout, target_etypes):
    """ This function specifies a zero fanout for the target etype
        removing this etype from the message passing graph

        Parameters
        ----------
        g:
            The graph
        fanout:
            Sampling fanout
        target_etypes:
            Target etype to change the fanout

        Returns
        -------
        Modified fanout: list
    """

    edge_fanout_lis = []
    # The user can decide to not use the target etype for message passing.
    for fan in fanout:
        edge_fanout_dic = {}
        for etype in g.etypes:
            if g.to_canonical_etype(etype) not in target_etypes:
                edge_fanout_dic[etype] = fan if not isinstance(fan, dict) else fan[etype]
            else:
                print(f"Ignoring edges for {etype} etype")
                edge_fanout_dic[etype] = 0
        edge_fanout_lis.append(edge_fanout_dic)
    return edge_fanout_lis


def rand_gen_trainmask(g, input_nodes, num_train_nodes, disable_training):
    """ Generate random train mask

        Parameters
        ----------
        g :
            The graph
        input_nodes :
            All the nodes for which bert embeddings are required
        num_train_nodes :
            The number of nodes for which the loss for the BERT
                    embedding will be backpropagated
        disable_training :
            This flag can be set to disable training of the BERT model.

        Returns
        -------
        Generated training masks: dict of th.Tensor
    """
    train_masks = {}
    for ntype, nid in input_nodes.items():
        if disable_training:
            train_mask = th.full((nid.shape[0],), False, dtype=th.bool)
        else:
            if num_train_nodes >= 0 and TOKEN_IDX in g.nodes[ntype].data:
                if nid.shape[0] <= num_train_nodes: # all nodes is trainable
                    train_mask = th.full((nid.shape[0],), True, dtype=th.bool)
                else: # random select # of train_nodes
                    train_mask = th.full((nid.shape[0],), False, dtype=th.bool)
                    train_idx = th.tensor(np.random.choice(nid.shape[0],
                                                           size=num_train_nodes,
                                                           replace=False))
                    train_mask[train_idx] = True
            else:
                train_mask = None
        train_masks[ntype] = train_mask

    return train_masks

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

def calc_distmult_pos_score(h_emb, t_emb, r_emb, device=None):
    """ Calculate DistMulti Score for positive pairs

        score = sum(head_emb * relation_emb * tail_emb)

        Parameters
        ----------
        h_emb: th.Tensor
            Head node embedding
        t_emb: th.Tensor
            Tail node embedding
        r_emb: th.Tensor
            Relation type embedding

        Return
        ------
        Distmult score: th.Tensor
    """
    # DistMult
    if device is not None:
        r_emb = r_emb.to(device)
        h_emb = h_emb.to(device)
        t_emb = t_emb.to(device)

    score = th.sum(h_emb * r_emb * t_emb, dim=-1)
    return score

def calc_distmult_neg_tail_score(heads, tails, r_emb, num_chunks, chunk_size,
    neg_sample_size, device=None):
    """ Calculate DistMulti Score for negative pairs when tail nodes are negative

        score = sum(head_emb * relation_emb * tail_emb)

        Parameters
        ----------
        heads: th.Tensor
            Head node embedding
        tails: th.Tensor
            Tail node embedding
        r_emb: th.Tensor
            Relation type embedding
        num_chunks: int
            Number of shared negative chunks
        chunk_size: int
            Chunk size
        neg_sample_size: int
            Number of negative samples for each positive node
        device: th.device
            Device to run the computation

        Return
        ------
        Distmult score: th.Tensor
    """
    hidden_dim = heads.shape[1]
    r = r_emb

    if device is not None:
        r = r.to(device)
        heads = heads.to(device)
        tails = tails.to(device)
    tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
    tails = th.transpose(tails, 1, 2)
    tmp = (heads * r).reshape(num_chunks, chunk_size, hidden_dim)
    return th.bmm(tmp, tails)

def calc_distmult_neg_head_score(heads, tails, r_emb, num_chunks, chunk_size,
    neg_sample_size, device=None):
    """ Calculate DistMulti Score for negative pairs when head nodes are negative

        score = sum(head_emb * relation_emb * tail_emb)

        Parameters
        ----------
        heads: th.Tensor
            Head node embedding
        tails: th.Tensor
            Tail node embedding
        r_emb: th.Tensor
            Relation type embedding
        num_chunks: int
            Number of shared negative chunks
        chunk_size: int
            Chunk size
        neg_sample_size: int
            Number of negative samples for each positive node
        device: th.device
            Device to run the computation

        Return
        ------
        Distmult score: th.Tensor
    """
    hidden_dim = tails.shape[1]
    r = r_emb
    if device is not None:
        r = r.to(device)
        heads = heads.to(device)
        tails = tails.to(device)
    heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
    heads = th.transpose(heads, 1, 2)
    tmp = (tails * r).reshape(num_chunks, chunk_size, hidden_dim)
    return th.bmm(tmp, heads)

def calc_dot_pos_score(h_emb, t_emb):
    """ Calculate Dot product Score for positive pairs

        score = sum(head_emb * tail_emb)

        Parameters
        ----------
        h_emb: th.Tensor
            Head node embedding
        t_emb: th.Tensor
            Tail node embedding

        Returns
        -------
        Dot product score: th.Tensor
    """
    # DistMult
    score = th.sum(h_emb * t_emb, dim=-1)
    return score

def calc_dot_neg_tail_score(heads, tails, num_chunks, chunk_size,
    neg_sample_size, device=None):
    """ Calculate Dot product Score for negative pairs when tail nodes are negative

        score = sum(head_emb * tail_emb)

        Parameters
        ----------
        heads: th.Tensor
            Head node embedding
        tails: th.Tensor
            Tail node embedding
        num_chunks: int
            Number of shared negative chunks
        chunk_size: int
            Chunk size
        neg_sample_size: int
            Number of negative samples for each positive node
        device: th.device
            Device to run the computation

        Returns
        -------
        Dot product score: th.Tensor
    """
    hidden_dim = heads.shape[1]

    if device is not None:
        heads = heads.to(device)
        tails = tails.to(device)
    tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
    tails = th.transpose(tails, 1, 2)
    tmp = heads.reshape(num_chunks, chunk_size, hidden_dim)
    return th.bmm(tmp, tails)

def calc_dot_neg_head_score(heads, tails, num_chunks, chunk_size,
    neg_sample_size, device=None):
    """ Calculate Dot product Score for negative pairs when head nodes are negative

        score = sum(head_emb * tail_emb)

        Parameters
        ----------
        heads: th.Tensor
            Head node embedding
        tails: th.Tensor
            Tail node embedding
        num_chunks: int
            Number of shared negative chunks
        chunk_size: int
            Chunk size
        neg_sample_size: int
            Number of negative samples for each positive node
        device: th.device
            Device to run the computation

        Returns
        -------
        Dot product score: th.Tensor
    """
    hidden_dim = tails.shape[1]
    if device is not None:
        heads = heads.to(device)
        tails = tails.to(device)
    heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
    heads = th.transpose(heads, 1, 2)
    tmp = tails.reshape(num_chunks, chunk_size, hidden_dim)
    return th.bmm(tmp, heads)

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

def save_model(conf, model_path, gnn_model=None, embed_layer=None, bert_model=None, decoder=None):
    """ A model should have three parts:
        * GNN model
        * embedding layer
        * Bert model.
        We may have multiple Bert models.
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
        bert_model: model or a dict of models
            A bert model or a dict of bert models for multiple node types.
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

    if bert_model is not None:
        local_bert_model = {}
        for ntype in bert_model:
            local_bert_model[ntype] = bert_model[ntype].module \
                    if isinstance(bert_model[ntype], DistributedDataParallel) \
                    else bert_model[ntype]
        bert_model = local_bert_model

    model_states = {}
    if gnn_model is not None:
        model_states['gnn'] = gnn_model.state_dict()
    if embed_layer is not None:
        model_states['embed'] = embed_layer.state_dict()
    if decoder is not None:
        model_states['decoder'] = decoder.state_dict()
    if bert_model is not None:
        for name in bert_model:
            model_states['bert/' + name] = bert_model[name].state_dict()

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

def save_opt_state(model_path, dense_opt, fine_tune_opt, emb_opt):
    """ Save the states of the optimizers.

        There are usually three optimizers:
        * for the dense model parameters.
        * for fine-tuning the BERT model.
        * for the sparse embedding layers.

        Parameters
        ----------
        model_path : str
            The path of the folder where the model is saved.
            We save the optimizer states with the model.
        dense_opt : optimizer
            The optimizer for dense model parameters.
        fine_tune_opt : optimizer
            The optimizer for fine-tuning the BERT models.
        emb_opt : optimizer
            The optimizer for sparse embedding layer.
    """
    opt_states = {}
    if dense_opt is not None:
        opt_states['dense'] = dense_opt.state_dict()
    if fine_tune_opt is not None:
        opt_states['fine_tune'] = fine_tune_opt.state_dict()
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

def load_model(model_path, gnn_model=None, embed_layer=None, bert_model=None, decoder=None):
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
        bert_model: model
            Bert model to load
        decoder: model
            Decoder to load
    """
    gnn_model = gnn_model.module \
        if isinstance(gnn_model, DistributedDataParallel) else gnn_model
    embed_layer = embed_layer.module \
        if isinstance(embed_layer, DistributedDataParallel) else embed_layer
    decoder = decoder.module \
        if isinstance(decoder, DistributedDataParallel) else decoder
    if isinstance(bert_model, dict):
        local_bert_model = {}
        for ntype in bert_model:
            local_bert_model[ntype] = bert_model[ntype].module \
                    if isinstance(bert_model[ntype], DistributedDataParallel) \
                    else bert_model[ntype]
        bert_model = local_bert_model
    else:
        bert_model = bert_model.module \
            if isinstance(bert_model, DistributedDataParallel) else bert_model

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
    if bert_model is not None:
        for name in bert_model:
            model_name = 'bert/' + name
            if model_name in checkpoint:
                print(f"Loading BERT model for {model_name}")
                bert_model[name].load_state_dict(checkpoint[model_name])
    else:
        if 'bert' in checkpoint:
            print("Loading BERT model")
            bert_model.load_state_dict(checkpoint['bert'])
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

def load_opt_state(model_path, dense_opt, fine_tune_opt, emb_opt):
    """ Load the optimizer states and resotre the optimizers.

        Parameters
        ----------
        model_path: str
            The path of the model is saved.
        dense_opt: optimizer
            Optimzer for dense layers
        fine_tune_opt: optimizer
            Optimzer for bert
        emb_opt: optimizer
            Optimizer for emb layer
    """
    checkpoint = th.load(os.path.join(model_path, 'optimizers.bin'))
    dense_opt.load_state_dict(checkpoint['dense'])
    if 'fine_tune' in checkpoint and fine_tune_opt is not None:
        fine_tune_opt.load_state_dict(checkpoint['fine_tune'])
    # TODO(zhengda) we need to change DGL to make it work.
    if 'emb' in checkpoint and emb_opt is not None:
        raise NotImplementedError('We cannot load the state of sparse optimizer')
    #    emb_opt.load_state_dict(checkpoint['emb'])

def calc_ranking(pos_score, neg_score):
    """ Calculate ranking of positive scores among negative scores

        Parameters
        ----------
        pos_score: torch.Tensor
            positive scores
        neg_score: torch.Tensor
            negative screos

        Returns
        -------
        ranking of positive scores: th.Tensor
    """
    pos_score = pos_score.view(-1, 1)
    # perturb object
    scores = th.cat([pos_score, neg_score], dim=1)
    scores = th.sigmoid(scores)
    _, indices = th.sort(scores, dim=1, descending=True)
    indices = th.nonzero(indices == 0)
    rankings = indices[:, 1].view(-1) + 1
    rankings = rankings.cpu().detach()

    return rankings

def gen_lp_score(ranking):
    """ Get link prediction metrics

        Parameters
        ----------
        ranking:
            ranking of each positive edge

        Returns
        -------
        link prediction eval metrics: list of dict
    """
    logs = []
    for rank in ranking:
        logs.append({
            'MRR': 1.0 / rank,
            'MR': float(rank),
            'HITS@1': 1.0 if rank <= 1 else 0.0,
            'HITS@3': 1.0 if rank <= 3 else 0.0,
            'HITS@10': 1.0 if rank <= 10 else 0.0
        })
    return logs

# pylint: disable=invalid-name
def fullgraph_eval(g, embs, relation_embs, device, target_etype, pos_eids,
                   num_negative_edges_eval=-1, client=None, mlflow_report_frequency=100):
    """ The evaluation is done in a minibatch fasion.
        Firstly, we use fullgraph_emb to generate the node embeddings for all the nodes
        in g. And then evaluate each positive edge with all possible negative edges.
        Negative edges are constructed as: given a positive edge and a selected (randomly or
        sequentially) edge, we substitute the head node in the positive edge with the head node
        in the selected edge to construct one negative edge and substitute the tail node in the
        positive edge with the taisl node in the selected edge to construct another negative edge.

        Parameters
        ----------
        g: DGLGraph
            Validation graph or testing graph
        embs: th.Tensor
            The embedding
        relation_embs: th.Tensor
            Relation embedding
        device: th device
            The device to run evaluation
        target_etype: str or tuple of str
            Edge type to do evaluation
        pos_eids: th.Tensor
            Positive edge ids
        num_negative_edges_eval: int
            Number of negative edges for each positive edge. if -1, use all edges.
            Default: -1
        client: Mlflow client
            Mlflow log client
        mlflow_report_frequency: int
            MLflow report frequency
    """
    metrics = {}
    with th.no_grad():
        srctype, etype_name, dsttype =g.to_canonical_etype(etype=target_etype)

        # calculate mmr for the target etype
        logs = []
        t0 = time.time()
        pos_batch_size = 1024
        pos_cnt = pos_eids.shape[0]
        # we find the head and tail for the positive edge ids (used in testing)
        u, v = g.find_edges(pos_eids,etype=etype_name)
        # randomly select num_negative_edges_eval edges and
        # corrupt them int neg heads and neg tails
        if num_negative_edges_eval > 0:
            rand_n_u = th.randint(g.num_nodes(srctype),
                (num_negative_edges_eval * ((pos_cnt // pos_batch_size) + 1),))
            rand_n_v = th.randint(g.num_nodes(dsttype),
                (num_negative_edges_eval * ((pos_cnt // pos_batch_size) + 1),))
        # treat all nodes in head or tail as neg nodes
        else:
            n_u = th.arange(g.num_nodes(srctype))
            n_v = th.arange(g.num_nodes(dsttype))


        # batch based evaluation to fit in GPU
        return_metric_per_trainer = {
            'MRR': 0,
            'MR': 0,
            'HITS@1': 0,
            'HITS@3': 0,
            'HITS@10': 0
        }

        for p_i in range(int((pos_cnt + pos_batch_size - 1) // pos_batch_size)):
            if client is not None and p_i % mlflow_report_frequency == 0:
                client.log_param("Dummy", "Keep alive")

            left_batch_end = p_i * pos_batch_size
            right_batch_end = (p_i + 1) * pos_batch_size \
                if (p_i + 1) * pos_batch_size < pos_cnt else pos_cnt

            s_pu = u[left_batch_end: right_batch_end]
            s_pv = v[left_batch_end: right_batch_end]

            phead_emb = embs[srctype][s_pu].to(device)
            ptail_emb = embs[dsttype][s_pv].to(device)
            # calculate the positive batch score
            pos_score = calc_dot_pos_score(phead_emb, ptail_emb).cpu() \
                            if relation_embs is None else \
                                calc_distmult_pos_score(
                                    phead_emb, ptail_emb, relation_embs, device).cpu()
            if num_negative_edges_eval > 0:
                n_u = rand_n_u[p_i * num_negative_edges_eval:(p_i + 1) * num_negative_edges_eval]
                n_v = rand_n_v[p_i * num_negative_edges_eval:(p_i + 1) * num_negative_edges_eval]
                n_u, _ = th.sort(n_u)
                n_v, _ = th.sort(n_v)

            neg_batch_size = 10000
            head_num_negative_edges_eval = n_u.shape[0]
            tail_num_negative_edges_eval = n_v.shape[0]
            t_neg_score = []
            h_neg_score = []
            # we calculate and collect the negative scores for the perturbed
            # tail and haid nodes in batch form head node scores
            for n_i in range(int((head_num_negative_edges_eval + neg_batch_size - 1) \
                // neg_batch_size)):
                sn_u = n_u[n_i * neg_batch_size : \
                        (n_i + 1) * neg_batch_size \
                        if (n_i + 1) * neg_batch_size < head_num_negative_edges_eval
                        else head_num_negative_edges_eval]
                nhead_emb = embs[srctype][sn_u].to(device)
                if relation_embs is None:
                    h_neg_score.append(
                        calc_dot_neg_head_score(nhead_emb,
                                                ptail_emb,
                                                1,
                                                ptail_emb.shape[0],
                                                nhead_emb.shape[0],
                                                device).reshape(-1, nhead_emb.shape[0]).cpu())
                else:
                    h_neg_score.append(
                        calc_distmult_neg_head_score(nhead_emb,
                                                     ptail_emb,
                                                     relation_embs,
                                                     1,
                                                     ptail_emb.shape[0],
                                                     nhead_emb.shape[0],
                                                     device).reshape(-1, nhead_emb.shape[0]).cpu())

            # tail node scores
            for n_i in range(int((tail_num_negative_edges_eval + neg_batch_size - 1) \
                // neg_batch_size)):
                sn_v = n_v[n_i * neg_batch_size : \
                        (n_i + 1) * neg_batch_size \
                        if (n_i + 1) * neg_batch_size < tail_num_negative_edges_eval \
                        else tail_num_negative_edges_eval]
                ntail_emb = embs[dsttype][sn_v].to(device)
                if relation_embs is None:
                    t_neg_score.append(
                        calc_dot_neg_tail_score(phead_emb,
                                                ntail_emb,
                                                1,
                                                phead_emb.shape[0],
                                                ntail_emb.shape[0],
                                                device).reshape(-1, ntail_emb.shape[0]).cpu())
                else:
                    t_neg_score.append(
                        calc_distmult_neg_tail_score(phead_emb,
                                                     ntail_emb,
                                                     relation_embs,
                                                     1,
                                                     phead_emb.shape[0],
                                                     ntail_emb.shape[0],
                                                     device).reshape(-1, ntail_emb.shape[0]).cpu())

            # the following piece of code filters the false negative edges from the scores
            t_neg_score = th.cat(t_neg_score, dim=1)
            h_neg_score = th.cat(h_neg_score, dim=1)

            # the following pieces of code calculate the mrr for the perturbed head and tail nodes
            h_ranking = calc_ranking(pos_score, h_neg_score)
            # perturb subject
            t_ranking = calc_ranking(pos_score, t_neg_score)

            # print((left_batch_end, right_batch_end), eids.shape, h_ranking.shape)
            h_logs = gen_lp_score(h_ranking.float())
            t_logs = gen_lp_score(t_ranking.float())
            logs = h_logs + t_logs
            metrics = {}
            for metric in logs[0]:
                metrics[metric] = sum(log[metric] for log in logs) / len(logs)
                return_metric_per_trainer[metric] += metrics[metric]
        for metric in return_metric_per_trainer:
            return_metric_per_trainer[metric] = \
                th.tensor(return_metric_per_trainer[metric] / \
                    int((pos_cnt + pos_batch_size - 1) // pos_batch_size)).to(device)
        t1 = time.time()

        # When world size == 1, we do not need the barrier
        if th.distributed.get_world_size() > 1:
            th.distributed.barrier()
        for _, metric_val in return_metric_per_trainer.items():
            th.distributed.all_reduce(metric_val)
        return_metric = {}
        for metric, metric_val in return_metric_per_trainer.items():
            return_metric_i = \
                metric_val / th.distributed.get_world_size()
            return_metric[metric] = return_metric_i.item()
        if g.rank() == 0:
            print(f"Full eval {pos_eids.shape[0]} exmpales takes {t1 - t0} seconds")

    return return_metric

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
def do_mini_batch_inference(model, embed_layer, bert_train,
                            bert_static, bert_hidden_size,
                            device, bert_emb_cache, target_nidx, g, pb, n_hidden,
                            fanout, eval_batch_size,
                            use_bert_embeddings_for_validation=False,
                            client=None,
                            mlflow_report_frequency=100, feat_field='feat'):
    """ Do mini batch inference

        Parameters
        ----------
        model: torch model
            GNN model
        embed_layer: torch model
            GNN input embedding layer
        bert_train: bert model
            A trainable bert wrapper
        bert_static: bert model
            A static bert wrapper
        bert_hidden_size: int
            A dict of hidden sizes of bert models
        device: th.device
            Device
        bert_emb_cache: dict of th.Tensor
            A global bert cache
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
        use_bert_embeddings_for_validation: bool
             whether use bert embeddings only
        client: Mlflow client
            Mlflow log client
        mlflow_report_frequency: int
            MLflow report frequency
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
            if 'trainer_id' in g.nodes[key].data \
            else None
        target_idx_dist = dgl.distributed.node_split(
                target_mask.bool(),
                pb, ntype=key, force_even=False,
                node_trainer_ids=node_trainer_ids)
        target_idxs_dist[key] = target_idx_dist

        if use_bert_embeddings_for_validation or embed_layer is None:
            hidden_size = (bert_hidden_size[key] \
                if isinstance(bert_hidden_size, dict) \
                else bert_hidden_size)
        else:
            hidden_size = n_hidden
        embeddings[key] = dgl.distributed.DistTensor(
            (g.number_of_nodes(key), hidden_size),
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
    for ntype in g.ntypes:
        if len(bert_train) > 0:
            if isinstance(bert_train, dict):
                if ntype in bert_train.keys():
                    bert_train[ntype].eval()
                    bert_static[ntype].eval()
            else:
                bert_train.eval()
        # else pure GNN
    th.cuda.empty_cache()
    with th.no_grad():
        for iter_l, (input_nodes, seeds, blocks) in enumerate(loader):
            if client is not None and iter_l % mlflow_report_frequency == 0:
                client.log_param("Dummy", "Keep alive")

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
            if use_bert_embeddings_for_validation:
                train_mask = {ntype: th.full((seed.shape[0],), False, dtype=th.bool) \
                                for ntype, seed in seeds.items()}
                assert bert_hidden_size is None or isinstance(bert_hidden_size, dict)
                final_embs={}
                for ntype, nid in seeds.items():
                    mask = train_mask[ntype] if train_mask is not None else None
                    text_embs, _ = \
                        extract_bert_embed(nid=nid,
                                           mask=mask,
                                           bert_train=bert_train[ntype],
                                           bert_static=bert_static[ntype],
                                           emb_cache=bert_emb_cache[ntype] \
                                               if bert_emb_cache is not None \
                                               else None,
                                           bert_hidden_size=bert_hidden_size[ntype] \
                                               if isinstance(bert_hidden_size, dict) \
                                               else bert_hidden_size,
                                            dev=device)
                    final_embs[ntype] = text_embs.type(th.float32)
            else:
                train_mask = {ntype: th.full((input.shape[0],), False, dtype=th.bool) \
                                for ntype, input in input_nodes.items()}
                inputs, _ = prepare_batch_input(g,
                                                bert_train,
                                                bert_static,
                                                bert_hidden_size,
                                                input_nodes,
                                                train_mask=train_mask,
                                                emb_cache=bert_emb_cache,
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
    for ntype in g.ntypes:
        if len(bert_train) > 0:
            if isinstance(bert_train, dict):
                if ntype in bert_train.keys():
                    bert_train[ntype].train()
                    bert_static[ntype].train()
            else:
                bert_train.train()
    g.barrier()
    t1 = time.time()
    if use_bert_embeddings_for_validation:
        for key in final_embs:
            lembeddings = embeddings[key][0:g.number_of_nodes(key)][:]
            embeddings[key] = th.nn.functional.normalize(lembeddings, p=2, dim=1)

    if use_bert_embeddings_for_validation and g.rank() == 0:
        print(f'Computing language model embeddings: {(t1 - t0):.4f} seconds')
    elif g.rank() == 0:
        print(f'Computing GNN embeddings: {(t1 - t0):.4f} seconds')

    if target_nidx is not None:
        embeddings = {ntype: LazyDistTensor(embeddings[ntype], target_nidx[ntype]) \
            for ntype in target_nidx.keys()}
    return embeddings

def do_fullgraph_infer(g, model, embed_layer, bert_train, bert_static,
                       bert_hidden_size, device, bert_emb_cache, bert_infer_bs, eval_fanout_list,
                       eval_batch_size=None, client=None,
                       mlflow_report_frequency=100, feat_field='feat'):
    """ Do fullgraph inference

        Parameters
        ----------
        g: DistDGLGraph
            DGLGraph
        model: torch model
            GNN model
        embed_layer: torch model
            GNN input embedding layer
        bert_train: bert model
            A trainable bert wrapper
        bert_static: bert model
            A static bert wrapper
        bert_hidden_size: int
            A dict of hidden sizes of bert models
        device: th.device
            Device
        bert_emb_cache: dict of th.Tensor
            A global bert cache
        bert_infer_bs: int
            Bert inference batch size
        eval_fanout_list: list
            The evaluation fanout list
        eval_batch_size: int
            The batch size
        client: Mlflow client
            Mlflow log client
        mlflow_report_frequency: int
            MLflow report frequency
        feat_field: str
            Field to extract features

        Returns
        -------
        Node embeddings: dict of str to th.Tensor
    """
    t0 = time.time() # pylint: disable=invalid-name
    node_embed = extract_all_embeddings_dist(g,
        bert_infer_bs, embed_layer,
        bert_train, bert_static,
        bert_hidden_size, dev=device,
        emb_cache=bert_emb_cache,
        client=client, mlflow_report_frequency=mlflow_report_frequency,
        feat_field=feat_field)
    t1 = time.time() # pylint: disable=invalid-name
    # full graph evaluation
    g.barrier()
    model.eval()
    if not isinstance(model, DistributedDataParallel):
        embeddings = model.dist_inference(g, eval_batch_size, device, 0,
            node_embed, eval_fanout_list, client=client,
            mlflow_report_frequency=mlflow_report_frequency)
    else:
        embeddings = model.module.dist_inference(g, eval_batch_size,
            device, 0, node_embed, eval_fanout_list, client=client,
            mlflow_report_frequency=mlflow_report_frequency)
    if g.rank() == 0:
        print(f"computing Bert embeddings: {t1 - t0:.4f} seconds, " \
              f"computing GNN embeddings: {time.time() - t1:.4f} seconds")
    model.train()
    return embeddings

def trim_data(nids, device):
    """ In distributed traning scenario, we need to make sure that
        each worker has same number of batches. Otherwise the synchronization
        (barrier) is called diffirent times, which results in the worker
        with more batches hangs up.

        This function trims the nids to the same size for all workers.

        Parameters
        ----------
        nids: th.Tensor
            node ids
        device: th.device
            Device

        Returns
        -------
        Trimed nids: th.Tensor
    """
    # NCCL backend only supports GPU tensors, thus here we need to allocate it to gpu
    num_nodes = th.tensor(nids.numel()).to(device)
    assert num_nodes.is_cuda, "NCCL does not support CPU all_reduce"
    dist.all_reduce(num_nodes, dist.ReduceOp.MIN)
    min_num_nodes = int(num_nodes)
    nids_length = nids.shape[0]
    if min_num_nodes < nids_length:
        new_nids = nids[:min_num_nodes]
        print(f"Pad nids from {nids_length} to {min_num_nodes}")
    else:
        new_nids = nids
    assert new_nids.shape[0] == min_num_nodes
    return new_nids

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

def broadcast_data(rank, world_size, data_tensor):
    """ Broadcast local data to all trainers in the cluster using all2all

        After broadcast_data, each trainer will get all the data (data_tensor)

        Parameters
        ----------
        rank : int
            The rank of current worker
        world_size : int
            The size of the entire
        data_tensor:
            Data to exchange
    """
    if world_size == 1: # world size is 1, nothing to do
        return data_tensor

    # exchange the data size of each trainer
    if th.distributed.get_backend() == "gloo":
        device = "cpu"
    elif th.distributed.get_backend() == "nccl":
        assert data_tensor.is_cuda
        device = data_tensor.device
    else:
        assert False, f"backend {th.distributed.get_backend()} not supported."

    data_size = th.zeros((world_size,), dtype=th.int64, device=device)
    data_size[rank] = data_tensor.shape[0]
    th.distributed.all_reduce(data_size,
        op=th.distributed.ReduceOp.SUM)

    gather_list = [th.empty([int(size)]+list(data_tensor.shape[1:]),
        dtype=data_tensor.dtype,
        device=device) for size in data_size]
    data_tensors = [data_tensor for _ in data_size]
    if th.distributed.get_backend() == "gloo":
        alltoallv_cpu(rank, world_size, gather_list, data_tensors)
    else: #th.distributed.get_backend() == "nccl"
        alltoallv_nccl(rank, world_size, gather_list, data_tensors)


    data_tensor = th.cat(gather_list, dim=0)
    return data_tensor
