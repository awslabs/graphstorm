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

    Utility functions for evaluation
"""
import torch as th

from ..data.utils import alltoallv_nccl, alltoallv_cpu

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
            'mrr': 1.0 / rank,
            'mr': float(rank),
            'hits@1': 1.0 if rank <= 1 else 0.0,
            'hits@3': 1.0 if rank <= 3 else 0.0,
            'hits@10': 1.0 if rank <= 10 else 0.0
        })
    metrics = {}
    for metric in logs[0]:
        metrics[metric] = th.tensor(sum(log[metric] for log in logs) / len(logs))
    return metrics

def gen_mrr_score(ranking):
    """ Get link prediction mrr metrics

        Parameters
        ----------
        ranking:
            ranking of each positive edge

        Returns
        -------
        link prediction eval metrics: list of dict
    """
    logs = th.div(1.0, ranking)
    metrics = {"mrr": th.tensor(th.div(th.sum(logs),len(logs)))}
    return metrics


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
