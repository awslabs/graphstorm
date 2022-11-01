import time
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
            'MRR': 1.0 / rank,
            'MR': float(rank),
            'HITS@1': 1.0 if rank <= 1 else 0.0,
            'HITS@3': 1.0 if rank <= 3 else 0.0,
            'HITS@10': 1.0 if rank <= 10 else 0.0
        })
    return logs

# pylint: disable=invalid-name
def fullgraph_eval(g, embs, relation_embs, device, target_etype, pos_eids,
                   num_negative_edges_eval=-1, task_tracker=None):
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
        task_tracker: GSTaskTrackerAbc
            task tracker
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
            if task_tracker is not None:
                task_tracker.keep_alive(p_i)

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
