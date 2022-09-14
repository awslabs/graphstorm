import torch as th
import numpy as np
from collections.abc import Mapping
import dgl
from dgl import backend as F
from dgl.dataloading.negative_sampler import Uniform

class LocalUniform(Uniform):
    """Negative sampler that randomly chooses negative destination nodes
    for each source node according to a uniform distribution.
    For each edge ``(u, v)`` of type ``(srctype, etype, dsttype)``, DGL generates
    :attr:`k` pairs of negative edges ``(u, v')``, where ``v'`` is chosen
    uniformly from all the nodes of type ``dsttype``.  The resulting edges will
    also have type ``(srctype, etype, dsttype)``.

    Parameters
    ----------
    k : int
        The number of negative examples per edge.
    Examples
    --------
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
    >>> neg_sampler(g, [0, 1])
    (tensor([0, 0, 1, 1]), tensor([1, 0, 2, 3]))
    """
    def __init__(self, k, per_trainer=False):
        self._local_neg_nids = {}
        self._per_trainer = per_trainer
        super(LocalUniform, self).__init__(k)

    def _generate(self, g, eids, canonical_etype):
        _, _, vtype = canonical_etype
        shape = F.shape(eids)
        dtype = F.dtype(eids)
        ctx = F.context(eids)
        shape = (shape[0] * self.k,)
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = F.repeat(src, self.k, 0)

        if vtype not in self._local_neg_nids:
            pb = g.get_partition_book()
            if self._per_trainer:
                neg_idx = dgl.distributed.node_split(th.full((g.num_nodes(vtype),), True, dtype=th.bool),
                                                     pb, ntype=vtype, force_even=False,
                                                     node_trainer_ids=g.nodes[vtype].data['trainer_id'])
            else:
                neg_idx = dgl.distributed.node_split(th.full((g.num_nodes(vtype),), True, dtype=th.bool),
                                                     pb, ntype=vtype, force_even=True)
            self._local_neg_nids[vtype] = neg_idx

        dst = F.randint(shape, dtype, ctx, 0, self._local_neg_nids[vtype].shape[0])
        return src, self._local_neg_nids[vtype][dst]

class JointUniform(object):
    '''Jointly corrupt a group of edges.
    The main idea is to sample a set of nodes and use them to corrupt all edges in a mini-batch.
    This algorithm won't change the sampling probability for each individual edge, but can
    significantly reduce the number of nodes in a mini-batch.

    Parameters
    ----------
    k : int
        The number of negative examples per edge.
    '''
    def __init__(self, k):
        self.k = k

    def _generate(self, g, eids, canonical_etype):
        _, _, vtype = canonical_etype
        shape = eids.shape
        dtype = eids.dtype
        ctx = eids.device
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = np.repeat(src.numpy(), self.k)
        dst = th.randint(g.number_of_nodes(vtype), (shape[0],), dtype=dtype, device=ctx)
        dst = np.tile(dst, self.k)
        return th.as_tensor(src), th.as_tensor(dst)

    def __call__(self, g, eids):
        """Returns negative examples.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        eids : Tensor or dict[etype, Tensor]
            The sampled edges in the minibatch.

        Returns
        -------
        tuple[Tensor, Tensor] or dict[etype, tuple[Tensor, Tensor]]
            The returned source-destination pairs as negative examples.
        """
        if isinstance(eids, Mapping):
            eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
            neg_pair = {k: self._generate(g, v, k) for k, v in eids.items()}
        else:
            assert len(g.etypes) == 1, \
                'please specify a dict of etypes and ids for graphs with multiple edge types'
            neg_pair = self._generate(g, eids, g.canonical_etypes[0])

        return neg_pair
