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

    Addtional graph samplers for GSF
"""
from collections.abc import Mapping
import torch as th
import numpy as np
from dgl import backend as F
from dgl.distributed import node_split
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
    def __init__(self, k):
        self._local_neg_nids = {}
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
            neg_idx = node_split(th.full((g.num_nodes(vtype),), True, dtype=th.bool),
                                    pb, ntype=vtype, force_even=True)
            self._local_neg_nids[vtype] = neg_idx

        dst = F.randint(shape, dtype, ctx, 0, self._local_neg_nids[vtype].shape[0])
        return src, self._local_neg_nids[vtype][dst]

class GlobalUniform(Uniform):
    """Negative sampler that randomly chooses negative destination nodes
    for each source node according to a uniform distribution.
    """

    def gen_neg_pairs(self, g, pos_pairs):
        """Returns negative examples associated with positive examples.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        pos_pairs : (Tensor, Tensor) or dict[etype, (Tensor, Tensor)]
            The positive node pairs

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor] or
        dict[etype, tuple(Tensor, Tensor Tensor, Tensor)
            The returned [positive source, negative source,
            postive destination, negatve destination]
            tuples as pos-neg examples.
        """
        def _gen_neg_pair(pos_pair, utype, vtype):
            """ generate negative pairs
            """
            src, pos_dst = pos_pair
            shape = src.shape
            ctx = src.device
            neg_dst = th.randint(g.number_of_nodes(vtype),
                (shape[0], self.k), device=ctx)
            neg_src = th.randint(g.number_of_nodes(utype),
                (shape[0], self.k), device=ctx)
            return (src, neg_src, pos_dst, neg_dst)

        if isinstance(pos_pairs, Mapping):
            pos_neg_tuple = {}
            for canonical_etype, pos_pair in pos_pairs.items():
                utype, _, vtype = canonical_etype
                pos_neg_tuple[canonical_etype] = _gen_neg_pair(pos_pair, utype, vtype)
        else:
            assert len(g.canonical_etypes) == 1, \
                'please specify a dict of etypes and ids for graphs with multiple edge types'
            pos_neg_tuple = _gen_neg_pair(pos_pairs,
                g.canonical_etypes[0][0], g.canonical_etypes[0][2])
        return pos_neg_tuple

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
            assert len(g.canonical_etypes) == 1, \
                'please specify a dict of etypes and ids for graphs with multiple edge types'
            neg_pair = self._generate(g, eids, g.canonical_etypes[0])

        return neg_pair

    def gen_neg_pairs(self, g, pos_pairs):
        """Returns negative examples associated with positive examples.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        pos_pairs : (Tensor, Tensor) or dict[etype, (Tensor, Tensor)]
            The positive node pairs

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor] or
        dict[etype, tuple(Tensor, Tensor Tensor, Tensor)
            The returned [positive source, negative source,
            postive destination, negatve destination]
            tuples as pos-neg examples.
        Note: we only corrupt destination nodes here. We will
        set negative source to None.
        """
        def _gen_neg_pair(pos_pair, utype, vtype):
            """ generate negative pairs
            """
            src, pos_dst = pos_pair
            dtype = src.dtype
            ctx = src.device
            # only sample k negatives, they will be shared across all pos
            neg_dst = th.randint(g.number_of_nodes(vtype),
                (self.k,), dtype=dtype, device=ctx)
            neg_src = th.randint(g.number_of_nodes(utype),
                (self.k,), dtype=dtype, device=ctx)
            return (src, neg_src, pos_dst, neg_dst)

        if isinstance(pos_pairs, Mapping):
            pos_neg_tuple = {}
            for canonical_etype, pos_pair in pos_pairs.items():
                utype, _, vtype = canonical_etype
                pos_neg_tuple[canonical_etype] = _gen_neg_pair(pos_pair, utype, vtype)
        else:
            assert len(g.canonical_etypes) == 1, \
                'please specify a dict of etypes and ids for graphs with multiple edge types'
            pos_neg_tuple = _gen_neg_pair(pos_pairs,
                g.canonical_etypes[0][0], g.canonical_etypes[0][2])
        return pos_neg_tuple

class JointLocalUniform(JointUniform):
    '''Jointly corrupt a group of edges.

    The main idea is to sample a set of nodes and use them to corrupt all edges in a mini-batch.
    This algorithm won't change the sampling probability for each positive edge, but can
    significantly reduce the number of nodes in a mini-batch.
    The difference between JointUniform and JointLocalUniform is that JointUniform sample
    negative nodes from the entire graph, but JointLocalUniform only sample negative nodes
    from the local partition.

    Parameters
    ----------
    k : int
        The number of negative examples per edge.
    '''
    def __init__(self, k):
        self._local_neg_nids = {}
        super(JointLocalUniform, self).__init__(k)

    def _generate(self, g, eids, canonical_etype):
        _, _, vtype = canonical_etype
        shape = eids.shape
        dtype = eids.dtype
        ctx = eids.device
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = np.repeat(src.numpy(), self.k)

        if vtype not in self._local_neg_nids:
            pb = g.get_partition_book()
            neg_idx = node_split(th.full((g.num_nodes(vtype),), True, dtype=th.bool),
                                    pb, ntype=vtype, force_even=True)
            self._local_neg_nids[vtype] = neg_idx

        dst = th.randint(len(self._local_neg_nids[vtype]), (shape[0],), dtype=dtype, device=ctx)
        dst = self._local_neg_nids[vtype][dst]
        dst = np.tile(dst, self.k)
        return th.as_tensor(src), th.as_tensor(dst)
