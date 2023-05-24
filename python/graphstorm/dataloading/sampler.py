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
from dgl import transforms
from dgl.distributed import node_split
from dgl.dataloading import utils
from dgl.dataloading.negative_sampler import Uniform
from dgl.dataloading.dist_dataloader import EdgeCollator, _find_exclude_eids
from dgl.base import EID, NID
from dgl.convert import heterograph

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
                neg_idx = node_split(th.full((g.num_nodes(vtype),), True, dtype=th.bool),
                                     pb, ntype=vtype, force_even=False,
                                     node_trainer_ids=g.nodes[vtype].data['trainer_id'])
            else:
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

class WeightedEdgeCollator(EdgeCollator):
    """DGL collator with edge weight stored in both pos_graph

       The implementation follows dgl.dataloading.EdgeCollator
    """
    def __init__(self, g, eids, graph_sampler, g_sampling=None, exclude=None,
                 reverse_eids=None, reverse_etypes=None, negative_sampler=None,
                 edge_weight_fields=None):
        self.edge_weight_fields = edge_weight_fields
        super(WeightedEdgeCollator, self).__init__(
            g, eids, graph_sampler, g_sampling, exclude,
            reverse_eids, reverse_etypes, negative_sampler)

    def _collate_with_negative_sampling(self, items):
        """

            Parameters
            ----------
            items : list[int] or list[tuple[str, int]]
                Either a list of edge IDs (for homogeneous graphs), or a list of edge type-ID
                pairs (for heterogeneous graphs).
        """
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = utils.group_as_dict(items)
        items = utils.prepare_tensor_or_dict(self.g_sampling, items, "items")

        pair_graph = self.g.edge_subgraph(items, relabel_nodes=False)
        induced_edges = pair_graph.edata[EID]

        neg_srcdst = self.negative_sampler(self.g, items)
        if not isinstance(neg_srcdst, Mapping):
            assert len(self.g.etypes) == 1, (
                "graph has multiple or no edge types; "
                "please return a dict in negative sampler."
            )
            neg_srcdst = {self.g.canonical_etypes[0]: neg_srcdst}
        # Get dtype from a tuple of tensors
        dtype = F.dtype(list(neg_srcdst.values())[0][0])
        ctx = F.context(pair_graph)
        neg_edges = {
            etype: neg_srcdst.get(
                etype,
                (
                    F.copy_to(F.tensor([], dtype), ctx),
                    F.copy_to(F.tensor([], dtype), ctx),
                ),
            )
            for etype in self.g.canonical_etypes
        }
        neg_pair_graph = heterograph(
            neg_edges,
            {ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes},
        )

        pair_graph, neg_pair_graph = transforms.compact_graphs(
            [pair_graph, neg_pair_graph]
        )
        pair_graph.edata[EID] = induced_edges
        # Get edge weight
        for etype in pair_graph.canonical_etypes:
            pair_graph.edges[etype].data[self.edge_weight_fields] = \
                self.g.edges[etype].data[self.edge_weight_fields][induced_edges[etype]]

        seed_nodes = pair_graph.ndata[NID]

        exclude_eids = _find_exclude_eids(
            self.g_sampling,
            self.exclude,
            items,
            reverse_eid_map=self.reverse_eids,
            reverse_etype_map=self.reverse_etypes,
        )

        input_nodes, _, blocks = self.graph_sampler.sample_blocks(
            self.g_sampling, seed_nodes, exclude_eids=exclude_eids
        )

        return input_nodes, pair_graph, neg_pair_graph, blocks
