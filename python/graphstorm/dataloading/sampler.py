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
import os
import logging
import random
from collections.abc import Mapping
import torch as th
import numpy as np
from dgl import backend as F
from dgl import EID, NID
from dgl.distributed import node_split
from dgl.dataloading.negative_sampler import (Uniform,
                                              _BaseNegativeSampler)
from dgl.dataloading import NeighborSampler
from dgl.transforms import to_block

from ..utils import is_wholegraph

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

class GSHardEdgeDstNegativeSampler(_BaseNegativeSampler):
    """ GraphStorm negative sampler that chooses negative destination nodes
        from a fixed set to create negative edges.

        Parameters
        ----------
        k: int
            Number of negatives to sample.
        dst_negative_field: str or dict of str
            The field storing the hard negatives.
        negative_sampler: sampler
            The negative sampler to generate negatives
            if there is not enough hard negatives.
        num_hard_negs: int or dict of int
            Number of hard negatives.
    """
    def __init__(self, k, dst_negative_field, negative_sampler, num_hard_negs=None):
        assert is_wholegraph() is False, \
                "Hard negative is not supported for WholeGraph."
        self._dst_negative_field = dst_negative_field
        self._k = k
        self._negative_sampler = negative_sampler
        self._num_hard_negs = num_hard_negs

    def _generate(self, g, eids, canonical_etype):
        """ _generate() is called by DGL BaseNegativeSampler to generate negative pairs.

        See https://github.com/dmlc/dgl/blob/1.1.x/python/dgl/dataloading/negative_sampler.py#L7
        For more detials
        """
        if isinstance(self._dst_negative_field, str):
            dst_negative_field = self._dst_negative_field
        elif canonical_etype in self._dst_negative_field:
            dst_negative_field = self._dst_negative_field[canonical_etype]
        else:
            dst_negative_field = None

        if isinstance(self._num_hard_negs, int):
            required_num_hard_neg = self._num_hard_negs
        elif canonical_etype in self._num_hard_negs:
            required_num_hard_neg = self._num_hard_negs[canonical_etype]
        else:
            required_num_hard_neg = 0

        if dst_negative_field is None or required_num_hard_neg == 0:
            # no hard negative, fallback to random negative
            return self._negative_sampler._generate(g, eids, canonical_etype)

        hard_negatives = g.edges[canonical_etype].data[dst_negative_field][eids]
        # It is possible that different edges may have different number of
        # pre-defined negatives. For pre-defined negatives, the corresponding
        # value in `hard_negatives` will be integers representing the node ids.
        # For others, they will be -1s meaning there are missing fixed negatives.
        if th.sum(hard_negatives == -1) == 0:
            # Fast track, there is no -1 in hard_negatives
            max_num_hard_neg = hard_negatives.shape[1]
            neg_idx = th.randperm(max_num_hard_neg)
            # shuffle the hard negatives
            hard_negatives = hard_negatives[:,neg_idx]

            if required_num_hard_neg >= self._k and max_num_hard_neg >= self._k:
                # All negative should be hard negative and
                # there are enough hard negatives.
                hard_negatives = hard_negatives[:,:self._k]
                src, _ = g.find_edges(eids, etype=canonical_etype)
                src = F.repeat(src, self._k, 0)
                return src, hard_negatives.reshape((-1,))
            else:
                if required_num_hard_neg < max_num_hard_neg:
                    # Only need required_num_hard_neg hard negatives.
                    hard_negatives = hard_negatives[:,:required_num_hard_neg]
                    num_hard_neg = required_num_hard_neg
                else:
                    # There is not enough hard negative to fill required_num_hard_neg
                    num_hard_neg = max_num_hard_neg

                # There is not enough negatives
                src, neg = self._negative_sampler._generate(g, eids, canonical_etype)
                # replace random negatives with fixed negatives
                neg = neg.reshape(-1, self._k)
                neg[:,:num_hard_neg] = hard_negatives[:,:num_hard_neg]
                return src, neg.reshape((-1,))
        else:
            # slow track, we need to handle cases when there are -1s
            hard_negatives, _ = th.sort(hard_negatives, dim=1, descending=True)

            src, neg = self._negative_sampler._generate(g, eids, canonical_etype)
            for i in range(len(eids)):
                hard_negative = hard_negatives[i]
                # ignore -1s
                hard_negative = hard_negative[hard_negative > -1]
                max_num_hard_neg = hard_negative.shape[0]
                hard_negative = hard_negative[th.randperm(max_num_hard_neg)]

                if required_num_hard_neg < max_num_hard_neg:
                    # Only need required_num_hard_neg hard negatives.
                    hard_negative = hard_negative[:required_num_hard_neg]
                    num_hard_neg = required_num_hard_neg
                else:
                    num_hard_neg = max_num_hard_neg

                # replace random negatives with fixed negatives
                neg[i*self._k:i*self._k + (num_hard_neg \
                              if num_hard_neg < self._k else self._k)] = \
                    hard_negative[:num_hard_neg if num_hard_neg < self._k else self._k]
            return src, neg

class GSFixedEdgeDstNegativeSampler(object):
    """ GraphStorm negative sampler that uses fixed negative destination nodes
        to create negative edges.

        GSFixedEdgeDstNegativeSampler only works with test dataloader.

        Parameters
        ----------
        dst_negative_field: str or dict of str
            The field storing the hard negatives.
    """
    def __init__(self, dst_negative_field):
        assert is_wholegraph() is False, \
                "Hard negative is not supported for WholeGraph."
        self._dst_negative_field = dst_negative_field

    def gen_etype_neg_pairs(self, g, etype, pos_eids):
        """ Returns negative examples associated with positive examples.
            It only return dst negatives.

            This function is called by GSgnnLinkPredictionTestDataLoader._next_data()
            to generate testing edges.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        pos_eids : (Tensor, Tensor) or dict[etype, (Tensor, Tensor)]
            The positive edge ids.

        Returns
        -------
        dict[etype, tuple(Tensor, Tensor Tensor, Tensor)
            The returned [positive source, negative source,
            postive destination, negatve destination]
            tuples as pos-neg examples.
        """
        def _gen_neg_pair(eids, canonical_etype):
            src, pos_dst = g.find_edges(eids, etype=canonical_etype)

            if isinstance(self._dst_negative_field, str):
                dst_negative_field = self._dst_negative_field
            elif canonical_etype in self._dst_negative_field:
                dst_negative_field = self._dst_negative_field[canonical_etype]
            else:
                raise RuntimeError(f"{etype} does not have pre-defined negatives")

            fixed_negatives = g.edges[canonical_etype].data[dst_negative_field][eids]

            # Users may use HardEdgeDstNegativeTransform
            # to prepare the fixed negatives.
            assert th.sum(fixed_negatives == -1) == 0, \
                "When using fixed negative destination nodes to construct testing edges," \
                "it is required that for each positive edge there are enough negative " \
                f"destination nodes. Please check the {dst_negative_field} feature " \
                f"of edge type {canonical_etype}"

            num_fixed_neg = fixed_negatives.shape[1]
            logging.debug("The number of fixed negative is %d", num_fixed_neg)
            return (src, None, pos_dst, fixed_negatives)

        assert etype in g.canonical_etypes, \
            f"Edge type {etype} does not exist in graph. Expecting an edge type in " \
            f"{g.canonical_etypes}, but get {etype}"

        return {etype: _gen_neg_pair(pos_eids, etype)}

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

class InbatchJointUniform(JointUniform):
    '''Jointly corrupt a group of edges.
    The main idea is to sample a set of nodes and use them to corrupt all edges in a mini-batch.
    This algorithm won't change the sampling probability for each individual edge, but can
    significantly reduce the number of nodes in a mini-batch.
    '''

    def _generate(self, g, eids, canonical_etype):
        """ The return negative edges will be in the format of:
            src-uniform-joint | src-in-batch
            dst-uniform-joint | dst-in-batch

            The first part comes from uniform joint negative sampling.
            The second part comes from in batch negative sampling.
        """
        _, _, vtype = canonical_etype
        shape = eids.shape
        dtype = eids.dtype
        ctx = eids.device
        pos_src, pos_dst = g.find_edges(eids, etype=canonical_etype)
        pos_src = pos_src.numpy()
        num_pos_edges = len(pos_src)
        src = np.repeat(pos_src, self.k)
        dst = th.randint(g.number_of_nodes(vtype), (shape[0],), dtype=dtype, device=ctx)
        dst = np.tile(dst, self.k)
        if num_pos_edges > 1:
            # Only when there are more than 1 edges, we can do in batch negative
            src_in_batch = np.repeat(pos_src, num_pos_edges)
            dst_in_batch = np.repeat(pos_dst.reshape(1, -1), num_pos_edges, axis=0).reshape(-1,)

            # remove false negatives
            # 1 1 1 2 2 2 3 3 3
            # 1 2 3 1 2 3 1 2 3
            # ->
            # 1 1 2 2 3 3
            # 2 3 1 3 1 2
            in_batch_negs = th.ones(num_pos_edges*num_pos_edges, dtype=th.bool)
            false_negative_idx = th.arange(num_pos_edges) * (num_pos_edges + 1)
            in_batch_negs[false_negative_idx] = 0
            src_in_batch = th.as_tensor(src_in_batch)[in_batch_negs]
            dst_in_batch = th.as_tensor(dst_in_batch)[in_batch_negs]

            src = th.cat((th.as_tensor(src), src_in_batch))
            dst = th.cat((th.as_tensor(dst), dst_in_batch))
        else:
            src = th.as_tensor(src)
            dst = th.as_tensor(dst)
        return src, dst

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

class FastMultiLayerNeighborSampler(NeighborSampler):
    """ Fast MultiLayerNeighborSampler

        If mask is None, it acts the same as dgl.dataloading.MultiLayerNeighborSampler

    Parameters
    ----------
    reverse_edge_types_map: dict
        A map for reverse edge type
    fanouts : list[int] or list[dict[etype, int]]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.

        If only a single integer is provided, DGL assumes that every edge type
        will have the same fanout.

        If -1 is provided for one edge type on one layer, then all inbound edges
        of that edge type will be included.
    edge_dir : str, default ``'in'``
        Can be either ``'in' `` where the neighbors will be sampled according to
        incoming edges, or ``'out'`` otherwise, same as :func:`dgl.sampling.sample_neighbors`.
    prob : str, optional
        If given, the probability of each neighbor being sampled is proportional
        to the edge feature value with the given name in ``g.edata``.  The feature must be
        a scalar on each edge.
    mask : str, optional
        If given, a neighbor could be picked only if the edge mask with the given
        name in ``g.edata`` is True.  The data must be boolean on each edge.

        This argument is mutually exclusive with :attr:`prob`.  If you want to
        specify both a mask and a probability, consider multiplying the probability
        with the mask instead.
    replace : bool, default False
        Whether to sample with replacement
    prefetch_node_feats : list[str] or dict[ntype, list[str]], optional
        The source node data to prefetch for the first MFG, corresponding to the
        input node features necessary for the first GNN layer.
    prefetch_labels : list[str] or dict[ntype, list[str]], optional
        The destination node data to prefetch for the last MFG, corresponding to
        the node labels of the minibatch.
    prefetch_edge_feats : list[str] or dict[etype, list[str]], optional
        The edge data names to prefetch for all the MFGs, corresponding to the
        edge features necessary for all GNN layers.
    output_device : device, optional
        The device of the output subgraphs or MFGs.  Default is the same as the
        minibatch of seed nodes.
    reverse_edge_types_map: dict
        A dict of reverse edge type info.
    """
    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        reverse_edge_types_map=None,
    ):
        self.mask = mask
        self.reverse_edge_types_map = reverse_edge_types_map
        # original_edge_types_map is the reverse map of reverse_edge_types_map
        self.original_edge_types_map = {
            val: key for key, val in reverse_edge_types_map.items()
        } if reverse_edge_types_map is not None else None

        super().__init__(
            fanouts=fanouts,
            edge_dir=edge_dir,
            prob=prob,
            mask=None, # Do neighbor sampling with out edge mask
            replace=replace,
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device
        )

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """Generates a list of DGL MFGs from the given seed nodes.

        This function must return a triplet where the first element is the input node IDs
        for the first GNN layer (a tensor or a dict of tensors for heterogeneous graphs),
        the second element is the output node IDs for the last GNN layer, and the third
        element is the said list of blocks.

        Parameters
        ----------
        g: DGLGraph
            Graph to sample blocks.
        seed_nodes: dict of tensors
            Seed nodes.
        exclude_eids: func
            Operations to exlude eids.
        """
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            eid = {etype: frontier.edges[etype].data[EID] \
                   for etype in frontier.canonical_etypes}
            new_eid = dict(eid)
            if self.mask is not None:
                new_edges = {}
                for etype in frontier.canonical_etypes:
                    if self.mask in g.edges[etype].data:
                        # train mask in data
                        if etype in eid:
                            mask = g.edges[etype].data[self.mask][eid[etype]].bool()
                            new_edges[etype] = mask
                            new_eid[etype] = eid[etype][mask]

                    elif self.original_edge_types_map is not None and \
                        self.mask in g.edges[self.original_edge_types_map[etype]].data:
                        # handle rev-etype edges
                        # get etype from rev-etype.
                        original_etype = self.original_edge_types_map[etype]
                        rev_mask = g.edges[original_etype].data[self.mask][eid[etype]].bool()
                        new_edges[etype] = rev_mask
                        new_eid[etype] = eid[etype][rev_mask]
                    else:
                        # There is no train mask here
                        new_edges[etype] = th.full(eid[etype].shape, True, dtype=th.bool)
                new_frontier = frontier.edge_subgraph(new_edges, relabel_nodes=False)
            else:
                new_frontier = frontier
            block = to_block(new_frontier, seed_nodes)
            # When there is only one etype
            # we can not use block.edata[EID] = new_eid
            for etype in block.canonical_etypes:
                block.edges[etype].data[EID] = new_eid[etype]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

class FileSamplerInterface:
    r"""File Sampler Interface. This interface defines the
    # operation supported by a file sampler and the common
    # check and processing for dataset_path.

    Parameters:
    ---------
    dataset_path : str
        Path to the data files.
        Mutually exclusive with :attr:`files`.
    """

    def __init__(self, dataset_path):
        if not dataset_path:
            raise ValueError(
                "Dataset_path={}, should be specified.".format(
                    dataset_path
                )
            )
        self.dataset_path = dataset_path
        self.files = [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if os.path.isfile(os.path.join(dataset_path, f))
        ]
        self.files = [f for f in self.files if os.path.getsize(f)>0]
        if len(self.files) == 0:
            raise ValueError (
                f"no non-empty files found at top directory {dataset_path}.")
        self.files.sort()

        self.num_files = len(self.files)
        logging.info("found %s files from %s", self.num_files, dataset_path)

    def __len__(self):
        """ Return number of files in the sampler."""
        return self.num_files

    def __iter__(self):
        """ Return the iterator of the sampler."""
        raise NotImplementedError

    def __next__(self):
        """ Return next file name from the sampler iterator."""
        raise NotImplementedError

class SequentialFileSampler():
    r"""Sequential File Sampler. It samples files sequentially.

    Parameters:
    ----------
    file_indices : list of int
        File indices for a local trainer
    is_train : bool
        Set to ``True`` if it's training set.
    infinite : bool
        Set to ``True`` to make it infinite.
    """
    def __init__(self, file_indices, is_train=True, infinite=True):
        self.file_indices = file_indices
        self.num_local_files = len(self.file_indices)
        self._indices = list(range(self.num_local_files))
        self.is_train = is_train
        self.infinite = infinite

    def __iter__(self):
        self._index = -1
        return self

    def __next__(self):
        """ Get index for next file"""
        self._index += 1

        if self._index >= self.num_local_files:
            if self.is_train and self.infinite:
                self._index = 0
            else:
                # non-infinite sampler.
                # set _index to -1 for next evaluation.
                self._index = -1
                return None

        return self.file_indices[self._index]

class RandomShuffleFileSampler():
    r"""Random File Sampler. Has files reshuffled for sampling.

    Parameters:
    ----------
    file_indices : list of int
        File indices for a local trainer
    infinite : bool
        Set to ``True`` to make it infinite.
    """
    def __init__(self, file_indices, infinite=True):
        self.file_indices = file_indices
        self.num_local_files = len(self.file_indices)
        self._indices = list(range(self.num_local_files))
        self.infinite = infinite

    def __iter__(self):
        self._index = -1
        # shuffle the file indices
        random.shuffle(self._indices)
        return self

    def __next__(self):
        """ Get index for next file"""
        self._index += 1

        if self._index >= self.num_local_files:
            # make it infinite sampler
            random.shuffle(self._indices)
            self._index = 0
            if not self.infinite:
                return None

        return self.file_indices[self._indices[self._index]]

class DistributedFileSampler(FileSamplerInterface):
    r"""Distributed File Sampler. Samples file from the data path.
    Each rank only has access to a partition of all the data shards.

    Parameters:
    ---------
    dataset_path : str
        path to the data files.
        Mutually exclusive with :attr:`files`.
    shuffle : bool
        Set to ``True`` to have the files reshuffled
        at every epoch. Shuffling is performed on the files of the local partition.
    local_rank : int
        Local rank ID.
    world_size : int
        Number of all trainers.
    is_train : bool
        Set to ``True`` if it's training set.
    infinite : bool
        Set to ``True`` to make it infinite.
    """
    def __init__(
        self,
        dataset_path=None,
        shuffle=False,
        local_rank=-1,
        world_size=1,
        is_train=True,
        infinite=True,
    ):
        super().__init__(dataset_path)

        # Initialize distributed ranks
        self.local_rank = local_rank
        self.world_size = world_size

        # distribute file index
        self._file_index_distribute()

        if not is_train:
            self.part_len = self.num_files
        file_indices = list(range(self.part_len))
        if shuffle and is_train:
            sampler = RandomShuffleFileSampler(file_indices=file_indices, \
                infinite=infinite)
        else:
            sampler = SequentialFileSampler(file_indices=file_indices, \
                is_train=is_train, infinite=infinite)
        self.sampler = sampler
        self.sampler_iter = iter(sampler)

        self.shuffle = shuffle

    def _file_index_distribute(self):
        """
        Assign a slice window of file index to each worker.
        The slice window of each worker is specified
        by self.global_start and self.global_end
        """
        if self.world_size > self.num_files:
            # If num of workers is greater than num of files,
            # the slice windows are same across all workers,
            # which covers all files.
            self.remainder = self.world_size % self.num_files
            self.global_start = 0
            self.global_end = self.num_files
            self.part_len = self.global_end
        else:
            # If num of workers is smaller than num of files,
            # the slice windows are different for each worker.
            # In the case where the files cannot be evenly distributed,
            # the remainder will be assigned to one or multiple workers evenly.
            part_len = self.num_files // self.world_size
            self.remainder = self.num_files % self.world_size
            self.global_start = part_len * self.local_rank + min(self.local_rank, self.remainder)
            self.global_end = self.global_start + part_len + (self.local_rank < self.remainder)
            self.part_len = self.global_end - self.global_start

    def get_file(self, offset):
        """ Get the file name with corresponding index"""
        if self.world_size > self.num_files:
            # e.g, when self.world_size=8 and self.num_files=3:
            # local_rank=0|offset|file_index % self.num_files
            #             |0     |0
            #             |1     |1
            #             |2     |2
            #             |0     |0
            #             |1     |1
            #             |2     |2
            #             ...    ...
            # local_rank=1|offset|file_index % self.num_files
            #             |0     |1
            #             |1     |2
            #             |2     |0
            #             |0     |1
            #             |1     |2
            #             |2     |0
            #             ...    ...
            # local_rank=2|offset|file_index % self.num_files
            #             |0     |2
            #             |1     |0
            #             |2     |1
            #             |0     |2
            #             |1     |0
            #             |2     |1
            #             ...    ...
            # ...
            # local_rank=7|offset|file_index % self.num_files
            #             |0     |1
            #             |1     |2
            #             |2     |0
            #             |0     |1
            #             |1     |2
            #             |2     |0
            #             ...    ...
            file_index = (
                (offset * self.world_size) + self.local_rank + (self.remainder * offset)
            )
            file_index = self.global_start + file_index % self.part_len
        else:
            # e.g, when self.world_size=3 and self.num_files=7:
            # local_rank=0|offset|file_index % self.num_files
            #             |0     |0
            #             |1     |1
            #             |2     |2
            #             |0     |0
            #             |1     |1
            #             |2     |2
            #             ...    ...
            # local_rank=1|offset|file_index % self.num_files
            #             |0     |3
            #             |1     |4
            #             |0     |3
            #             |1     |4
            #             |0     |3
            #             |1     |4
            #             ...    ...
            # local_rank=2|offset|file_index % self.num_files
            #             |0     |5
            #             |1     |6
            #             |0     |5
            #             |1     |6
            #             |0     |5
            #             |1     |6
            #             ...    ...
            file_index = self.global_start + offset % self.part_len
        return self.files[file_index % self.num_files]

    def __len__(self):
        return self.part_len

    def __iter__(self):
        return self

    def __next__(self):
        """ Get the file name for next file from sampler"""
        ret = None
        offset = next(self.sampler_iter)

        if offset is not None:
            ret = self.get_file(offset)

        return ret
