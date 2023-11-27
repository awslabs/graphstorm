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

    Various dataloaders for the GSF
"""
import math
import inspect
import logging
import dgl
import torch as th
from torch.utils.data import DataLoader
import torch.distributed as dist

from dgl.dataloading import DistDataLoader
from dgl.dataloading import EdgeCollator
from dgl.dataloading.dist_dataloader import _remove_kwargs_dist

from .sampler import (LocalUniform,
                      JointUniform,
                      GlobalUniform,
                      JointLocalUniform,
                      InbatchJointUniform,
                      FastMultiLayerNeighborSampler,
                      DistributedFileSampler)
from .utils import trim_data, modify_fanout_for_target_etype
from .dataset import GSDistillData

################ Minibatch DataLoader (Edge Prediction) #######################

class _ReconstructedNeighborSampler():
    """ This samples an additional hop for a mini-batch.

    The additional hop is used to compute the features of the nodes in the input layer.
    Users can specify which input nodes requires to construct node features.

    Parameters
    ----------
    dataset: GSgnnData
        The GraphStorm dataset
    constructed_embed_ntypes : a list of strings.
        The node type in the input layer that requires to construct node features.
    fanout : int
        The fanout for the additional layer.
    """
    def __init__(self, dataset, constructed_embed_ntypes, fanout):
        assert fanout > 0 or fanout == -1, \
                "Constructing features requires to sample neighbors or -1 " + \
                "if we use all neighbors."
        self._g = dataset.g
        etypes = self._g.canonical_etypes
        self._subg_etypes = []
        target_ntypes = set()
        for dst_ntype in constructed_embed_ntypes:
            for etype in etypes:
                if etype[2] == dst_ntype and (dataset.has_node_feats(etype[0]) \
                        or dataset.has_node_lm_feats(etype[0])):
                    self._subg_etypes.append(etype)
                    target_ntypes.add(dst_ntype)
        remain_ntypes = set(constructed_embed_ntypes) - target_ntypes
        # We need to make sure all node types that require feature construction
        # can be constructed.
        assert len(remain_ntypes) == 0, \
                f"The features of some node types cannot be constructed: {remain_ntypes}"
        self._fanout = {}
        for etype in etypes:
            self._fanout[etype] = fanout if etype in self._subg_etypes else 0
        assert len(self._subg_etypes) > 0, "The sampled edge types is empty."

    def sample(self, seeds):
        """ Sample an additional hop for the input block.

        Parameters
        ----------
        seeds : dict of Tensors
            The seed nodes for the input layer.

        Returns
        -------
        DGLBlock : an additional hop for computing the features of the input nodes.
        """
        subg = self._g.sample_neighbors(seeds, self._fanout)
        block = dgl.to_block(subg, seeds)
        input_nodes = {ntype: block.srcnodes[ntype].data[dgl.NID] for ntype in block.srctypes}
        return block, input_nodes

class GSgnnEdgeDataLoaderBase():
    """ The base dataloader class for edge tasks.

    If users want to customize the dataloader for edge prediction tasks
    they should extend this base class by implementing the special methods
    `__iter__` and `__next__`.

    Parameters
    ----------
    dataset : GSgnnEdgeData
        The dataset for the edge task.
    target_idx : dict of Tensors
        The target edge IDs.
    fanout : list or dict of lists
        The fanout for each GNN layer.
    """
    def __init__(self, dataset, target_idx, fanout):
        self._data = dataset
        self._target_eidx = target_idx
        self._fanout = fanout

    def __iter__(self):
        """ Returns an iterator object
        """

    def __next__(self):
        """ Return a mini-batch data for the edge task.

        A mini-batch comprises three objects: the input node IDs,
        the target edges and the subgraph blocks for message passing.

        Returns
        -------
        dict of Tensors : the input node IDs of the mini-batch.
        DGLGraph : the target edges.
        list of DGLGraph : the subgraph blocks for message passing.
        """

    def __len__(self):
        """ Return the length (number of mini-batches) of the data loader

        Returns
        int: length
        """

    @property
    def data(self):
        """ The dataset of this dataloader.

        Returns
        -------
        GSgnnEdgeData : The dataset of the dataloader.
        """
        return self._data

    @property
    def target_eidx(self):
        """ Target edge idx for prediction

        Returns
        -------
        dict of Tensors : the target edge IDs.
        """
        return self._target_eidx

    @property
    def fanout(self):
        """ The fan out of each GNN layers

        Returns
        -------
        list or a dict of list : the fanouts for each GNN layer.
        """
        return self._fanout

class MultiLayerNeighborSamplerForReconstruct(dgl.dataloading.BlockSampler):
    """ A wrapper of MultiLayerNeighborSampler

    This is a wrapper on MultiLayerNeighborSampler to sample additional neighbors
    for feature construction.

    Parameters
    ----------
    sampler : MultiLayerNeighborSampler
        A sampler to sample multi-hop neighbors.
    dataset: GSgnnData
        The GraphStorm dataset
    construct_feat_ntype : list of str
        The node types that requires to construct node features.
    construct_feat_fanout : int
        The fanout required to construct node features.
    """
    def __init__(self, sampler, dataset, construct_feat_ntype, construct_feat_fanout):
        super().__init__()
        self._sampler = sampler
        self._construct_feat_sampler = _ReconstructedNeighborSampler(
                dataset, construct_feat_ntype, construct_feat_fanout)

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """ Sample blocks for message passing.

        Parameters
        ----------
        g : DistGraph
            The distributed graph.
        seed_nodes : dict of Tensors
            The seed nodes

        Returns
        -------
        dict of Tensors : the input node IDs.
        dict of Tensors : the seed node IDs.
        list of DGLBlock : the blocks for message passing.
        """
        input_nodes, seed_nodes, blocks = \
                self._sampler.sample_blocks(g, seed_nodes, exclude_eids)
        if len(blocks) > 0:
            block, input_nodes = self._construct_feat_sampler.sample(input_nodes)
            blocks.insert(0, block)
        return input_nodes, seed_nodes, blocks

class GSgnnEdgeDataLoader(GSgnnEdgeDataLoaderBase):
    """ The minibatch dataloader for edge prediction

    GSgnnEdgeDataLoader samples GraphStorm edge dataset into an iterable over mini-batches
    of samples. Both source and destination nodes are included in the batch_graph, which
    will be used by GraphStorm Trainers and Inferrers.

    Parameters
    ------------
    dataset: GSgnnEdgeData
        The GraphStorm edge dataset
    target_idx : dict of Tensors
        The target edges for prediction
    fanout: list of int or dict of list
        Neighbor sample fanout. If it's a dict, it indicates the fanout for each edge type.
    batch_size: int
        Batch size
    device: torch.device
        the device trainer is running on.
    train_task : bool
        Whether or not for training.
    reverse_edge_types_map: dict
        A map for reverse edge type
    exclude_training_targets: bool
        Whether to exclude training edges during neighbor sampling
    remove_target_edge_type: bool
        Whether we will exclude all edges of the target edge type in message passing.
    construct_feat_ntype : list of str
        The node types that requires to construct node features.
    construct_feat_fanout : int
        The fanout required to construct node features.

    Examples
    ------------
    To train a 2-layer GNN for edge prediction on a set of edges ``target_idx`` on
    a graph where each nodes takes messages from 15 neighbors on the first layer
    and 10 neighbors on the second.

    .. code:: python

        from graphstorm.dataloading import GSgnnEdgeTrainData
        from graphstorm.dataloading import GSgnnEdgeDataLoader
        from graphstorm.trainer import GSgnnEdgePredictionTrainer

        ep_data = GSgnnEdgeTrainData(...)
        ep_dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, fanout=[15, 10], batch_size=128)
        ep_trainer = GSgnnEdgePredictionTrainer(...)
        ep_trainer.fit(ep_dataloader, num_epochs=10)
    """
    def __init__(self, dataset, target_idx, fanout, batch_size, device='cpu',
                 train_task=True, reverse_edge_types_map=None,
                 remove_target_edge_type=True,
                 exclude_training_targets=False,
                 construct_feat_ntype=None,
                 construct_feat_fanout=5):
        super().__init__(dataset, target_idx, fanout)
        self._device = device
        if remove_target_edge_type:
            assert reverse_edge_types_map is not None, \
                    "To remove target etype, the reversed etype should be provided."
            # We need to duplicate this etype list.
            target_etypes = list(target_idx.keys())
            for e in target_etypes:
                if e in reverse_edge_types_map and reverse_edge_types_map[e] not in target_etypes:
                    target_etypes.append(reverse_edge_types_map[e])
            edge_fanout_lis = modify_fanout_for_target_etype(g=dataset.g,
                                                             fanout=fanout,
                                                             target_etypes=target_etypes)
        else:
            edge_fanout_lis = fanout

        for etype in target_idx:
            assert etype in dataset.g.canonical_etypes, \
                    "edge type {} does not exist in the graph".format(etype)
        self.dataloader = self._prepare_dataloader(dataset,
                                                   target_idx,
                                                   edge_fanout_lis,
                                                   batch_size,
                                                   exclude_training_targets,
                                                   reverse_edge_types_map,
                                                   train_task=train_task,
                                                   construct_feat_ntype=construct_feat_ntype,
                                                   construct_feat_fanout=construct_feat_fanout)

    def _prepare_dataloader(self, dataset, target_idxs, fanout, batch_size,
                            exclude_training_targets=False, reverse_edge_types_map=None,
                            train_task=True, construct_feat_ntype=None, construct_feat_fanout=5):
        g = dataset.g
        if construct_feat_ntype is None:
            construct_feat_ntype = []
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        if len(construct_feat_ntype) > 0:
            sampler = MultiLayerNeighborSamplerForReconstruct(sampler,
                    dataset, construct_feat_ntype, construct_feat_fanout)
        # edge loader
        exclude_val = 'reverse_types' if exclude_training_targets else None
        loader = dgl.dataloading.DistEdgeDataLoader(g,
                                                    target_idxs,
                                                    sampler,
                                                    batch_size=batch_size,
                                                    shuffle=train_task,
                                                    drop_last=False,
                                                    exclude=exclude_val,
                                                    reverse_etypes=reverse_edge_types_map
                                                    if exclude_training_targets else None)
        return loader

    def __iter__(self):
        self.dataloader.__iter__()
        return self

    def __next__(self):
        return self.dataloader.__next__()

    def __len__(self):
        # Follow
        # https://github.com/dmlc/dgl/blob/1.0.x/python/dgl/distributed/dist_dataloader.py#L116
        # In DGL, DistDataLoader.expected_idxs is the length (number of batches)
        # of the datalaoder.
        return self.dataloader.expected_idxs

    @property
    def data(self):
        """ The dataset of this dataloader.
        """
        return self._data

    @property
    def target_eidx(self):
        """ Target edge idx for prediction
        """
        return self._target_eidx

    @property
    def fanout(self):
        """ The fan out of each GNN layers
        """
        return self._fanout

################ Minibatch DataLoader (Link Prediction) #######################

BUILTIN_LP_UNIFORM_NEG_SAMPLER = 'uniform'
BUILTIN_LP_JOINT_NEG_SAMPLER = 'joint'
BUILTIN_LP_INBATCH_JOINT_NEG_SAMPLER = 'inbatch_joint'
BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER = 'localuniform'
BUILTIN_LP_LOCALJOINT_NEG_SAMPLER = 'localjoint'
BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER = 'all_etype_uniform'
BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER = 'all_etype_joint'
BUILTIN_FAST_LP_UNIFORM_NEG_SAMPLER = 'fast_uniform'
BUILTIN_FAST_LP_JOINT_NEG_SAMPLER = 'fast_joint'
BUILTIN_FAST_LP_LOCALUNIFORM_NEG_SAMPLER = 'fast_localuniform'
BUILTIN_FAST_LP_LOCALJOINT_NEG_SAMPLER = 'fast_localjoint'

class GSgnnLinkPredictionDataLoaderBase():
    """ The base class of link prediction dataloader.

    If users want to customize the dataloader for link prediction tasks
    they should extend this base class by implementing the special methods
    `__iter__` and `__next__`.

    Parameters
    ----------
    dataset: GSgnnEdgeData
        The GraphStorm edge dataset
    target_idx : dict of Tensors
        The target edges for prediction
    fanout: list of int or dict of list
        Neighbor sample fanout. If it's a dict, it indicates the fanout for each edge type.
    """
    def __init__(self, dataset, target_idx, fanout):
        self._dataset = dataset
        self._target_idx = target_idx
        self._fanout = fanout

    def __iter__(self):
        """ Returns an iterator object
        """

    def __next__(self):
        """ Return a mini-batch for link prediction.

        A mini-batch of link prediction contains four objects:
        * the input node IDs of the mini-batch,
        * the target positive edges for prediction,
        * the negative edges for prediction,
        * the subgraph blocks for message passing.

        Returns
        -------
        Tensor or dict of Tensors : the input nodes of a mini-batch.
        DGLGraph : positive edges.
        DGLGraph : negative edges.
        list of DGLGraph : subgraph blocks for message passing.
        """

    def __len__(self):
        """ Return the length (number of mini-batches) of the data loader

        Returns
        int: length
        """

    @property
    def data(self):
        """ The dataset of this dataloader.

        Returns
        -------
        GSgnnEdgeData : The dataset of the dataloader.
        """
        return self._dataset

    @property
    def fanout(self):
        """ The fan out of each GNN layers

        Returns
        -------
        list or a dict of list : the fanouts for each GNN layer.
        """
        return self._fanout

    @property
    def target_eidx(self):
        """ The target edges for prediction.

        Returns
        -------
        dict of Tensors : the target edge IDs.
        """
        return self._target_idx

class GSgnnLinkPredictionDataLoader(GSgnnLinkPredictionDataLoaderBase):
    """ Link prediction minibatch dataloader

    GSgnnLinkPredictionDataLoader samples GraphStorm edge dataset into an iterable over mini-batches
    of samples. In each batch, pos_graph and neg_graph are sampled subgraph for positive and
    negative edges, which will be used by GraphStorm Trainers and Inferrers. Given a positive edge,
    a negative edge is composed of the source node and a random negative destination nodes
    according to a uniform distribution.

    Argument
    --------
    dataset: GSgnnEdgeData
        The GraphStorm edge dataset
    target_idx : dict of Tensors
        The target edges for prediction
    fanout: list of int or dict of list
        Neighbor sample fanout. If it's a dict, it indicates the fanout for each edge type.
    batch_size: int
        Batch size
    num_negative_edges: int
        The number of negative edges per positive edge
    device: torch.device
        the device trainer is running on.
    train_task : bool
        Whether or not for training.
    reverse_edge_types_map: dict
        A map for reverse edge type
    exclude_training_targets: bool
        Whether to exclude training edges during neighbor sampling
    edge_mask_for_gnn_embeddings : str
        The mask that indicates the edges used for computing GNN embeddings. By default,
        the dataloader uses the edges in the training graphs to compute GNN embeddings to
        avoid information leak for link prediction.
    construct_feat_ntype : list of str
        The node types that requires to construct node features.
    construct_feat_fanout : int
        The fanout required to construct node features.

    Examples
    ------------
    To train a 2-layer GNN for link prediction on a set of positive edges ``target_idx`` on
    a graph where each nodes takes messages from 15 neighbors on the first layer
    and 10 neighbors on the second. We use 10 negative edges per positive in this example.

    .. code:: python

        from graphstorm.dataloading import GSgnnEdgeTrainData
        from graphstorm.dataloading import GSgnnLinkPredictionDataLoader
        from graphstorm.trainer import GSgnnLinkPredictionTrainer

        lp_data = GSgnnEdgeTrainData(...)
        lp_dataloader = GSgnnLinkPredictionDataLoader(lp_data, target_idx, fanout=[15, 10],
                                                    num_negative_edges=10, batch_size=128)
        lp_trainer = GSgnnLinkPredictionTrainer(...)
        lp_trainer.fit(lp_dataloader, num_epochs=10)
    """
    def __init__(self, dataset, target_idx, fanout, batch_size, num_negative_edges, device='cpu',
                 train_task=True, reverse_edge_types_map=None, exclude_training_targets=False,
                 edge_mask_for_gnn_embeddings='train_mask',
                 construct_feat_ntype=None, construct_feat_fanout=5):
        super().__init__(dataset, target_idx, fanout)
        self._device = device
        for etype in target_idx:
            assert etype in dataset.g.canonical_etypes, \
                    "edge type {} does not exist in the graph".format(etype)

        self.dataloader = self._prepare_dataloader(dataset, target_idx, fanout,
                num_negative_edges, batch_size, device,
                train_task=train_task,
                exclude_training_targets=exclude_training_targets,
                reverse_edge_types_map=reverse_edge_types_map,
                edge_mask_for_gnn_embeddings=edge_mask_for_gnn_embeddings,
                construct_feat_ntype=construct_feat_ntype,
                construct_feat_fanout=construct_feat_fanout)

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(num_negative_edges)
        return negative_sampler

    def _prepare_dataloader(self, dataset, target_idxs, fanout,
                            num_negative_edges, batch_size, device, train_task=True,
                            exclude_training_targets=False, reverse_edge_types_map=None,
                            edge_mask_for_gnn_embeddings=None, construct_feat_ntype=None,
                            construct_feat_fanout=5):
        g = dataset.g
        if construct_feat_ntype is None:
            construct_feat_ntype = []
        # The dataloader can only sample neighbors from the training graph.
        # This can avoid information leak during the link prediction training.
        # This avoids two types of information leak: it avoids sampling neighbors
        # from the test graph during the training; it also avoid sampling neighbors
        # from the test graph to generate embeddings for evaluating the model performance
        # on the test set.
        if edge_mask_for_gnn_embeddings is not None and \
                any(edge_mask_for_gnn_embeddings in g.edges[etype].data
                    for etype in g.canonical_etypes):
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout,
                                                                mask=edge_mask_for_gnn_embeddings)
        else:
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        if len(construct_feat_ntype) > 0:
            sampler = MultiLayerNeighborSamplerForReconstruct(sampler,
                    dataset, construct_feat_ntype, construct_feat_fanout)
        negative_sampler = self._prepare_negative_sampler(num_negative_edges)

        # edge loader
        if train_task:
            if isinstance(target_idxs, dict):
                for etype in target_idxs:
                    target_idxs[etype] = trim_data(target_idxs[etype], device)
            else:
                target_idxs = trim_data(target_idxs, device)
        # for validation and test, there is no need to trim data

        exclude = 'reverse_types' if exclude_training_targets else None
        reverse_etypes = reverse_edge_types_map if exclude_training_targets else None
        loader = dgl.dataloading.DistEdgeDataLoader(g,
                                                    target_idxs,
                                                    sampler,
                                                    batch_size=batch_size,
                                                    negative_sampler=negative_sampler,
                                                    shuffle=train_task,
                                                    drop_last=False,
                                                    exclude=exclude,
                                                    reverse_etypes=reverse_etypes)
        return loader

    def __iter__(self):
        self.dataloader.__iter__()
        return self

    def __next__(self):
        return self.dataloader.__next__()

    def __len__(self):
        # Follow
        # https://github.com/dmlc/dgl/blob/1.0.x/python/dgl/distributed/dist_dataloader.py#L116
        # In DGL, DistDataLoader.expected_idxs is the length (number of batches)
        # of the datalaoder.
        return self.dataloader.expected_idxs

class GSgnnLPJointNegDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with joint negative sampler

    """

    def _prepare_negative_sampler(self, num_negative_edges):
        # The negative sampler is the joint uniform negative sampler.
        negative_sampler = JointUniform(num_negative_edges)
        return negative_sampler

class GSgnnLPInBatchJointNegDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with in-batch and joint negative sampler

    """

    def _prepare_negative_sampler(self, num_negative_edges):
        # The negative sampler is the in-batch joint uniform negative sampler.
        negative_sampler = InbatchJointUniform(num_negative_edges)
        return negative_sampler

class GSgnnLPLocalUniformNegDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with local uniform negative sampler

    """

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = LocalUniform(num_negative_edges)
        return negative_sampler

class GSgnnLPLocalJointNegDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with local joint negative sampler

    """

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = JointLocalUniform(num_negative_edges)
        return negative_sampler

class FastGSgnnLinkPredictionDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader that does not send train_mask to
        DGL sampler but use the train_mask to trim the sampled graph.
    """

    def _prepare_dataloader(self, dataset, target_idxs, fanout,
                            num_negative_edges, batch_size, device, train_task=True,
                            exclude_training_targets=False, reverse_edge_types_map=None,
                            edge_mask_for_gnn_embeddings=None, construct_feat_ntype=None,
                            construct_feat_fanout=5):
        g = dataset.g
        if construct_feat_ntype is None:
            construct_feat_ntype = []
        # The dataloader can only sample neighbors from the training graph.
        # This can avoid information leak during the link prediction training.
        # This avoids two types of information leak: it avoids sampling neighbors
        # from the test graph during the training; it also avoid sampling neighbors
        # from the test graph to generate embeddings for evaluating the model performance
        # on the test set.
        if edge_mask_for_gnn_embeddings is not None and \
                any(edge_mask_for_gnn_embeddings in g.edges[etype].data
                    for etype in g.canonical_etypes):
            sampler = FastMultiLayerNeighborSampler(fanout,
                                                    mask=edge_mask_for_gnn_embeddings)
        else:
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        if len(construct_feat_ntype) > 0:
            sampler = MultiLayerNeighborSamplerForReconstruct(sampler,
                    dataset, construct_feat_ntype, construct_feat_fanout)
        negative_sampler = self._prepare_negative_sampler(num_negative_edges)

        # edge loader
        if train_task:
            if isinstance(target_idxs, dict):
                for etype in target_idxs:
                    target_idxs[etype] = trim_data(target_idxs[etype], device)
            else:
                target_idxs = trim_data(target_idxs, device)
        # for validation and test, there is no need to trim data

        exclude = 'reverse_types' if exclude_training_targets else None
        reverse_etypes = reverse_edge_types_map if exclude_training_targets else None
        loader = dgl.dataloading.DistEdgeDataLoader(g,
                                                    target_idxs,
                                                    sampler,
                                                    batch_size=batch_size,
                                                    negative_sampler=negative_sampler,
                                                    shuffle=train_task,
                                                    drop_last=False,
                                                    exclude=exclude,
                                                    reverse_etypes=reverse_etypes)
        return loader

class FastGSgnnLPJointNegDataLoader(FastGSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with joint negative sampler

    """

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = JointUniform(num_negative_edges)
        return negative_sampler

class FastGSgnnLPLocalUniformNegDataLoader(FastGSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with local uniform negative sampler

    """

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = LocalUniform(num_negative_edges)
        return negative_sampler

class FastGSgnnLPLocalJointNegDataLoader(FastGSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with local joint negative sampler

    """

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = JointLocalUniform(num_negative_edges)
        return negative_sampler

######## Per etype sampler ########

class AllEtypeDistEdgeDataLoader(DistDataLoader):
    """ Distributed edge data sampler that samples at least one
        edge for each edge type in a minibatch

        Parameters
        ----------
        g: DistGraph
            Input graph
        eids: dict
            Target edge ids
        graph_sampler:
            Graph neighbor sampler
        device: str:
            Device
        kwargs: list
            Other arguments
    """
    def __init__(self, g, eids, graph_sampler, device=None, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        _collator_arglist = inspect.getfullargspec(EdgeCollator).args
        for k, v in kwargs.items():
            if k in _collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v

        if device is None:
            # for the distributed case default to the CPU
            device = 'cpu'
        assert device == 'cpu', 'Only cpu is supported in the case of a DistGraph.'
        self.collator = EdgeCollator(g, eids, graph_sampler, **collator_kwargs)
        _remove_kwargs_dist(dataloader_kwargs)
        super().__init__(eids,
                         collate_fn=self.collator.collate,
                         **dataloader_kwargs)

        self._reinit_dataset()
        self.device = device

    def _reinit_dataset(self):
        """ Reinitialize the dataset

            Here we record the edge ids of different edge types separately.
            When doing sampling we will sample edges per edge type.
        """
        batch_size = self.batch_size
        data_idx = {}
        total_edges = 0
        for key, val in self.dataset.items():
            data_idx[key] = th.arange(0, len(val))
            total_edges += len(val)
        self.data_idx = data_idx

        max_expected_idxs = 0
        bs_per_type = {}
        for etype, idxs in self.data_idx.items():
            # compute the number of edges to be sampled for
            # each edge type in a minibatch.
            # If batch_size * num_edges / total_edges < 0, then set 1.
            #
            # Note: The resulting batch size of a mini batch may be larger
            #       than the batch size set by a user.
            bs = math.ceil(batch_size * len(idxs) / total_edges)
            bs_per_type[etype] = bs
            exp_idxs = len(idxs) // bs
            if not self.drop_last and len(idxs) % bs != 0:
                exp_idxs += 1

            # Get the maximum expected idx across all edge types.
            # The epoch size is decided by max_expected_idxs
            max_expected_idxs = max(max_expected_idxs, exp_idxs)
        self.bs_per_type = bs_per_type
        self.expected_idxs = max_expected_idxs

        self.current_pos = {etype:0 for etype, _ in self.data_idx.items()}

    def __iter__(self):
        if self.shuffle:
            self.data_idx = {etype: th.randperm(len(idxs)) \
                for etype, idxs in self.data_idx.items()}
        self.current_pos = {etype:0 for etype, _ in self.data_idx.items()}
        self.recv_idxs = 0
        self.num_pending = 0
        return self

    def _next_data_etype(self, etype):
        """ Get postive edges for the next iteration for a specific edge type

            Return a tensor of eids. If return None, we will randomly
            generate an eid for the etype if the dataloader does not
            reach the end of epoch.
        """
        if self.current_pos[etype] == len(self.dataset[etype]):
            return None

        end_pos = 0
        if self.current_pos[etype] + self.bs_per_type[etype] > len(self.dataset[etype]):
            if self.drop_last:
                return None
            else:
                end_pos = len(self.dataset[etype])
        else:
            end_pos = self.current_pos[etype] + self.bs_per_type[etype]
        idx = self.data_idx[etype][self.current_pos[etype]:end_pos].tolist()
        ret = self.dataset[etype][idx]

        # Sharing large number of tensors between processes will consume too many
        # file descriptors, so let's convert each tensor to scalar value beforehand.
        self.current_pos[etype] = end_pos

        return ret

    def _rand_gen(self, etype):
        """ Randomly select one edge for a specific edge type
        """
        return self.dataset[etype][th.randint(len(self.dataset[etype]), (1,))]

    def _next_data(self):
        """ Get postive edges for the next iteration
        """
        end = True
        ret = []
        for etype, _ in self.dataset.items():
            next_data = self._next_data_etype(etype)
            # Only if all etypes reach end of iter,
            # the current iter is done
            end = (next_data is None) & end
            ret.append((etype, next_data))

        if end:
            return None
        else:
            new_ret = []
            # for i in range(len(ret)):
            for rel_t in ret:
                # rel_t is (etype, eids)
                if rel_t[1] is None:
                    # if eids is None, randomly generate one more data point
                    new_ret.append((rel_t[0], self._rand_gen(rel_t[0]).item()))
                else:
                    for data in rel_t[1]:
                        new_ret.append((rel_t[0], data.item()))
            return new_ret

class GSgnnAllEtypeLinkPredictionDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction minibatch dataloader. In each minibatch,
        at least one edge is sampled from each etype.

        Note: using this dataloader with a graph with massive etypes
        may cause memory issue, as the batch_size will be implicitly
        increased.

        The negative edges are sampled uniformly.

        Note: The resulting batch size of a mini batch may be larger
              than the batch size set by a user.

    """

    def _prepare_dataloader(self, dataset, target_idxs, fanout, num_negative_edges,
                            batch_size, device, train_task=True,
                            exclude_training_targets=False,
                            reverse_edge_types_map=None,
                            edge_mask_for_gnn_embeddings=None,
                            construct_feat_ntype=None,
                            construct_feat_fanout=5):
        g = dataset.g
        if construct_feat_ntype is None:
            construct_feat_ntype = []
        # See the comment in GSgnnLinkPredictionDataLoader
        if edge_mask_for_gnn_embeddings is not None and \
                any(edge_mask_for_gnn_embeddings in g.edges[etype].data
                    for etype in g.canonical_etypes):
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout,
                                                                mask=edge_mask_for_gnn_embeddings)
        else:
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        if len(construct_feat_ntype) > 0:
            sampler = MultiLayerNeighborSamplerForReconstruct(sampler,
                    dataset, construct_feat_ntype, construct_feat_fanout)
        negative_sampler = self._prepare_negative_sampler(num_negative_edges)

        # edge loader
        if train_task:
            if isinstance(target_idxs, dict):
                for etype in target_idxs:
                    target_idxs[etype] = trim_data(target_idxs[etype], device)
            else:
                target_idxs = trim_data(target_idxs, device)
        # for validation and test, there is no need to trim data

        exclude_val = 'reverse_types' if exclude_training_targets else None
        loader = AllEtypeDistEdgeDataLoader(g,
                                            target_idxs,
                                            sampler,
                                            batch_size=batch_size,
                                            negative_sampler=negative_sampler,
                                            shuffle=train_task,
                                            drop_last=False,
                                            exclude=exclude_val,
                                            reverse_etypes=reverse_edge_types_map \
                                                if exclude_training_targets else None)
        return loader

    def __iter__(self):
        self.dataloader.__iter__()
        return self

    def __next__(self):
        return self.dataloader.__next__()

    def __len__(self):
        # Follow
        # https://github.com/dmlc/dgl/blob/1.0.x/python/dgl/distributed/dist_dataloader.py#L116
        # In DGL, DistDataLoader.expected_idxs is the length (number of batches)
        # of the datalaoder.
        # AllEtypeDistEdgeDataLoader is a child class of DistDataLoader.
        return self.dataloader.expected_idxs

class GSgnnAllEtypeLPJointNegDataLoader(GSgnnAllEtypeLinkPredictionDataLoader):
    """ Link prediction dataloader with joint negative sampler.
        In each minibatch, at least one edge is sampled from each etype.

    """

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = JointUniform(num_negative_edges)
        return negative_sampler

class GSgnnLinkPredictionTestDataLoader():
    """ Link prediction minibatch dataloader for validation and test.
    In order to efficiently compute positive and negative scores for
    link prediction tasks, GSgnnLinkPredictionTestDataLoader is designed
    to only generates edges, i.e., (src, dst) pairs.

    The negative edges are sampled uniformly.

    Parameters
    -----------
    dataset: GSgnnEdgeData
        The GraphStorm edge dataset
    target_idx : dict of Tensors
        The target edges for prediction
    batch_size: int
        Batch size
    num_negative_edges: int
        The number of negative edges per positive edge
    fanout: int
        Evaluation fanout for computing node embedding
    fixed_test_size: int
        Fixed number of test data used in evaluation.
        If it is none, use the whole testset.
        When test is huge, using fixed_test_size
        can save validation and test time.
        Default: None.
    """
    def __init__(self, dataset, target_idx, batch_size, num_negative_edges, fanout=None,
    fixed_test_size=None):
        self._data = dataset
        self._fanout = fanout
        for etype in target_idx:
            assert etype in dataset.g.canonical_etypes, \
                    "edge type {} does not exist in the graph".format(etype)
        self._batch_size = batch_size
        self._target_idx = target_idx
        self._fixed_test_size = {}
        for etype, t_idx in target_idx.items():
            self._fixed_test_size[etype] = fixed_test_size \
                if fixed_test_size is not None else len(t_idx)
            if self._fixed_test_size[etype] > len(t_idx):
                logging.warning("The size of the test set of etype %s" \
                                "is %d, which is smaller than the expected"
                                "test size %d, force it to %d",
                                etype, len(t_idx), self._fixed_test_size[etype], len(t_idx))
        self._negative_sampler = self._prepare_negative_sampler(num_negative_edges)
        self._reinit_dataset()

    def _reinit_dataset(self):
        """ Reinitialize the dataset
        """
        self._current_pos = {etype:0 for etype, _ in self._target_idx.items()}
        self.remaining_etypes = list(self._target_idx.keys())
        for etype, t_idx in self._target_idx.items():
            # If the expected test size is smaller than the size of test set
            # shuffle the test ids
            if self._fixed_test_size[etype] < len(t_idx):
                self._target_idx[etype] = self._target_idx[etype][th.randperm(len(t_idx))]

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        self._neg_sample_type = BUILTIN_LP_UNIFORM_NEG_SAMPLER
        negative_sampler = GlobalUniform(num_negative_edges)
        return negative_sampler

    def __iter__(self):
        self._reinit_dataset()
        return self

    def _next_data(self, etype):
        """ Get postive edges for the next iteration for a specific edge type
        """
        g = self._data.g
        current_pos = self._current_pos[etype]
        end_of_etype = current_pos + self._batch_size >= self._fixed_test_size[etype]

        pos_eids = self._target_idx[etype][current_pos:self._fixed_test_size[etype]] \
            if end_of_etype \
            else self._target_idx[etype][current_pos:current_pos+self._batch_size]
        pos_pairs = g.find_edges(pos_eids, etype=etype)
        pos_neg_tuple = self._negative_sampler.gen_neg_pairs(g, {etype:pos_pairs})
        self._current_pos[etype] += self._batch_size
        return pos_neg_tuple, end_of_etype

    def __next__(self):
        if len(self.remaining_etypes) == 0:
            raise StopIteration

        curr_etype = self.remaining_etypes[0]
        cur_iter, end_of_etype = self._next_data(curr_etype)
        if end_of_etype:
            self.remaining_etypes.pop(0)

        # return pos, neg pairs
        return cur_iter, self._neg_sample_type

    @property
    def fanout(self):
        """ Get eval fanout
        """
        return self._fanout

class GSgnnLinkPredictionJointTestDataLoader(GSgnnLinkPredictionTestDataLoader):
    """ Link prediction minibatch dataloader for validation and test
        with joint negative sampler
    """

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = JointUniform(num_negative_edges)
        self._neg_sample_type = BUILTIN_LP_JOINT_NEG_SAMPLER
        return negative_sampler

################ Minibatch DataLoader (Node classification) #######################

class GSgnnNodeDataLoaderBase():
    """ The base dataloader class for node tasks.

    If users want to customize the dataloader for node prediction tasks
    they should extend this base class by implementing the special methods
    `__iter__` and `__next__`.

    Parameters
    ----------
    dataset : GSgnnNodeData
        The dataset for the node task.
    target_idx : dict of Tensors
        The target node IDs.
    fanout : list or dict of lists
        The fanout for each GNN layer.
    """
    def __init__(self, dataset, target_idx, fanout):
        self._data = dataset
        self._target_idx = target_idx
        self._fanout = fanout

    def __iter__(self):
        """ Returns an iterator object
        """

    def __next__(self):
        """ Return a mini-batch data for the node task.

        A mini-batch comprises three objects: the input node IDs of the mini-batch,
        the target nodes and the subgraph blocks for message passing.

        Returns
        -------
        dict of Tensors : the input node IDs of the mini-batch.
        dict of Tensors : the target node IDs.
        list of DGLGraph : the subgraph blocks for message passing.
        """

    def __len__(self):
        """ Return the length (number of mini-batches) of the data loader

        Returns
        int: length
        """

    @property
    def data(self):
        """ The dataset of this dataloader.

        Returns
        -------
        GSgnnNodeData : The dataset of the dataloader.
        """
        return self._data

    @property
    def target_nidx(self):
        """ Target edge idx for prediction

        Returns
        -------
        dict of Tensors : the target edge IDs.
        """
        return self._target_idx

    @property
    def fanout(self):
        """ The fan out of each GNN layers

        Returns
        -------
        list or a dict of list : the fanouts for each GNN layer.
        """
        return self._fanout

class GSgnnNodeDataLoader(GSgnnNodeDataLoaderBase):
    """ Minibatch dataloader for node tasks

    GSgnnNodeDataLoader samples GraphStorm node dataset into an iterable over mini-batches of
    samples including target nodes and sampled neighbor nodes, which will be used by GraphStorm
    Trainers and Inferrers.

    Parameters
    ----------
    dataset: GSgnnNodeData
        The GraphStorm dataset
    target_idx : dict of Tensors
        The target nodes for prediction
    fanout: list of int or dict of list
        Neighbor sample fanout. If it's a dict, it indicates the fanout for each edge type.
    batch_size: int
        Batch size
    device: torch.device
        the device trainer is running on.
    train_task : bool
        Whether or not for training.
    construct_feat_ntype : list of str
        The node types that requires to construct node features.
    construct_feat_fanout : int
        The fanout required to construct node features.

    Examples
    ----------
    To train a 2-layer GNN for node classification on a set of nodes ``target_idx`` on
    a graph where each nodes takes messages from 15 neighbors on the first layer
    and 10 neighbors on the second.

    .. code:: python

        from graphstorm.dataloading import GSgnnNodeTrainData
        from graphstorm.dataloading import GSgnnNodeDataLoader
        from graphstorm.trainer import GSgnnNodePredictionTrainer

        np_data = GSgnnNodeTrainData(...)
        np_dataloader = GSgnnNodeDataLoader(np_data, target_idx, fanout=[15, 10], batch_size=128)
        np_trainer = GSgnnNodePredictionTrainer(...)
        np_trainer.fit(np_dataloader, num_epochs=10)
    """
    def __init__(self, dataset, target_idx, fanout, batch_size, device, train_task=True,
                 construct_feat_ntype=None, construct_feat_fanout=5):
        super().__init__(dataset, target_idx, fanout)
        assert isinstance(target_idx, dict)
        for ntype in target_idx:
            assert ntype in dataset.g.ntypes, \
                    "node type {} does not exist in the graph".format(ntype)
        self.dataloader = self._prepare_dataloader(dataset,
                                                   target_idx,
                                                   fanout,
                                                   batch_size,
                                                   train_task,
                                                   device,
                                                   construct_feat_ntype=construct_feat_ntype,
                                                   construct_feat_fanout=construct_feat_fanout)

    def _prepare_dataloader(self, dataset, target_idx, fanout, batch_size,
            train_task, device, construct_feat_ntype=None, construct_feat_fanout=5):
        g = dataset.g
        if construct_feat_ntype is None:
            construct_feat_ntype = []
        if train_task:
            for ntype in target_idx:
                target_idx[ntype] = trim_data(target_idx[ntype], device)
        # for validation and test, there is no need to trim data

        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        if len(construct_feat_ntype) > 0:
            sampler = MultiLayerNeighborSamplerForReconstruct(sampler,
                    dataset, construct_feat_ntype, construct_feat_fanout)
        loader = dgl.dataloading.DistNodeDataLoader(g, target_idx, sampler,
            batch_size=batch_size, shuffle=train_task)

        return loader

    def __iter__(self):
        self.dataloader = iter(self.dataloader)
        return self

    def __next__(self):
        return self.dataloader.__next__()

    def __len__(self):
        # Follow
        # https://github.com/dmlc/dgl/blob/1.0.x/python/dgl/distributed/dist_dataloader.py#L116
        # In DGL, DistDataLoader.expected_idxs is the length (number of batches)
        # of the datalaoder.
        return self.dataloader.expected_idxs

class GSgnnNodeSemiSupDataLoader(GSgnnNodeDataLoader):
    """ Semisupervised Minibatch dataloader for node tasks

    Parameters
    ----------
    dataset: GSgnnNodeData
        The GraphStorm dataset
    target_idx : dict of Tensors
        The target nodes for prediction
    unlabeled_idx : dict of Tensors
        The unlabeled nodes for semi-supervised training
    fanout: list of int or dict of list
        Neighbor sample fanout. If it's a dict, it indicates the fanout for each edge type.
    batch_size: int
        Batch size, the sum of labeled and unlabeled nodes
    device: torch.device
        the device trainer is running on.
    train_task : bool
        Whether or not for training.
    """
    def __init__(self, dataset, target_idx, unlabeled_idx, fanout, batch_size, device,
                 train_task=True, construct_feat_ntype=None, construct_feat_fanout=5):
        super().__init__(dataset, target_idx, fanout, batch_size // 2, device,
                         train_task=train_task, construct_feat_ntype=construct_feat_ntype,
                         construct_feat_fanout=construct_feat_fanout)
        # loader for unlabeled nodes:
        self.unlabeled_dataloader = self._prepare_dataloader(dataset,
                                                   unlabeled_idx,
                                                   fanout,
                                                   batch_size // 2,
                                                   train_task,
                                                   device,
                                                   construct_feat_ntype=construct_feat_ntype,
                                                   construct_feat_fanout=construct_feat_fanout)

    def __iter__(self):
        return zip(self.dataloader.__iter__(), self.unlabeled_dataloader.__iter__())

    def __next__(self):
        return self.dataloader.__next__(), self.unlabeled_dataloader.__next__()

    def __len__(self):
        # Follow
        # https://github.com/dmlc/dgl/blob/1.0.x/python/dgl/distributed/dist_dataloader.py#L116
        # In DGL, DistDataLoader.expected_idxs is the length (number of batches)
        # of the datalaoder.
        # As it uses two dataloader, either one throws
        # an End of Iter error will stop the dataloader.
        return min(self.dataloader.expected_idxs,
                   self.unlabeled_dataloader.expected_idxs)

####################### Distillation #############################

class DistillDataManager:
    r"""Distill Data Manager. Combines a file sampler and a dataloader generator,
    and streamingly provides an iterable over a set of files.

    Parameters:
    ----------
    dataloader_generator : DataloaderGenerator:
        A dataloader generator
        Generates dataloader based on given a file path.
    dataset_path str :
        Path to the data files.
    shuffle : bool
        Set to ``True`` to have the files reshuffled at every epoch (default: ``False``).
    local_rank : int
        Local rank for the trainer (default: ``-1``).
    world_size : int
        Number of all trainers.
    is_train : bool
        Set to ``True`` if the provider is for training set (default: ``True``).
    """

    def __init__(
        self,
        dataloader_generator,
        dataset_path,
        shuffle=False,
        local_rank=-1,
        world_size=1,
        is_train=True,
    ):
        # TODO (HZ): Implement prefetch_iterator function
        # to use zero-copy shared memory (e.g., /dev/shm)
        # for reducing the costs of serialization and deserialization.
        # Make sure that each data shard is
        # not too large so that workers_per_node * #DataProvider *
        # num_prefetches * data_shard_size < shm_size.

        self.is_train = is_train
        assert dataset_path is not None, "dataset_path needs to be specified."
        if not isinstance(dataset_path, str):
            raise TypeError(
                f"dataset_path should be a str, but got {type(dataset_path)} "
                f"dataset_path={dataset_path}"
            )
        file_sampler = DistributedFileSampler(
            dataset_path=dataset_path,
            shuffle=shuffle,
            local_rank=local_rank,
            world_size=world_size,
            is_train=is_train,
        )

        self.file_sampler = file_sampler
        self.file_sampler_iter = iter(self.file_sampler)
        self.dataloader_generator = dataloader_generator
        self.dataloader = None
        self.data_file = None

        logging.info("DataProvider - Initialization: num_files = %s", len(file_sampler))

    def get_iterator_name(self):
        """ Return shard name.
        """
        return self.data_file

    def refresh_manager(self):
        """ refresh manager."""
        self.file_sampler.sampler._index = -1

    def get_iterator(self):
        """ Get dataloader iterator for a data file.
        """
        dataloader = None
        data_file = next(self.file_sampler_iter)

        # TODO (HZ): implement a queue to store and pop the files by prefetch and get_iterator
        if data_file is not None:
            dataloader = self.dataloader_generator.generate(data_file, is_train=self.is_train)
        self.data_file = data_file
        self.dataloader = dataloader
        return dataloader

    def __len__(self):
        return len(self.file_sampler)

    def __next__(self):
        return self.get_iterator()

    def __iter__(self):
        return self

    def release_iterator(self):
        """ Release the dataloader iterator.
        """
        self.dataloader = None
        self.data_file = None

class DistillDataloaderGenerator:
    r""" Distill Data Generator that generates pytorch dataloader based on the given file.

    Parameters:
    ----------
    tokenizer : transformers.AutoTokenizer
        HuggingFace Tokenizer.
    max_seq_len : int
        Maximum sequence length.
    device : str
        Device name.
    batch_size : int
        How many samples per batch to load.
    collate_fn : func
        Function to merge a list of samples to form a
           mini-batch of Tensor(s)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        device,
        batch_size=1,
        collate_fn=None,
    ):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device
        self.collate_fn = collate_fn

    def _data_len_sync(self, data):
        r""" Drop additional samples to make sure each dataloader
        has the same number of batches. This is to avoid the training
        stuck in distributed mode if any trainer has more or less batches.

        Parameters:
        ----------
        data : GSDistillData
            The pytorch dataset for a trainer.

        Returns:
        -------
        GSDistillData : The pytorch dataset for a trainer after the sync.
        """
        num_data = th.tensor(len(data), dtype=th.int64, device=self.device)
        dist.all_reduce(num_data, op=dist.ReduceOp.MIN)
        min_size = num_data.item()
        select_index = th.randperm(len(data))[:min_size].tolist()
        data.token_id_inputs = [data.token_id_inputs[i] for i in select_index]
        data.labels = data.labels[select_index]
        return data

    def generate(self, input_files, is_train=True):
        """ Generate dataloader given input files"

        Parameters:
        ----------
        input_files : list of str
            list of input files

        Returns:
        -------
        torch.DataLoader : dataloaders for training
        int : Number of samples in the dataloader
        """
        if not isinstance(input_files, (list, tuple)):
            input_files = [input_files]

        data = GSDistillData(input_files, self.tokenizer, self.max_seq_len, self.device)

        # do all_reduce here:
        if is_train:
            data = self._data_len_sync(data)

        collate_fn = data.get_collate_fn() if self.collate_fn is None else self.collate_fn
        dataloader = DataLoader(
            data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return dataloader
