"""various dataloaders for the GSF
"""
import math
import inspect
import torch as th

import dgl
from dgl.dataloading import DistDataLoader
from dgl.dataloading import EdgeCollator
from dgl.dataloading.dist_dataloader import _remove_kwargs_dist

from .sampler import LocalUniform, JointUniform
from .utils import trim_data, modify_fanout_for_target_etype

################ Minibatch DataLoader (Edge Prediction) #######################

class GSgnnEdgeDataLoader():
    """ The minibatch dataloader for edge prediction

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
    """
    def __init__(self, dataset, target_idx, fanout, batch_size, device='cpu',
                 train_task=True, reverse_edge_types_map=None,
                 remove_target_edge_type=True,
                 exclude_training_targets=False):
        self._data = dataset
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
        self.dataloader = self._prepare_dataloader(dataset.g,
                                                   target_idx,
                                                   edge_fanout_lis,
                                                   batch_size,
                                                   exclude_training_targets,
                                                   reverse_edge_types_map,
                                                   train_task=train_task)

    def _prepare_dataloader(self, g, target_idxs, fanout, batch_size,
                            exclude_training_targets=False, reverse_edge_types_map=None,
                            train_task=True):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        # edge loader
        exclude_val = 'reverse_types' if exclude_training_targets else None
        loader = dgl.dataloading.DistEdgeDataLoader(g,
                                                    target_idxs,
                                                    sampler,
                                                    batch_size=batch_size,
                                                    shuffle=train_task,
                                                    drop_last=False,
                                                    num_workers=0,
                                                    exclude=exclude_val,
                                                    reverse_etypes=reverse_edge_types_map
                                                    if exclude_training_targets else None)
        return loader

    def __iter__(self):
        return self.dataloader.__iter__()

    def __next__(self):
        return self.dataloader.__next__()

    @property
    def data(self):
        """ The dataset of this dataloader.
        """
        return self._data


################ Minibatch DataLoader (Link Prediction) #######################

BUILTIN_LP_UNIFORM_NEG_SAMPLER = 'uniform'
BUILTIN_LP_JOINT_NEG_SAMPLER = 'joint'
BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER = 'localuniform'
BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER = 'all_etype_uniform'
BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER = 'all_etype_joint'

class GSgnnLinkPredictionDataLoader():
    """ Link prediction minibatch dataloader

    The negative edges are sampled uniformly.

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
    """
    def __init__(self, dataset, target_idx, fanout, batch_size, num_negative_edges, device='cpu',
                 train_task=True, reverse_edge_types_map=None, exclude_training_targets=False,
                 edge_mask_for_gnn_embeddings='train_mask'):
        self._data = dataset
        for etype in target_idx:
            assert etype in dataset.g.canonical_etypes, \
                    "edge type {} does not exist in the graph".format(etype)

        self.dataloader = self._prepare_dataloader(dataset.g, target_idx, fanout,
                num_negative_edges, batch_size, device,
                train_task=train_task,
                exclude_training_targets=exclude_training_targets,
                reverse_edge_types_map=reverse_edge_types_map,
                edge_mask_for_gnn_embeddings=edge_mask_for_gnn_embeddings)

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(num_negative_edges)
        return negative_sampler

    def _prepare_dataloader(self, g, target_idxs, fanout,
                            num_negative_edges, batch_size, device, train_task=True,
                            exclude_training_targets=False, reverse_edge_types_map=None,
                            edge_mask_for_gnn_embeddings=None):
        # The dataloader can only sample neighbors from the training graph.
        # This can avoid information leak during the link prediction training.
        # This avoids two types of information leak: it avoids sampling neighbors
        # from the test graph during the training; it also avoid sampling neighbors
        # from the test graph to generate embeddings for evaluating the model performance
        # on the test set.
        if edge_mask_for_gnn_embeddings is not None and \
                any(edge_mask_for_gnn_embeddings in g.edges[etype].data for etype in g.etypes):
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout,
                                                                mask=edge_mask_for_gnn_embeddings)
        else:
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        negative_sampler = self._prepare_negative_sampler(num_negative_edges)

        # edge loader
        if isinstance(target_idxs, dict):
            for etype in target_idxs:
                target_idxs[etype] = trim_data(target_idxs[etype], device)
        else:
            target_idxs = trim_data(target_idxs, device)
        exclude = 'reverse_types' if exclude_training_targets else None
        reverse_etypes = reverse_edge_types_map if exclude_training_targets else None
        loader = dgl.dataloading.DistEdgeDataLoader(g,
                                                    target_idxs,
                                                    sampler,
                                                    batch_size=batch_size,
                                                    negative_sampler=negative_sampler,
                                                    shuffle=train_task,
                                                    drop_last=False,
                                                    num_workers=0,
                                                    exclude=exclude,
                                                    reverse_etypes=reverse_etypes)
        return loader

    def __iter__(self):
        return self.dataloader.__iter__()

    def __next__(self):
        return self.dataloader.__next__()

    @property
    def data(self):
        """ The dataset of this dataloader.
        """
        return self._data

class GSgnnLPJointNegDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with joint negative sampler

    """

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = JointUniform(num_negative_edges)
        return negative_sampler

class GSgnnLPLocalUniformNegDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with local uniform negative sampler

    """

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = LocalUniform(num_negative_edges)
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

    def _prepare_dataloader(self, g, target_idxs, fanout, num_negative_edges,
                            batch_size, device, train_task=True,
                            exclude_training_targets=False,
                            reverse_edge_types_map=None,
                            edge_mask_for_gnn_embeddings=None):
        # See the comment in GSgnnLinkPredictionDataLoader
        if edge_mask_for_gnn_embeddings is not None and \
                any(edge_mask_for_gnn_embeddings in g.edges[etype].data for etype in g.etypes):
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout,
                                                                mask=edge_mask_for_gnn_embeddings)
        else:
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        negative_sampler = self._prepare_negative_sampler(num_negative_edges)

        # edge loader
        if isinstance(target_idxs, dict):
            for etype in target_idxs:
                target_idxs[etype] = trim_data(target_idxs[etype], device)
        else:
            target_idxs = trim_data(target_idxs, device)
        exclude_val = 'reverse_types' if exclude_training_targets else None
        loader = AllEtypeDistEdgeDataLoader(g,
                                            target_idxs,
                                            sampler,
                                            batch_size=batch_size,
                                            negative_sampler=negative_sampler,
                                            shuffle=train_task,
                                            drop_last=False,
                                            num_workers=0,
                                            exclude=exclude_val,
                                            reverse_etypes=reverse_edge_types_map \
                                                if exclude_training_targets else None)
        return loader

    def __iter__(self):
        return self.dataloader.__iter__()

    def __next__(self):
        return self.dataloader.__next__()

class GSgnnAllEtypeLPJointNegDataLoader(GSgnnAllEtypeLinkPredictionDataLoader):
    """ Link prediction dataloader with joint negative sampler.
        In each minibatch, at least one edge is sampled from each etype.

    """

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = JointUniform(num_negative_edges)
        return negative_sampler

################ Minibatch DataLoader (Node classification) #######################

class GSgnnNodeDataLoader():
    """ Minibatch dataloader for node tasks

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
    """
    def __init__(self, dataset, target_idx, fanout, batch_size, device, train_task=True):
        self._data = dataset
        assert isinstance(target_idx, dict)
        for ntype in target_idx:
            assert ntype in dataset.g.ntypes, \
                    "node type {} does not exist in the graph".format(ntype)
        self.dataloader = self._prepare_dataloader(dataset.g,
                                                   target_idx,
                                                   fanout,
                                                   batch_size,
                                                   train_task,
                                                   device)

    def _prepare_dataloader(self, g, target_idx, fanout, batch_size, train_task, device):
        for ntype in target_idx:
            target_idx[ntype] = trim_data(target_idx[ntype], device)
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        loader = dgl.dataloading.DistNodeDataLoader(g, target_idx, sampler,
            batch_size=batch_size, shuffle=train_task, num_workers=0)

        return loader

    def __iter__(self):
        return self.dataloader.__iter__()

    def __next__(self):
        return self.dataloader.__next__()

    @property
    def data(self):
        """ The dataset of this dataloader.
        """
        return self._data
