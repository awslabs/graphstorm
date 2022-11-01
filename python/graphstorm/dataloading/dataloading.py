import math
import torch as th

import dgl
from dgl.dataloading import DistDataLoader
from dgl.dataloading import EdgeCollator
from dgl.dataloading.dist_dataloader import _remove_kwargs_dist

from .sampler import LocalUniform, JointUniform
from .utils import trim_data, modify_fanout_for_target_etype

################ Minibatch DataLoader (Edge Prediction) #######################

class GSgnnEdgePredictionDataLoader():
    """ Edge prediction minibatch dataloader
        The dataloader we use here will
    Argument
    --------
    g: DGLGraph
    train_dataset: GSgnnEdgePredictionDataLoader
        The dataset used in training. It must includes train_idxs.
    fanout: neighbor sample fanout
    n_layers: number of GNN layers
    batch_size: minibatch size
    reverse_edge_types_map: A map for reverse edge type
    exclude_training_targets: Whether to exclude training edges during neighbor sampling
    remove_target_edge: This boolean controls whether we will exclude all edges of the target during message passing.
    device : the device for which the sampling will be performed.
    """
    def __init__(self, g, train_dataset, fanout, n_layers, batch_size,
                 reverse_edge_types_map, remove_target_edge=True, exclude_training_targets=False, device='cpu'):
        assert len(fanout) == n_layers
        # set up dictionary fanout
        # remove the target edge type from computational graph

        target_etypes = train_dataset.train_etypes
        for e in target_etypes:
            if e in reverse_edge_types_map and reverse_edge_types_map[e] not in target_etypes:
                target_etypes.append(reverse_edge_types_map[e])
        if g.rank() == 0:
            if remove_target_edge:
                print("Target edge will be removed from message passing graph")
            else:
                print("Target edge will not be removed from message passing graph")

        if remove_target_edge:
            edge_fanout_lis = modify_fanout_for_target_etype(g=g, fanout=fanout, target_etypes=target_etypes)
        else:
            edge_fanout_lis = fanout

        self.dataloader = self._prepare_train_dataloader(g,
                                                         train_dataset.train_idxs,
                                                         edge_fanout_lis,
                                                         batch_size,
                                                         exclude_training_targets=exclude_training_targets,
                                                         reverse_edge_types_map=reverse_edge_types_map,
                                                         device='cpu')

    def _prepare_train_dataloader(self, g, train_idxs, fanout, batch_size, device='cpu',
                                  exclude_training_targets=False, reverse_edge_types_map=None):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)

        # edge loader
        loader = dgl.dataloading.DistEdgeDataLoader(g,
                                                    train_idxs,
                                                    sampler,
                                                    batch_size=batch_size,
                                                    device=device,
                                                    shuffle=True,
                                                    drop_last=False,
                                                    num_workers=0,
                                                    exclude='reverse_types' if exclude_training_targets else None,
                                                    reverse_etypes=reverse_edge_types_map
                                                    if exclude_training_targets else None)
        return loader

    def __iter__(self):
        return self.dataloader.__iter__()

    def __next__(self):
        return self.dataloader.__next__()


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
    g: DGLGraph
    train_dataset: GSgnnLinkPredictionTrainData
        The dataset used in training. It must includes train_idxs.
    fanout: neighbor sample fanout
    n_layers: number of GNN layers
    batch_size: minibatch size
    num_negative_edges: num of negative edges per positive edge
    device : the device trainer is running on.
    exclude_training_targets: Whether to exclude training edges during neighbor sampling
    reverse_edge_types_map: A map for reverse edge type
    """
    def __init__(self, g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device,
        exclude_training_targets=False, reverse_edge_types_map=None):
        assert len(fanout) == n_layers
        self.dataloader = self._prepare_train_dataloader(g,
                                                         train_dataset.train_idxs,
                                                         fanout,
                                                         num_negative_edges,
                                                         batch_size,
                                                         device,
                                                         exclude_training_targets,
                                                         reverse_edge_types_map)

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(num_negative_edges)
        return negative_sampler

    def _prepare_train_dataloader(self, g, train_idxs, fanout, num_negative_edges, batch_size, device,
                                  exclude_training_targets=False, reverse_edge_types_map=None):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        negative_sampler = self._prepare_negative_sampler(num_negative_edges)

        # edge loader
        if isinstance(train_idxs, dict):
            for etype in train_idxs:
                train_idxs[etype] = trim_data(train_idxs[etype], device)
        else:
            train_idxs = trim_data(train_idxs, device)
        # latest documentation https://docs.dgl.ai/generated/dgl.dataloading.as_edge_prediction_sampler.html?highlight=as_edge_prediction_sampler
        loader = dgl.dataloading.DistEdgeDataLoader(g,
                                                    train_idxs,
                                                    sampler,
                                                    batch_size=batch_size,
                                                    negative_sampler=negative_sampler,
                                                    device="cpu", # Only cpu is supported in DistGraph
                                                    shuffle=True,
                                                    drop_last=False,
                                                    num_workers=0,
                                                    exclude='reverse_types' if exclude_training_targets else None,
                                                    reverse_etypes=reverse_edge_types_map \
                                                        if exclude_training_targets else None)
        return loader

    def __iter__(self):
        return self.dataloader.__iter__()

    def __next__(self):
        return self.dataloader.__next__()

class GSgnnLPJointNegDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with joint negative sampler

    """
    def __init__(self, g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device,
        exclude_training_targets=False, reverse_edge_types_map=None):
        super(GSgnnLPJointNegDataLoader, self).__init__(
            g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device, exclude_training_targets, reverse_edge_types_map)

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = JointUniform(num_negative_edges)
        return negative_sampler

class GSgnnLPLocalUniformNegDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with local uniform negative sampler

    """
    def __init__(self, g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device,
        exclude_training_targets=False, reverse_edge_types_map=None):
        super(GSgnnLPLocalUniformNegDataLoader, self).__init__(
            g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device, exclude_training_targets, reverse_edge_types_map)

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = LocalUniform(num_negative_edges)
        return negative_sampler

######## Per etype sampler ########
import inspect

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
            for i in range(len(ret)):
                # ret[i] is (etype, eids)
                if ret[i][1] is None:
                    # if eids is None, randomly generate one more data point
                    new_ret.append((ret[i][0], self._rand_gen(ret[i][0]).item()))
                else:
                    for data in ret[i][1]:
                        new_ret.append((ret[i][0], data.item()))
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

    Parameters
    ----------
    g: DGLGraph
        Input graph
    train_dataset: GSgnnLinkPredictionTrainData
        The dataset used in training. It must includes train_idxs.
    fanout: list of int
        Neighbor sample fanout
    n_layers: int
        Number of GNN layers
    batch_size: int
        Minibatch size
    num_negative_edges: int
        Num of negative edges per positive edge
    device : device
        The device trainer is running on.
    exclude_training_targets:
        Whether to exclude training edges during neighbor sampling
    reverse_edge_types_map:
        A map for reverse edge type
    """
    def __init__(self, g, train_dataset, fanout, n_layers,
        batch_size, num_negative_edges, device,
        exclude_training_targets=False, reverse_edge_types_map=None):
        super(GSgnnAllEtypeLinkPredictionDataLoader, self).__init__(
            g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device, exclude_training_targets, reverse_edge_types_map)

    def _prepare_train_dataloader(self, g, train_idxs, fanout, num_negative_edges, batch_size, device,
                                  exclude_training_targets=False, reverse_edge_types_map=None):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        negative_sampler = self._prepare_negative_sampler(num_negative_edges)

        # edge loader
        if isinstance(train_idxs, dict):
            for etype in train_idxs:
                train_idxs[etype] = trim_data(train_idxs[etype], device)
        else:
            train_idxs = trim_data(train_idxs, device)
        # latest documentation https://docs.dgl.ai/generated/dgl.dataloading.as_edge_prediction_sampler.html?highlight=as_edge_prediction_sampler
        loader = AllEtypeDistEdgeDataLoader(g,
                                            train_idxs,
                                            sampler,
                                            batch_size=batch_size,
                                            negative_sampler=negative_sampler,
                                            device="cpu", # Only cpu is supported in DistGraph
                                            shuffle=True,
                                            drop_last=False,
                                            num_workers=0,
                                            exclude='reverse_types' if exclude_training_targets else None,
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
    def __init__(self, g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device,
        exclude_training_targets=False, reverse_edge_types_map=None):
        super(GSgnnAllEtypeLPJointNegDataLoader, self).__init__(
            g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device, exclude_training_targets, reverse_edge_types_map)

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = JointUniform(num_negative_edges)
        return negative_sampler

################ Minibatch DataLoader (Node classification) #######################

class GSgnnNodeDataLoader():
    """ Minibatch dataloader for node tasks

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    train_dataset: GSgnnNodeTrainData
        Training ataset
    fanout: list of int
        Neighbor sample fanout
    n_layers: int
        Number of GNN layers
    batch_size: int
        Batch size
    device: torch.device
        the device trainer is running on.
    """
    def __init__(self, g, train_dataset, fanout, n_layers, batch_size, device):
        assert len(fanout) == n_layers
        self.dataloader = self._prepare_train_dataloader(g,
                                                         train_dataset.predict_ntype,
                                                         train_dataset.train_idx,
                                                         fanout,
                                                         batch_size,
                                                         device)

    def _prepare_train_dataloader(self, g, predict_ntype, train_idx, fanout, batch_size, device):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        train_idx = trim_data(train_idx, device)
        loader = dgl.dataloading.DistNodeDataLoader(g, {predict_ntype: train_idx}, sampler,
            batch_size=batch_size, shuffle=True, num_workers=0)

        return loader

    def __iter__(self):
        return self.dataloader.__iter__()

    def __next__(self):
        return self.dataloader.__next__()
