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
import torch as th

import dgl
from dgl.dataloading import DistDataLoader
from dgl.dataloading import EdgeCollator
from dgl.dataloading.dist_dataloader import _remove_kwargs_dist

from .sampler import (LocalUniform,
                      JointUniform,
                      GlobalUniform,
                      JointLocalUniform, FastMultiLayerNeighborSampler)
from .utils import trim_data, modify_fanout_for_target_etype

################ Minibatch DataLoader (Edge Prediction) #######################
EP_DECODER_EDGE_FEAT = "ep_edge_feat"

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
                if etype[2] == dst_ntype and dataset.has_node_feats(etype[0]):
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
    construct_feat_ntype : list of str
        The node types that requires to construct node features.
    construct_feat_fanout : int
        The fanout required to construct node features.
    """
    def __init__(self, dataset, target_idx, fanout, batch_size, device='cpu',
                 train_task=True, reverse_edge_types_map=None,
                 remove_target_edge_type=True,
                 exclude_training_targets=False,
                 decoder_edge_feat=None,
                 construct_feat_ntype=None,
                 construct_feat_fanout=5):
        self._data = dataset
        self._device = device
        self._fanout = fanout
        self._target_eidx = target_idx
        self._decoder_edge_feat = decoder_edge_feat
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
        if construct_feat_ntype is None:
            construct_feat_ntype = []
        self._construct_feat_sampler = \
                _ReconstructedNeighborSampler(dataset, construct_feat_ntype,
                        construct_feat_fanout) if len(construct_feat_ntype) > 0 else None
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
        self.dataloader.__iter__()
        return self

    def __next__(self):
        input_nodes, batch_graph, blocks = self.dataloader.__next__()
        if self._construct_feat_sampler is not None and len(blocks) > 0:
            block, input_nodes = self._construct_feat_sampler.sample(input_nodes)
            blocks.insert(0, block)
        if self._decoder_edge_feat is not None:
            input_edges = {etype: batch_graph.edges[etype].data[dgl.EID] \
                           for etype in batch_graph.canonical_etypes}
            edge_feats = self._data.get_edge_feats(input_edges,
                                                   self._decoder_edge_feat,
                                                   batch_graph.device)
            # store edge feature into graph
            for etype, feat in edge_feats.items():
                batch_graph.edges[etype].data[EP_DECODER_EDGE_FEAT] = feat.to(th.float32)
        return (input_nodes, batch_graph, blocks)

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
BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER = 'localuniform'
BUILTIN_LP_LOCALJOINT_NEG_SAMPLER = 'localjoint'
BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER = 'all_etype_uniform'
BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER = 'all_etype_joint'
BUILTIN_FAST_LP_UNIFORM_NEG_SAMPLER = 'fast_uniform'
BUILTIN_FAST_LP_JOINT_NEG_SAMPLER = 'fast_joint'
BUILTIN_FAST_LP_LOCALUNIFORM_NEG_SAMPLER = 'fast_localuniform'
BUILTIN_FAST_LP_LOCALJOINT_NEG_SAMPLER = 'fast_localjoint'

LP_DECODER_EDGE_WEIGHT = "lp_edge_weight"

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
    lp_edge_weight_for_loss: str or dict of [str]
        The edge data fields that stores the edge weights used
        in computing link prediction loss
    construct_feat_ntype : list of str
        The node types that requires to construct node features.
    construct_feat_fanout : int
        The fanout required to construct node features.
    """
    def __init__(self, dataset, target_idx, fanout, batch_size, num_negative_edges, device='cpu',
                 train_task=True, reverse_edge_types_map=None, exclude_training_targets=False,
                 edge_mask_for_gnn_embeddings='train_mask', lp_edge_weight_for_loss=None,
                 construct_feat_ntype=None, construct_feat_fanout=5):
        self._data = dataset
        self._fanout = fanout
        self._lp_edge_weight_for_loss = lp_edge_weight_for_loss
        self._device = device
        for etype in target_idx:
            assert etype in dataset.g.canonical_etypes, \
                    "edge type {} does not exist in the graph".format(etype)

        if construct_feat_ntype is None:
            construct_feat_ntype = []
        self._construct_feat_sampler = \
                _ReconstructedNeighborSampler(dataset, construct_feat_ntype,
                        construct_feat_fanout) if len(construct_feat_ntype) > 0 else None
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
                any(edge_mask_for_gnn_embeddings in g.edges[etype].data
                    for etype in g.canonical_etypes):
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout,
                                                                mask=edge_mask_for_gnn_embeddings)
        else:
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
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
                                                    num_workers=0,
                                                    exclude=exclude,
                                                    reverse_etypes=reverse_etypes)
        return loader

    def __iter__(self):
        self.dataloader.__iter__()
        return self

    def __next__(self):
        input_nodes, pos_graph, neg_graph, blocks = self.dataloader.__next__()
        if self._construct_feat_sampler is not None and len(blocks) > 0:
            block, input_nodes = self._construct_feat_sampler.sample(input_nodes)
            blocks.insert(0, block)
        if self._lp_edge_weight_for_loss is not None:
            input_edges = {etype: pos_graph.edges[etype].data[dgl.EID] \
                for etype in pos_graph.canonical_etypes}
            edge_weight_feats = self._data.get_edge_feats(input_edges,
                                                          self._lp_edge_weight_for_loss,
                                                          pos_graph.device)
            # store edge feature into graph
            for etype, feat in edge_weight_feats.items():
                pos_graph.edges[etype].data[LP_DECODER_EDGE_WEIGHT] = feat
        return (input_nodes, pos_graph, neg_graph, blocks)

    @property
    def data(self):
        """ The dataset of this dataloader.
        """
        return self._data

    @property
    def fanout(self):
        """ The fan out of each GNN layers
        """
        return self._fanout

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
                any(edge_mask_for_gnn_embeddings in g.edges[etype].data
                    for etype in g.canonical_etypes):
            sampler = FastMultiLayerNeighborSampler(fanout,
                                                    mask=edge_mask_for_gnn_embeddings)
        else:
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
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
                                                    num_workers=0,
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

    def _prepare_dataloader(self, g, target_idxs, fanout, num_negative_edges,
                            batch_size, device, train_task=True,
                            exclude_training_targets=False,
                            reverse_edge_types_map=None,
                            edge_mask_for_gnn_embeddings=None):
        # See the comment in GSgnnLinkPredictionDataLoader
        if edge_mask_for_gnn_embeddings is not None and \
                any(edge_mask_for_gnn_embeddings in g.edges[etype].data
                    for etype in g.canonical_etypes):
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout,
                                                                mask=edge_mask_for_gnn_embeddings)
        else:
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
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
                                            num_workers=0,
                                            exclude=exclude_val,
                                            reverse_etypes=reverse_edge_types_map \
                                                if exclude_training_targets else None)
        return loader

    def __iter__(self):
        self.dataloader.__iter__()
        return self

    def __next__(self):
        input_nodes, pos_graph, neg_graph, blocks = self.dataloader.__next__()
        if self._lp_edge_weight_for_loss is not None:
            input_edges = {etype: pos_graph.edges[etype].data[dgl.EID] \
                for etype in pos_graph.canonical_etypes}
            edge_weight_feats = self._data.get_edge_feats(input_edges,
                                                          self._lp_edge_weight_for_loss,
                                                          pos_graph.device)
            # store edge feature into graph
            for etype, feat in edge_weight_feats.items():
                pos_graph.edges[etype].data[LP_DECODER_EDGE_WEIGHT] = feat
        return (input_nodes, pos_graph, neg_graph, blocks)

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

    Argument
    --------
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
    """
    def __init__(self, dataset, target_idx, batch_size, num_negative_edges, fanout=None):
        self._data = dataset
        self._fanout = fanout
        for etype in target_idx:
            assert etype in dataset.g.canonical_etypes, \
                    "edge type {} does not exist in the graph".format(etype)
        self._batch_size = batch_size
        self._target_idx = target_idx
        self._negative_sampler = self._prepare_negative_sampler(num_negative_edges)
        self._reinit_dataset()

    def _reinit_dataset(self):
        """ Reinitialize the dataset
        """
        self._current_pos = {etype:0 for etype, _ in self._target_idx.items()}
        self.remaining_etypes = list(self._target_idx.keys())

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
        end_of_etype = current_pos + self._batch_size >= len(self._target_idx[etype])
        pos_eids = self._target_idx[etype][current_pos:] if end_of_etype \
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
    construct_feat_ntype : list of str
        The node types that requires to construct node features.
    construct_feat_fanout : int
        The fanout required to construct node features.
    """
    def __init__(self, dataset, target_idx, fanout, batch_size, device, train_task=True,
                 construct_feat_ntype=None, construct_feat_fanout=5):
        self._data = dataset
        self._fanout = fanout
        self._target_nidx  = target_idx
        if construct_feat_ntype is None:
            construct_feat_ntype = []
        self._construct_feat_sampler = \
                _ReconstructedNeighborSampler(dataset, construct_feat_ntype,
                        construct_feat_fanout) if len(construct_feat_ntype) > 0 else None
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
        if train_task:
            for ntype in target_idx:
                target_idx[ntype] = trim_data(target_idx[ntype], device)
        # for validation and test, there is no need to trim data

        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        loader = dgl.dataloading.DistNodeDataLoader(g, target_idx, sampler,
            batch_size=batch_size, shuffle=train_task, num_workers=0)

        return loader

    def __iter__(self):
        self.dataloader = iter(self.dataloader)
        return self

    def __next__(self):
        input_nodes, seeds, blocks = self.dataloader.__next__()
        if self._construct_feat_sampler is not None and len(blocks) > 0:
            block, input_nodes = self._construct_feat_sampler.sample(input_nodes)
            blocks.insert(0, block)
        return input_nodes, seeds, blocks

    @property
    def data(self):
        """ The dataset of this dataloader.
        """
        return self._data

    @property
    def target_nidx(self):
        """ The target node ids for prediction.
        """
        return self._target_nidx

    @property
    def fanout(self):
        """ The fan out of each GNN layers
        """
        return self._fanout

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
                 train_task=True):
        super().__init__(dataset, target_idx, fanout, batch_size // 2, device,
                         train_task=train_task)
        # loader for unlabeled nodes:
        self.unlabeled_dataloader = self._prepare_dataloader(dataset.g,
                                                   unlabeled_idx,
                                                   fanout,
                                                   batch_size // 2,
                                                   train_task,
                                                   device)

    def __iter__(self):
        return zip(self.dataloader.__iter__(), self.unlabeled_dataloader.__iter__())

    def __next__(self):
        return self.dataloader.__next__(), self.unlabeled_dataloader.__next__()
