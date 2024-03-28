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

    Various datasets for the GSF
"""
import os
import abc
import logging
import re

import torch as th
import dgl
from dgl.distributed.constants import DEFAULT_NTYPE, DEFAULT_ETYPE
from torch.utils.data import Dataset
import pandas as pd

import  graphstorm as gs
from ..utils import get_rank, get_world_size, is_distributed, barrier, is_wholegraph
from ..utils import sys_tracker
from .utils import dist_sum, flip_node_mask

from ..wholegraph import is_wholegraph_embedding

def split_full_edge_list(g, etype, rank):
    ''' Split the full edge list of a graph.
    '''
    # TODO(zhengda) we need to split the edges to co-locate data and computation.
    # We assume that the number of edges is larger than the number of processes.
    # This should always be true unless a user's training set is extremely small.
    assert g.num_edges(etype) >= get_world_size()
    start = g.num_edges(etype) // get_world_size() * rank
    end = g.num_edges(etype) // get_world_size() * (rank + 1)
    return th.arange(start, end)

def prepare_batch_input(g, input_nodes,
                        dev='cpu', feat_field='feat'):
    """ Prepare minibatch input features

    Note: The output is stored in dev.

    Parameters
    ----------
    g: DGLGraph
        The graph.
    input_nodes: dict of tensor
        Input nodes.
    dev: th.device
        Device to put output in.
    feat_field: str or dict of list of str
        Fields to extract features

    Return:
    -------
    Dict of tensors.
        If a node type has features, it will get node features.
    """
    feat = {}
    for ntype, nid in input_nodes.items():
        feat_name = None if feat_field is None else \
            [feat_field] if isinstance(feat_field, str) \
            else feat_field[ntype] if ntype in feat_field else None

        if feat_name is not None:
            # concatenate multiple features together
            feats = []
            for fname in feat_name:
                data = g.nodes[ntype].data[fname]
                if is_wholegraph_embedding(data):
                    data = data.gather(nid.to(dev))
                else:
                    data = data[nid].to(dev)
                feats.append(data)
            feat[ntype] = th.cat(feats, dim=1)
    return feat

def prepare_batch_edge_input(g, input_edges,
                             dev='cpu', feat_field='feat'):
    """ Prepare minibatch edge input features

    Note: The output is stored in dev.

    Parameters
    ----------
    g: DGLGraph
        The graph.
    input_edges: dict of tensor
        Input edges.
    dev: th.device
        Device to put output in.
    feat_field: str or dict of list of str
        Fields to extract features

    Return:
    -------
    Dict of tensors.
        If a node type has features, it will get node features.
    """
    feat = {}
    for etypes, eid in input_edges.items():
        feat_name = None if feat_field is None else \
            [feat_field] if isinstance(feat_field, str) \
            else feat_field[etypes] if etypes in feat_field else None

        if feat_name is not None:
            # concatenate multiple features together
            feats = []
            for fname in feat_name:
                data = g.edges[etypes].data[fname]
                if is_wholegraph_embedding(data):
                    data = data.gather(eid.to(dev))
                else:
                    data = data[eid].to(dev)
                feats.append(data)
            feat[etypes] = th.cat(feats, dim=1)
    return feat

class GSgnnData():
    """ The GraphStorm data

    Parameters
    ----------
    graph_name : str
        The graph name
    part_config : str
        The path of the partition configuration file.
    node_feat_field: str or dict of list of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of list of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    decoder_edge_feat: str or dict of list of str
        Edge features used by decoder
    lm_feat_ntypes : list of str
        The node types that contains text features.
    lm_feat_etypes : list of tuples
        The edge types that contains text features.
    """

    def __init__(self, graph_name, part_config, node_feat_field, edge_feat_field,
                 decoder_edge_feat=None, lm_feat_ntypes=None, lm_feat_etypes=None):
        self._g = dgl.distributed.DistGraph(graph_name, part_config=part_config)
        self._node_feat_field = node_feat_field
        self._edge_feat_field = edge_feat_field
        self._lm_feat_ntypes = lm_feat_ntypes if lm_feat_ntypes is not None else []
        self._lm_feat_etypes = lm_feat_etypes if lm_feat_etypes is not None else []

        self._train_idxs = {}
        self._val_idxs = {}
        self._test_idxs = {}

        if get_rank() == 0:
            g = self._g
            for ntype in g.ntypes:
                logging.debug("%s has %d nodes.", ntype, g.number_of_nodes(ntype))
            for etype in g.canonical_etypes:
                logging.debug("%s has %d edges.", str(etype), g.number_of_edges(etype))

        # Use wholegraph for feature transfer
        if is_distributed() and is_wholegraph():
            from ..wholegraph import load_wg_feat
            logging.info("Allocate features with Wholegraph")
            num_parts = self._g.get_partition_book().num_partitions()

            # load node feature from wholegraph memory
            if node_feat_field:
                if isinstance(node_feat_field, str):
                    node_feat_field = {ntype: [node_feat_field] for ntype in self._g.ntypes}
                for ntype, feat_names in node_feat_field.items():
                    data = {}
                    for name in feat_names:
                        wg_folder = os.path.join(os.path.dirname(part_config), 'wholegraph')
                        assert len([feat_file for feat_file in os.listdir(wg_folder) \
                            if re.search(ntype + '~' + name, feat_file)]) > 0, \
                            f"Feature '{name}' of '{ntype}' is not in WholeGraph format. " \
                            f"Please convert all the available features to WholeGraph " \
                            f"format to utilize WholeGraph."
                        data[name] = load_wg_feat(part_config, num_parts, ntype, name)
                    if len(self._g.ntypes) == 1:
                        self._g._ndata_store.update(data)
                    else:
                        self._g._ndata_store[ntype].update(data)

            # load edge feature from wholegraph memory
            # TODO(IN): Add support for edge_feat_field
            if decoder_edge_feat:
                if isinstance(decoder_edge_feat, str):
                    decoder_edge_feat = {etype: [decoder_edge_feat] \
                        for etype in self._g.canonical_etypes}
                for etype, feat_names in decoder_edge_feat.items():
                    data = {}
                    etype_wg = ":".join(etype)
                    for name in feat_names:
                        wg_folder = os.path.join(os.path.dirname(part_config), 'wholegraph')
                        assert len([feat_file for feat_file in os.listdir(wg_folder) \
                            if re.search(etype_wg + '~' + name, feat_file)]) > 0, \
                            f"Feature '{name}' of '{etype}' is not in WholeGraph format. " \
                            f"Please convert all the available features to WholeGraph " \
                            f"format to utilize WholeGraph."
                        data[name] = load_wg_feat(part_config, num_parts, etype_wg, name)
                    if len(self._g.canonical_etypes) == 1:
                        self._g._edata_store.update(data)
                    else:
                        self._g._edata_store[etype].update(data)

            barrier()
        self.prepare_data(self._g)
        sys_tracker.check('construct training data')

    @abc.abstractmethod
    def prepare_data(self, g):
        """ Prepare the dataset.

        Arguement
        ---------
        g: Dist DGLGraph
        """

    @property
    def g(self):
        """ The distributed graph.
        """
        return self._g

    @property
    def node_feat_field(self):
        """the field of node feature"""
        return self._node_feat_field

    @property
    def edge_feat_field(self):
        """the field of edge feature"""
        return self._edge_feat_field

    def has_node_feats(self, ntype):
        """ Test if the specified node type has features.

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        bool : whether the node type has features.
        """
        if isinstance(self.node_feat_field, str):
            return True
        elif self.node_feat_field is None:
            return False
        else:
            return ntype in self.node_feat_field

    def has_edge_feats(self, etype):
        """ Test if the specified edge type has features.

        Parameters
        ----------
        etype : (str, str, str)
            The canonical edge type

        Returns
        -------
        bool : whether the edge type has features
        """
        if isinstance(self.edge_feat_field, str):
            return True
        elif self.edge_feat_field is None:
            return False
        else:
            return etype in self.edge_feat_field

    def has_node_lm_feats(self, ntype):
        """ Test if the specified node type has text features.

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        bool : whether the node type has features.
        """
        return ntype in self._lm_feat_ntypes

    def has_edge_lm_feats(self, etype):
        """ Test if the specified edge type has text features.

        Parameters
        ----------
        etype : (str, str, str)
            The edge type

        Returns
        -------
        bool : whether the node type has features.
        """
        return etype in self._lm_feat_etypes

    def get_node_feats(self, input_nodes, device='cpu'):
        """ Get the node features

        Parameters
        ----------
        input_nodes : Tensor or dict of Tensors
            The input node IDs
        device : Pytorch device
            The device where the returned node features are stored.

        Returns
        -------
        dict of Tensors : The returned node features.
        """
        g = self._g
        if not isinstance(input_nodes, dict):
            assert len(g.ntypes) == 1, \
                    "We don't know the input node type, but the graph has more than one node type."
            input_nodes = {g.ntypes[0]: input_nodes}
        return prepare_batch_input(g, input_nodes, dev=device,
                                   feat_field=self._node_feat_field)

    def get_edge_feats(self, input_edges, edge_feat_field, device='cpu'):
        """ Get the edge features

        Parameters
        ----------
        input_edges : Tensor or dict of Tensors
            The input edge IDs
        edge_feat_field: str or dict of [str ..]
            The edge data fields that stores the edge features to retrieve
        device : Pytorch device
            The device where the returned edge features are stored.

        Returns
        -------
        dict of Tensors : The returned edge features.
        """
        g = self._g
        if not isinstance(input_edges, dict):
            assert len(g.canonical_etypes) == 1, \
                    "We don't know the input edge type, but the graph has more than one edge type."
            input_edges = {g.canonical_etypes[0]: input_edges}
        return prepare_batch_edge_input(g, input_edges, dev=device,
                                        feat_field=edge_feat_field)

    def get_node_feat_size(self):
        """ Get node feat size using the given node_feat_field

        All parameters are coming from this class's own attributes.

        Note: If the self._node_feat_field is None, i.e., not given, the function will return a
              dictionary containing all node types in the self.g, and the feature sizes are all 0s.
              If given the node_feat_field, will return dictionary that only contains given node
              types.
        """
        return gs.get_node_feat_size(self.g, self._node_feat_field)

class GSgnnEdgeData(GSgnnData):  # pylint: disable=abstract-method
    """ Data for edge tasks

    Parameters
    ----------
    graph_name : str
        The graph name
    part_config : str
        The path of the partition configuration file.
    label_field : str
        The field for storing labels
    node_feat_field: str or dict of list of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of list of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    decoder_edge_feat: str or dict of list of str
        Edge features used by decoder
    lm_feat_ntypes : list of str
        The node types that contains text features.
    lm_feat_etypes : list of tuples
        The edge types that contains text features.
    """
    def __init__(self, graph_name, part_config, label_field=None,
                 node_feat_field=None, edge_feat_field=None,
                 decoder_edge_feat=None, lm_feat_ntypes=None, lm_feat_etypes=None):
        super(GSgnnEdgeData, self).__init__(graph_name, part_config,
                                            node_feat_field, edge_feat_field,
                                            decoder_edge_feat,
                                            lm_feat_ntypes=lm_feat_ntypes,
                                            lm_feat_etypes=lm_feat_etypes)
        self._label_field = label_field
        self._decoder_edge_feat = decoder_edge_feat
        if label_field is not None:
            self._labels = {}
            for etype in self._g.canonical_etypes:
                if label_field in self._g.edges[etype].data:
                    self._labels[etype] = self._g.edges[etype].data[label_field]
        else:
            self._labels = None

    def get_labels(self, eids, device='cpu'):
        """ Get the edge labels

        Parameters
        ----------
        eids : Tensor or dict of Tensors
            The edge IDs
        device : Pytorch device
            The device where the returned edge labels are stored.

        Returns
        -------
        dict of Tensors : the returned edge labels.
        """
        assert self._labels is not None, "The dataset does not have edge labels."
        assert isinstance(eids, dict)
        labels = {}
        for etype, eid in eids.items():
            assert etype in self._labels
            labels[etype] = self._labels[etype][eid].to(device)
        return labels

    @property
    def labels(self):
        """Labels"""
        return self._labels

    @property
    def decoder_edge_feat(self):
        """edge features used by decoder"""
        return self._decoder_edge_feat

    @property
    def train_idxs(self):
        """train set's indexes"""
        return self._train_idxs

    @property
    def val_idxs(self):
        """validation set's indexes"""
        return self._val_idxs

    @property
    def test_idxs(self):
        """test set's indexes"""
        return self._test_idxs

class GSgnnEdgeTrainData(GSgnnEdgeData):
    r""" Edge prediction training data

    The GSgnnEdgeTrainData prepares the data for training edge prediction.

    Parameters
    ----------
    graph_name : str
        The graph name
    part_config : str
        The path of the partition configuration file.
    train_etypes : tuple of str or list of tuples
        Target edge types for training
    eval_etypes : tuple of str or list of tuples
        Target edge types for evaluation
    label_field : str
        The field for storing labels
    node_feat_field: str or dict of list of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of list of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    decoder_edge_feat: str or dict of list of str
        Edge features used by decoder

    Examples
    ----------

    .. code:: python

        from graphstorm.dataloading import GSgnnEdgeTrainData
        from graphstorm.dataloading import GSgnnEdgeDataLoader
        ep_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                        train_etypes=[('n1', 'e1', 'n2')], label_field='label',
                                        node_feat_field='node_feat', edge_feat_field='edge_feat')
        ep_dataloader = GSgnnEdgeDataLoader(ep_data, target_idx={"e1":[0]},
                                            fanout=[15, 10], batch_size=128)
    """
    def __init__(self, graph_name, part_config, train_etypes, eval_etypes=None,
                 label_field=None, node_feat_field=None, edge_feat_field=None,
                 decoder_edge_feat=None, lm_feat_ntypes=None, lm_feat_etypes=None):
        if train_etypes is not None:
            assert isinstance(train_etypes, (tuple, list)), \
                    "The prediction etypes for training has to be a tuple or a list of tuples."
            if isinstance(train_etypes, tuple):
                train_etypes = [train_etypes]
            self._train_etypes = train_etypes
        else:
            self._train_etypes = None

        if eval_etypes is not None:
            assert isinstance(eval_etypes, (tuple, list)), \
                    "The prediction etypes for evaluation has to be a tuple or a list of tuples."
            if isinstance(eval_etypes, tuple):
                eval_etypes = [eval_etypes]
            self._eval_etypes = eval_etypes
        else:
            self._eval_etypes = train_etypes

        super(GSgnnEdgeTrainData, self).__init__(graph_name, part_config, label_field,
                                                 node_feat_field, edge_feat_field,
                                                 decoder_edge_feat,
                                                 lm_feat_ntypes=lm_feat_ntypes,
                                                 lm_feat_etypes=lm_feat_etypes)

        if self._train_etypes == [DEFAULT_ETYPE]:
            # DGL Graph edge type is not canonical. It is just list[str].
            assert self._g.ntypes == [DEFAULT_NTYPE] and \
                   self._g.etypes == [DEFAULT_ETYPE[1]], \
                f"It is required to be a homogeneous graph when target_etype is not provided " \
                f"or is set to {DEFAULT_ETYPE} on edge tasks, expect node type " \
                f"to be {[DEFAULT_NTYPE]} and edge type to be {[DEFAULT_ETYPE[1]]}, " \
                f"but get {self._g.ntypes} and {self._g.etypes}"

    def prepare_data(self, g):
        """
        Prepare the training, validation and testing edge set.

        It will setup the following class fields:
        self._train_idxs: the edge indices of the local training set.
        self._val_idxs: the edge indices of the local validation set, can be empty.
        self._test_idxs: the edge indices of the local test set, can be empty.

        Arguement
        ---------
        g: Dist DGLGraph
        """
        train_idxs = {}
        val_idxs = {}
        test_idxs = {}
        num_train = num_val = num_test = 0
        pb = g.get_partition_book()
        if self.train_etypes is None:
            self._train_etypes = g.canonical_etypes
        for canonical_etype in self.train_etypes:
            if 'train_mask' in g.edges[canonical_etype].data:
                train_idx = dgl.distributed.edge_split(
                    g.edges[canonical_etype].data['train_mask'],
                    pb, etype=canonical_etype, force_even=True)
            else:
                # If there are no training masks, we assume all edges can be used for training.
                # Therefore, we use a more memory efficient way to split the edge list.
                # TODO(zhengda) we need to split the edges properly to increase the data locality.
                train_idx = split_full_edge_list(g, canonical_etype, get_rank())
            assert train_idx is not None, "There is no training data."
            num_train += len(train_idx)
            train_idxs[canonical_etype] = train_idx

        # If eval_etypes is None, we use all edge types.
        if self.eval_etypes is None:
            self._eval_etypes = g.canonical_etypes
        for canonical_etype in self.eval_etypes:
            # user must provide validation mask
            if 'val_mask' in g.edges[canonical_etype].data:
                val_idx = dgl.distributed.edge_split(
                    g.edges[canonical_etype].data['val_mask'],
                    pb, etype=canonical_etype, force_even=True)
                val_idx = [] if val_idx is None else val_idx
                num_val += len(val_idx)
                # If there are validation data globally, we should add them to the dict.
                if dist_sum(len(val_idx)) > 0:
                    val_idxs[canonical_etype] = val_idx
            if 'test_mask' in g.edges[canonical_etype].data:
                test_idx = dgl.distributed.edge_split(
                    g.edges[canonical_etype].data['test_mask'],
                    pb, etype=canonical_etype, force_even=True)
                test_idx = [] if test_idx is None else test_idx
                num_test += len(test_idx)
                # If there are test data globally, we should add them to the dict.
                if dist_sum(len(test_idx)) > 0:
                    test_idxs[canonical_etype] = test_idx
        logging.info('part %d, train: %d, val: %d, test: %d',
                     get_rank(), num_train, num_val, num_test)

        self._train_idxs = train_idxs
        self._val_idxs = val_idxs
        self._test_idxs = test_idxs

    @property
    def train_etypes(self):
        """edge type for training"""
        return self._train_etypes

    @property
    def eval_etypes(self):
        """edge type for evaluation"""
        return self._eval_etypes

class GSgnnLPTrainData(GSgnnEdgeTrainData):
    """ Link prediction training data

    Parameters
    ----------
    graph_name : str
        The graph name
    part_config : str
        The path of the partition configuration file.
    train_etypes : tuple of str or list of tuples
        Target edge types for training
    eval_etypes : tuple of str or list of tuples
        Target edge types for evaluation
    label_field : str
        The field for storing labels
    node_feat_field: str or dict of list of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of list of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    pos_graph_feat_field: str or dist of str
        The field of the edge features used by positive graph in link prediction.
    lm_feat_ntypes : list of str
        The node types that contains text features.
    lm_feat_etypes : list of tuples
        The edge types that contains text features.
    """
    def __init__(self, graph_name, part_config, train_etypes, eval_etypes=None,
                 label_field=None, node_feat_field=None,
                 edge_feat_field=None, pos_graph_feat_field=None,
                 lm_feat_ntypes=None, lm_feat_etypes=None):
        super(GSgnnLPTrainData, self).__init__(graph_name, part_config,
                                               train_etypes, eval_etypes, label_field,
                                               node_feat_field, edge_feat_field,
                                               lm_feat_ntypes=lm_feat_ntypes,
                                               lm_feat_etypes=lm_feat_etypes)
        self._pos_graph_feat_field = pos_graph_feat_field

    @property
    def pos_graph_feat_field(self):
        """ Get edge feature fields of positive graphs
        """
        return self._pos_graph_feat_field

class GSgnnEdgeInferData(GSgnnEdgeData):
    r""" Edge prediction inference data

    GSgnnEdgeInferData prepares the data for edge prediction inference.

    Parameters
    ----------
    graph_name : str
        The graph name
    part_config : str
        The path of the partition configuration file.
    eval_etypes : tuple of str or list of tuples
        Target edge types for evaluation
    label_field : str
        The field for storing labels
    node_feat_field: str or dict of list of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of list of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    decoder_edge_feat: str or dict of list of str
        Edge features used by decoder
    lm_feat_ntypes : list of str
        The node types that contains text features.
    lm_feat_etypes : list of tuples
        The edge types that contains text features.

    Examples
    ----------

    .. code:: python

        from graphstorm.dataloading import GSgnnEdgeInferData
        from graphstorm.dataloading import GSgnnEdgeDataLoader
        ep_data = GSgnnEdgeInferData(graph_name='dummy', part_config=part_config,
                                        eval_etypes=[('n1', 'e1', 'n2')], label_field='label',
                                        node_feat_field='node_feat', edge_feat_field='edge_feat')
        ep_dataloader = GSgnnEdgeDataLoader(ep_data, target_idx={"e1":[0]},
                                            fanout=[15, 10], batch_size=128)
    """
    def __init__(self, graph_name, part_config, eval_etypes,
                 label_field=None, node_feat_field=None, edge_feat_field=None,
                 decoder_edge_feat=None, lm_feat_ntypes=None, lm_feat_etypes=None):
        if eval_etypes is not None:
            assert isinstance(eval_etypes, (tuple, list)), \
                    "The prediction etypes for evaluation has to be a tuple or a list of tuples."
            if isinstance(eval_etypes, tuple):
                eval_etypes = [eval_etypes]
            self._eval_etypes = eval_etypes
        else:
            self._eval_etypes = None # Test on all edge types

        super(GSgnnEdgeInferData, self).__init__(graph_name, part_config, label_field,
                                                 node_feat_field, edge_feat_field,
                                                 decoder_edge_feat,
                                                 lm_feat_ntypes=lm_feat_ntypes,
                                                 lm_feat_etypes=lm_feat_etypes)
        if self._eval_etypes == [DEFAULT_ETYPE]:
            # DGL Graph edge type is not canonical. It is just list[str].
            assert self._g.ntypes == [DEFAULT_NTYPE] and \
                   self._g.etypes == [DEFAULT_ETYPE[1]], \
                f"It is required to be a homogeneous graph when target_etype is not provided " \
                f"or is set to {DEFAULT_ETYPE} on edge tasks, expect node type " \
                f"to be {[DEFAULT_NTYPE]} and edge type to be {[DEFAULT_ETYPE[1]]}, " \
                f"but get {self._g.ntypes} and {self._g.etypes}"

    def prepare_data(self, g):
        """ Prepare the testing edge set if any

        It will setup self._test_idxs, the edge indices of the local test set.
        The test_idxs can be empty.

        Arguement
        ---------
        g: Dist DGLGraph
        """
        pb = g.get_partition_book()
        test_idxs = {}
        infer_idxs = {}
        # If eval_etypes is None, we use all edge types.
        if self.eval_etypes is None:
            self._eval_etypes = g.canonical_etypes
        for canonical_etype in self.eval_etypes:
            if 'test_mask' in g.edges[canonical_etype].data:
                # test_mask exists
                # we will do evaluation or inference on test data.
                test_idx = dgl.distributed.edge_split(
                    g.edges[canonical_etype].data['test_mask'],
                    pb, etype=canonical_etype, force_even=True)
                # If there are test data globally, we should add them to the dict.
                if test_idx is not None and dist_sum(len(test_idx)) > 0:
                    test_idxs[canonical_etype] = test_idx
                    infer_idxs[canonical_etype] = test_idx
            else:
                # Inference only
                # we will do inference on the entire edge set
                if get_rank() == 0:
                    logging.info("%s does not contains test_mask, skip testing %s. " + \
                            "We will do inference on the entire edge set.",
                            str(canonical_etype), str(canonical_etype))
                infer_idx = dgl.distributed.edge_split(
                    th.full((g.num_edges(canonical_etype),), True, dtype=th.bool),
                    pb, etype=canonical_etype, force_even=True)
                infer_idxs[canonical_etype] = infer_idx

        self._test_idxs = test_idxs
        self._infer_idxs = infer_idxs

    @property
    def eval_etypes(self):
        """edge type for evaluation"""
        return self._eval_etypes

    @property
    def infer_idxs(self):
        """ Set of edges to do inference.
        """
        return self._infer_idxs

#### Node classification/regression Task Data ####
class GSgnnNodeData(GSgnnData):  # pylint: disable=abstract-method
    """ Data for node tasks

    Parameters
    ----------
    graph_name : str
        The graph name
    part_config : str
        The path of the partition configuration file.
    label_field : str
        The field for storing labels
    node_feat_field: str or dict of list of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of list of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    lm_feat_ntypes : list of str
        The node types that contains text features.
    lm_feat_etypes : list of tuples
        The edge types that contains text features.
    """
    def __init__(self, graph_name, part_config, label_field=None,
                 node_feat_field=None, edge_feat_field=None,
                 lm_feat_ntypes=None, lm_feat_etypes=None):
        super(GSgnnNodeData, self).__init__(graph_name, part_config,
                                            node_feat_field, edge_feat_field,
                                            lm_feat_ntypes=lm_feat_ntypes,
                                            lm_feat_etypes=lm_feat_etypes)
        self._label_field = label_field
        if label_field is not None:
            self._labels = {}
            for ntype in self._g.ntypes:
                if label_field in self._g.nodes[ntype].data:
                    self._labels[ntype] = self._g.nodes[ntype].data[label_field]
        else:
            self._labels = None

    def get_labels(self, nids, device='cpu'):
        """ Get the node labels

        Parameters
        ----------
        nids : Tensor or dict of Tensors
            The seed nodes
        device : Pytorch device
            The device where the returned node labels are stored.

        Returns
        -------
        dict of Tensors : the returned node labels.
        """
        assert self._labels is not None, "The dataset does not have labels."
        assert isinstance(nids, dict)
        labels = {}
        for ntype, nid in nids.items():
            assert ntype in self._labels
            labels[ntype] = self._labels[ntype][nid].to(device)
        return labels

    @property
    def labels(self):
        """Labels"""
        return self._labels

    @property
    def train_idxs(self):
        """train set's indexes"""
        return self._train_idxs

    @property
    def val_idxs(self):
        """validation set's indexes"""
        return self._val_idxs

    @property
    def test_idxs(self):
        """test set's indexes"""
        return self._test_idxs

class GSgnnNodeTrainData(GSgnnNodeData):
    r""" Training data for node tasks

    GSgnnNodeTrainData prepares the data for training node prediction.

    Parameters
    ----------
    graph_name : str
        The graph name
    part_config : str
        The path of the partition configuration file.
    train_ntypes : str or list of str
        Target node types for training
    eval_ntypes : str or list of str
        Target node types for evaluation
    label_field : str
        The field for storing labels
    node_feat_field: str or dict of list of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of list of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    lm_feat_ntypes : list of str
        The node types that contains text features.
    lm_feat_etypes : list of tuples
        The edge types that contains text features.

    Examples
    ----------

    .. code:: python

        from graphstorm.dataloading import GSgnnNodeTrainData
        from graphstorm.dataloading import GSgnnNodeDataLoader

        np_data = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                        train_ntypes=['n1'], label_field='label',
                                        node_feat_field='feat')
        np_dataloader = GSgnnNodeDataLoader(np_data, target_idx={'n1':[0]},
                                            fanout=[15, 10], batch_size=128)
    """
    def __init__(self, graph_name, part_config, train_ntypes, eval_ntypes=None,
                 label_field=None, node_feat_field=None, edge_feat_field=None,
                 lm_feat_ntypes=None, lm_feat_etypes=None):
        if isinstance(train_ntypes, str):
            train_ntypes = [train_ntypes]
        assert isinstance(train_ntypes, list), \
                "prediction ntypes for training has to be a string or a list of strings."
        self._train_ntypes = train_ntypes
        if eval_ntypes is not None:
            if isinstance(eval_ntypes, str):
                eval_ntypes = [eval_ntypes]
            assert isinstance(eval_ntypes, list), \
                "prediction ntypes for evaluation has to be a string or a list of strings."
            self._eval_ntypes = eval_ntypes
        else:
            self._eval_ntypes = train_ntypes

        super(GSgnnNodeTrainData, self).__init__(graph_name, part_config,
                                                 label_field=label_field,
                                                 node_feat_field=node_feat_field,
                                                 edge_feat_field=edge_feat_field,
                                                 lm_feat_ntypes=lm_feat_ntypes,
                                                 lm_feat_etypes=lm_feat_etypes)
        if self._train_ntypes == [DEFAULT_NTYPE]:
            # DGL Graph edge type is not canonical. It is just list[str].
            assert self._g.ntypes == [DEFAULT_NTYPE] and \
                   self._g.etypes == [DEFAULT_ETYPE[1]], \
                f"It is required to be a homogeneous graph when target_ntype is not provided " \
                f"or is set to {DEFAULT_NTYPE} on node tasks, expect node type " \
                f"to be {[DEFAULT_NTYPE]} and edge type to be {[DEFAULT_ETYPE[1]]}, " \
                f"but get {self._g.ntypes} and {self._g.etypes}"

    def prepare_data(self, g):
        pb = g.get_partition_book()
        train_idxs = {}
        val_idxs = {}
        test_idxs = {}
        num_train = num_val = num_test = 0
        for ntype in self.train_ntypes:
            assert 'train_mask' in g.nodes[ntype].data, \
                    f"For training dataset, train_mask must be provided on nodes of {ntype}."

            if 'trainer_id' in g.nodes[ntype].data:
                node_trainer_ids = g.nodes[ntype].data['trainer_id']
                train_idx = dgl.distributed.node_split(g.nodes[ntype].data['train_mask'],
                                                       pb, ntype=ntype, force_even=True,
                                                       node_trainer_ids=node_trainer_ids)
            else:
                train_idx = dgl.distributed.node_split(g.nodes[ntype].data['train_mask'],
                                                       pb, ntype=ntype, force_even=True)
            assert train_idx is not None, "There is no training data."
            num_train += len(train_idx)
            train_idxs[ntype] = train_idx

        for ntype in self.eval_ntypes:
            if 'val_mask' in g.nodes[ntype].data:
                val_idx = dgl.distributed.node_split(g.nodes[ntype].data['val_mask'],
                                                     pb, ntype=ntype, force_even=True)
                # If there is no validation data, val_idx is None.
                val_idx = [] if val_idx is None else val_idx
                num_val += len(val_idx)
                # If there are validation data globally, we should add them to the dict.
                if dist_sum(len(val_idx)) > 0:
                    val_idxs[ntype] = val_idx
            if 'test_mask' in g.nodes[ntype].data:
                test_idx = dgl.distributed.node_split(g.nodes[ntype].data['test_mask'],
                                                      pb, ntype=ntype, force_even=True)
                # If there is no test data, test_idx is None.
                test_idx = [] if test_idx is None else test_idx
                num_test += len(test_idx)
                # If there are test data globally, we should add them to the dict.
                if dist_sum(len(test_idx)) > 0:
                    test_idxs[ntype] = test_idx

        logging.info('part %d, train: %d, val: %d, test: %d',
                     get_rank(), num_train, num_val, num_test)

        self._train_idxs = train_idxs
        self._val_idxs = val_idxs
        self._test_idxs = test_idxs

    def get_unlabeled_idxs(self):
        """ Collect indices of nodes not used for training.
        """
        g = self.g
        pb = g.get_partition_book()
        unlabeled_idxs = {}
        num_unlabeled = 0
        for ntype in self.train_ntypes:
            unlabeled_mask = flip_node_mask(g.nodes[ntype].data['train_mask'],
                                            self._train_idxs[ntype])
            if 'trainer_id' in g.nodes[ntype].data:
                node_trainer_ids = g.nodes[ntype].data['trainer_id']
                unlabeled_idx = dgl.distributed.node_split(unlabeled_mask,
                                                       pb, ntype=ntype, force_even=True,
                                                       node_trainer_ids=node_trainer_ids)
            else:
                unlabeled_idx = dgl.distributed.node_split(unlabeled_mask,
                                                       pb, ntype=ntype, force_even=True)
            assert unlabeled_idx is not None, "There is no training data."
            num_unlabeled += len(unlabeled_idx)
            unlabeled_idxs[ntype] = unlabeled_idx
        logging.info('part %d, unlabeled: %d', get_rank(), num_unlabeled)
        return unlabeled_idxs

    @property
    def train_ntypes(self):
        """node type for training"""
        return self._train_ntypes

    @property
    def eval_ntypes(self):
        """node type for evaluation"""
        return self._eval_ntypes

class GSgnnNodeInferData(GSgnnNodeData):
    r""" Inference data for node tasks

    GSgnnNodeInferData prepares the data for node prediction inference.

    Parameters
    ----------
    graph_name : str
        The graph name
    part_config : str
        The path of the partition configuration file.
    eval_ntypes : str or list of str
        Target node types
    label_field : str
        The field for storing labels
    node_feat_field: str or dict of list of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of list of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    lm_feat_ntypes : list of str
        The node types that contains text features.
    lm_feat_etypes : list of tuples
        The edge types that contains text features.

    Examples
    ----------

    .. code:: python

        from graphstorm.dataloading import GSgnnNodeInferData
        from graphstorm.dataloading import

        np_data = GSgnnNodeInferData(graph_name='dummy', part_config=part_config,
                                        eval_ntypes=['n1'], label_field='label',
                                        node_feat_field='feat')
        np_dataloader = GSgnnNodeDataLoader(np_data, target_idx={'n1':[0]},
                                            fanout=[15, 10], batch_size=128)
    """
    def __init__(self, graph_name, part_config, eval_ntypes,
                 label_field=None, node_feat_field=None, edge_feat_field=None,
                 lm_feat_ntypes=None, lm_feat_etypes=None):
        if isinstance(eval_ntypes, str):
            eval_ntypes = [eval_ntypes]
        assert isinstance(eval_ntypes, list), \
                "prediction ntypes for evaluation has to be a string or a list of strings."
        self._eval_ntypes = eval_ntypes

        super(GSgnnNodeInferData, self).__init__(graph_name, part_config,
                                                 label_field=label_field,
                                                 node_feat_field=node_feat_field,
                                                 edge_feat_field=edge_feat_field,
                                                 lm_feat_ntypes=lm_feat_ntypes,
                                                 lm_feat_etypes=lm_feat_etypes)

        if self._eval_ntypes == [DEFAULT_NTYPE]:
            # DGL Graph edge type is not canonical. It is just list[str].
            assert self._g.ntypes == [DEFAULT_NTYPE] and \
                   self._g.etypes == [DEFAULT_ETYPE[1]], \
                f"It is required to be a homogeneous graph when target_ntype is not provided " \
                f"or is set to {DEFAULT_NTYPE} on node tasks, expect node type " \
                f"to be {[DEFAULT_NTYPE]} and edge type to be {[DEFAULT_ETYPE[1]]}, " \
                f"but get {self._g.ntypes} and {self._g.etypes}"

    def prepare_data(self, g):
        """
        Prepare the testing node set if any

        It will setup self._test_idxs, the node indices of the local test set.
        The test_idxs can be empty.

        Arguement
        ---------
        g: DistGraph
            The distributed graph.
        """
        pb = g.get_partition_book()
        test_idxs = {}
        infer_idxs = {}
        for ntype in self.eval_ntypes:
            node_trainer_ids = g.nodes[ntype].data['trainer_id'] \
                if 'trainer_id' in g.nodes[ntype].data else None
            if 'test_mask' in g.nodes[ntype].data:
                # test_mask exists
                # we will do evaluation or inference on test data.
                test_idx = dgl.distributed.node_split(g.nodes[ntype].data['test_mask'],
                                                      pb, ntype=ntype, force_even=True,
                                                      node_trainer_ids=node_trainer_ids)
                # If there are test data globally, we should add them to the dict.
                if test_idx is not None and dist_sum(len(test_idx)) > 0:
                    test_idxs[ntype] = test_idx
                    infer_idxs[ntype] = test_idx
                elif test_idx is None:
                    logging.warning("%s does not contains test data, skip testing %s",
                                    ntype, ntype)
            else:
                # Inference only
                # we will do inference on the entire edge set
                logging.info("%s does not contains test_mask, skip testing %s. " + \
                        "We will do inference on the entire node set.", ntype, ntype)
                infer_idx = dgl.distributed.node_split(
                    th.full((g.num_nodes(ntype),), True, dtype=th.bool),
                    pb, ntype=ntype, force_even=True,
                    node_trainer_ids=node_trainer_ids)
                infer_idxs[ntype] = infer_idx
        self._test_idxs = test_idxs
        self._infer_idxs = infer_idxs

    @property
    def eval_ntypes(self):
        """node type for evaluation"""
        return self._eval_ntypes

    @property
    def infer_idxs(self):
        """ Set of nodes to do inference.
        """
        return self._infer_idxs

class GSDistillData(Dataset):
    """ Dataset for distillation

    Parameters
    ----------
    file_list : list of str
        List of input files.
    tokenizer : transformers.AutoTokenizer
        HuggingFace Tokenizer.
    max_seq_len : int
        Maximum sequence length.
    """
    def __init__(self, file_list, tokenizer, max_seq_len):
        super().__init__()
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.token_id_inputs, self.labels = self.get_inputs()

    def get_inputs(self):
        """ Tokenize textual data."""
        inputs = [pd.read_parquet(file_name) for file_name in self.file_list]
        inputs = pd.concat(inputs)

        token_id_inputs = []
        for i in range(len(inputs["textual_feats"])):
            # Do tokenization line by line. The length of token_ids may vary.
            # will do padding in the collate function.
            tokens = self.tokenizer.tokenize(inputs["textual_feats"][i])
            tokens.insert(0, self.tokenizer.cls_token) # cls token for pooling
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_ids = token_ids[0:min(len(token_ids), self.max_seq_len)]
            # token_id_inputs cannot be converted to tensor here
            # because of the different sequence length
            token_id_inputs.append(token_ids)

        labels = th.tensor(inputs["embeddings"], dtype=th.float, device="cpu")
        return token_id_inputs, labels

    def __len__(self):
        return len(self.token_id_inputs)

    def __getitem__(self, index):
        input_ids = th.tensor(self.token_id_inputs[index], dtype=th.int32, device="cpu")
        labels = self.labels[index]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def get_collate_fn(self):
        '''get collate function
        '''
        def collate_fn(batch):
            ''' Pad tensors in a batch to the same length.
            '''
            ## pad inputs
            input_ids_list = [x["input_ids"] for x in batch]

            padded_input_ids = th.nn.utils.rnn.pad_sequence(input_ids_list,
                batch_first=True, padding_value=self.tokenizer.pad_token_id)
            ## compute mask
            attention_mask = (padded_input_ids != self.tokenizer.pad_token_id).float()
            labels = th.stack([x["labels"] for x in batch], 0)

            return {
                "input_ids": padded_input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        return collate_fn
