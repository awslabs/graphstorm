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
import abc
import torch as th
import dgl

from ..utils import get_rank
from ..utils import sys_tracker
from .utils import dist_sum

def split_full_edge_list(g, etype, rank):
    ''' Split the full edge list of a graph.
    '''
    # TODO(zhengda) we need to split the edges to co-locate data and computation.
    # We assume that the number of edges is larger than the number of processes.
    # This should always be true unless a user's training set is extremely small.
    assert g.num_edges(etype) >= th.distributed.get_world_size()
    start = g.num_edges(etype) // th.distributed.get_world_size() * rank
    end = g.num_edges(etype) // th.distributed.get_world_size() * (rank + 1)
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
            feat[ntype] = th.cat([g.nodes[ntype].data[fname][nid].to(dev) \
                for fname in feat_name], dim=1)
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
            feat[etypes] = th.cat([g.edges[etypes].data[fname][eid].to(dev) \
                for fname in feat_name], dim=-1)
    return feat

class GSgnnData():
    """ The GraphStorm data

    Parameters
    ----------
    graph_name : str
        The graph name
    part_config : str
        The path of the partition configuration file.
    node_feat_field: str or dict of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    """

    def __init__(self, graph_name, part_config, node_feat_field, edge_feat_field):
        self._g = dgl.distributed.DistGraph(graph_name, part_config=part_config)
        self._node_feat_field = node_feat_field
        self._edge_feat_field = edge_feat_field

        self._train_idxs = {}
        self._val_idxs = {}
        self._test_idxs = {}

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
        """ Get the node features

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
    node_feat_field: str or dict of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    """
    def __init__(self, graph_name, part_config, label_field=None,
                 node_feat_field=None, edge_feat_field=None):
        super(GSgnnEdgeData, self).__init__(graph_name, part_config,
                                            node_feat_field, edge_feat_field)

        self._label_field = label_field
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
    """ Edge prediction training data

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
    node_feat_field: str or dict of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    """
    def __init__(self, graph_name, part_config, train_etypes, eval_etypes=None,
                 label_field=None, node_feat_field=None, edge_feat_field=None):
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
                                                 node_feat_field, edge_feat_field)

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
                num_val += len(val_idx)
                # If there are validation data globally, we should add them to the dict.
                if dist_sum(len(val_idx)) > 0:
                    val_idxs[canonical_etype] = val_idx
            if 'test_mask' in g.edges[canonical_etype].data:
                test_idx = dgl.distributed.edge_split(
                    g.edges[canonical_etype].data['test_mask'],
                    pb, etype=canonical_etype, force_even=True)
                num_test += len(test_idx)
                # If there are test data globally, we should add them to the dict.
                if dist_sum(len(test_idx)) > 0:
                    test_idxs[canonical_etype] = test_idx
        print('part {}, train: {}, val: {}, test: {}'.format(get_rank(), num_train,
                                                             num_val, num_test))

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

class GSgnnEdgeInferData(GSgnnEdgeData):
    """ Edge prediction inference data

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
    node_feat_field: str or dict of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    """
    def __init__(self, graph_name, part_config, eval_etypes,
                 label_field=None, node_feat_field=None, edge_feat_field=None):
        if eval_etypes is not None:
            assert isinstance(eval_etypes, (tuple, list)), \
                    "The prediction etypes for evaluation has to be a tuple or a list of tuples."
            if isinstance(eval_etypes, tuple):
                eval_etypes = [eval_etypes]
            self._eval_etypes = eval_etypes
        else:
            self._eval_etypes = None # Test on all edge types

        super(GSgnnEdgeInferData, self).__init__(graph_name, part_config, label_field,
                                                 node_feat_field, edge_feat_field)

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
        # If eval_etypes is None, we use all edge types.
        if self.eval_etypes is None:
            self._eval_etypes = g.canonical_etypes
        # test_mask exists
        for canonical_etype in self.eval_etypes:
            if 'test_mask' in g.edges[canonical_etype].data:
                test_idx = dgl.distributed.edge_split(
                    g.edges[canonical_etype].data['test_mask'],
                    pb, etype=canonical_etype, force_even=True)
                # If there are test data globally, we should add them to the dict.
                if dist_sum(len(test_idx)) > 0:
                    test_idxs[canonical_etype] = test_idx
            else:
                print(f"WARNING: {canonical_etype} does not contains " \
                      "test_mask, skip testing {canonical_etype}")
        self._test_idxs = test_idxs

    @property
    def eval_etypes(self):
        """edge type for evaluation"""
        return self._eval_etypes

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
    node_feat_field: str or dict of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    """
    def __init__(self, graph_name, part_config, label_field=None,
                 node_feat_field=None, edge_feat_field=None):
        super(GSgnnNodeData, self).__init__(graph_name, part_config,
                                            node_feat_field, edge_feat_field)
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
    """ Training data for node tasks

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
    node_feat_field: str or dict of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    """
    def __init__(self, graph_name, part_config, train_ntypes, eval_ntypes=None,
                 label_field=None, node_feat_field=None, edge_feat_field=None):
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
                                                 edge_feat_field=edge_feat_field)

    def prepare_data(self, g):
        pb = g.get_partition_book()
        train_idxs = {}
        val_idxs = {}
        test_idxs = {}
        num_train = num_val = num_test = 0
        for ntype in self.train_ntypes:
            assert 'train_mask' in g.nodes[ntype].data, \
                    "For training dataset, train_mask must be provided."

            if 'trainer_id' in g.nodes[ntype].data:
                node_trainer_ids = g.nodes[ntype].data['trainer_id']
                train_idx = dgl.distributed.node_split(g.nodes[ntype].data['train_mask'],
                                                       pb, ntype=ntype, force_even=True,
                                                       node_trainer_ids=node_trainer_ids)
            else:
                train_idx = dgl.distributed.node_split(g.nodes[ntype].data['train_mask'],
                                                       pb, ntype=ntype, force_even=True)
            num_train += len(train_idx)
            train_idxs[ntype] = train_idx

        for ntype in self.eval_ntypes:
            if 'val_mask' in g.nodes[ntype].data:
                val_idx = dgl.distributed.node_split(g.nodes[ntype].data['val_mask'],
                                                     pb, ntype=ntype, force_even=True)
                num_val += len(val_idx)
                # If there are validation data globally, we should add them to the dict.
                if dist_sum(len(val_idx)) > 0:
                    val_idxs[ntype] = val_idx
            if 'test_mask' in g.nodes[ntype].data:
                test_idx = dgl.distributed.node_split(g.nodes[ntype].data['test_mask'],
                                                      pb, ntype=ntype, force_even=True)
                num_test += len(test_idx)
                # If there are test data globally, we should add them to the dict.
                if dist_sum(len(test_idx)) > 0:
                    test_idxs[ntype] = test_idx

        print('part {}, train: {}, val: {}, test: {}'.format(get_rank(), num_train,
                                                             num_val, num_test))

        self._train_idxs = train_idxs
        self._val_idxs = val_idxs
        self._test_idxs = test_idxs

    @property
    def train_ntypes(self):
        """node type for training"""
        return self._train_ntypes

    @property
    def eval_ntypes(self):
        """node type for evaluation"""
        return self._eval_ntypes

class GSgnnNodeInferData(GSgnnNodeData):
    """ Inference data for node tasks

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
    node_feat_field: str or dict of str
        Fields to extract node features. It's a dict if different node types have
        different feature names.
    edge_feat_field : str or dict of str
        The field of the edge features. It's a dict if different edge types have
        different feature names.
    """
    def __init__(self, graph_name, part_config, eval_ntypes,
                 label_field=None, node_feat_field=None, edge_feat_field=None):
        if isinstance(eval_ntypes, str):
            eval_ntypes = [eval_ntypes]
        assert isinstance(eval_ntypes, list), \
                "prediction ntypes for evaluation has to be a string or a list of strings."
        self._eval_ntypes = eval_ntypes

        super(GSgnnNodeInferData, self).__init__(graph_name, part_config,
                                                 label_field=label_field,
                                                 node_feat_field=node_feat_field,
                                                 edge_feat_field=edge_feat_field)

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
        for ntype in self.eval_ntypes:
            if 'test_mask' in g.nodes[ntype].data:
                node_trainer_ids = g.nodes[ntype].data['trainer_id'] \
                        if 'trainer_id' in g.nodes[ntype].data else None
                test_idx = dgl.distributed.node_split(g.nodes[ntype].data['test_mask'],
                                                      pb, ntype=ntype, force_even=True,
                                                      node_trainer_ids=node_trainer_ids)
                # If there are test data globally, we should add them to the dict.
                if dist_sum(len(test_idx)) > 0:
                    test_idxs[ntype] = test_idx
        self._test_idxs = test_idxs

    @property
    def eval_ntypes(self):
        """node type for evaluation"""
        return self._eval_ntypes
