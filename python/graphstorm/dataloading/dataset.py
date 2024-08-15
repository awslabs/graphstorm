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
import logging
import re

import torch as th
import dgl
import pandas as pd
from dgl.distributed.constants import DEFAULT_NTYPE, DEFAULT_ETYPE
from torch.utils.data import Dataset

from ..utils import get_rank, get_world_size, is_distributed, barrier, is_wholegraph
from ..utils import sys_tracker
from .utils import dist_sum, flip_node_mask
from ..utils import get_graph_name

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
                assert fname in g.nodes[ntype].data, \
                    f"{fname} does not exist as a node feature of {ntype}"
                data = g.nodes[ntype].data[fname]
                if is_wholegraph_embedding(data):
                    data = data.gather(nid.to(dev))
                else:
                    data = data[nid].to(dev)
                feats.append(data)
            assert len(feats) > 0, \
                "No feature exists in the graph. " \
                f"Expecting the graph have following node features {feat_name}."

            if len(feats[0].shape) == 1:
                # The feature is 1D. It will be features for label
                assert len(feats) == 1, \
                    "For 1D features, we assume they are label features." \
                    f"Please access them 1 by 1, but get {feat_name}"
                feat[ntype] = feats[0]
            else:
                # The feature is 2D
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
    for etype, eid in input_edges.items():
        feat_name = None if feat_field is None else \
            [feat_field] if isinstance(feat_field, str) \
            else feat_field[etype] if etype in feat_field else None

        if feat_name is not None:
            # concatenate multiple features together
            feats = []
            for fname in feat_name:
                assert fname in g.edges[etype].data, \
                    f"{fname} does not exist as an edge feature of {etype}"
                data = g.edges[etype].data[fname]
                if is_wholegraph_embedding(data):
                    data = data.gather(eid.to(dev))
                else:
                    data = data[eid].to(dev)
                feats.append(data)

            assert len(feats) > 0, \
                "No feature exists in the graph. " \
                f"Expecting the graph have following edge features {feat_name}."

            if len(feats[0].shape) == 1:
                # The feature is 1D. It will be features for label
                assert len(feats) == 1, \
                    "For 1D features, we assume they are label features." \
                    f"Please access them 1 by 1, but get {feat_name}"
                feat[etype] = feats[0]
            else:
                # The feature is 2D
                feat[etype] = th.cat(feats, dim=1)
    return feat

class GSgnnData():
    """ The GraphStorm data class.

    Parameters
    ----------
    part_config : str
        The path of the partition configuration JSON file.
    node_feat_field: str or dict of list of str
        The fields of the node features that will be encoded by ``GSNodeInputLayer``.
        It's a dict if different node types have
        different feature names.
        Default: None.
    edge_feat_field : str or dict of list of str
        The fields of the edge features. It's a dict, if different edge types have
        different feature names. This argument is reserved for future usage when the
        ``GSEdgeInputLayer`` is implemented.
        Default: None.
    lm_feat_ntypes : list of str
        The node types that contains text features.
        Default: None.
    lm_feat_etypes : list of tuples
        The edge types that contains text features.
        Default: None.
    """

    def __init__(self, part_config, node_feat_field=None, edge_feat_field=None,
                 lm_feat_ntypes=None, lm_feat_etypes=None):
        graph_name = get_graph_name(part_config)
        self._g = dgl.distributed.DistGraph(graph_name, part_config=part_config)
        self._graph_name = graph_name
        # Note: node_feat_field and edge_feat_field are useful in three cases:
        # 1. WholeGraph: As the feature information is not stored in g,
        #    node_feat_field and edge_feat_field are used to tell GraphStorm
        #    what features should be loaded by WholeGraph
        # 2. Used by _ReconstructedNeighborSampler to decide whether a node
        #    or an edge has feature.
        # 3. Used by do_full_graph_inference and do_mini_batch_inference when
        #    computing input embeddings.
        self._check_node_feats(node_feat_field)
        self._node_feat_field = node_feat_field
        self._edge_feat_field = edge_feat_field
        self._lm_feat_ntypes = lm_feat_ntypes if lm_feat_ntypes is not None else []
        self._lm_feat_etypes = lm_feat_etypes if lm_feat_etypes is not None else []

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
            if edge_feat_field:
                if isinstance(edge_feat_field, str):
                    edge_feat_field = {etype: [edge_feat_field] \
                        for etype in self._g.canonical_etypes}
                for etype, feat_names in edge_feat_field.items():
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
        sys_tracker.check('construct training data')

    @property
    def g(self):
        """ The distributed graph loaded using information in the given part_config JSON file.
        """
        return self._g

    @property
    def graph_name(self):
        """ The distributed graph's name extracted from the given part_config JSON file.
        """
        return self._graph_name

    @property
    def node_feat_field(self):
        """ The fields of node features given in initialization.
        """
        return self._node_feat_field

    @property
    def edge_feat_field(self):
        """ The fields of edge features given in initialization.
        """
        return self._edge_feat_field

    def _check_node_feats(self, node_feat_field):
        g = self._g
        if node_feat_field is None:
            return

        for ntype in g.ntypes:
            if isinstance(node_feat_field, str):
                feat_names = [node_feat_field]
            elif isinstance(node_feat_field, dict):
                feat_names = node_feat_field[ntype] \
                    if ntype in node_feat_field else []
            else:
                raise TypeError("Node feature field must be a string " \
                                "or a dictionary of list of strings, " \
                                f"but get {node_feat_field}")
            for feat_name in feat_names:
                assert feat_name in g.nodes[ntype].data, \
                    f"Warning. The feature \"{feat_name}\" " \
                    f"does not exists for the node type \"{ntype}\"."

    def has_node_feats(self, ntype):
        """ Test if the specified node type has features.

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        bool : Whether the node type has features.
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
            The canonical edge type.

        Returns
        -------
        bool : Whether the edge type has features.
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
            The node type.

        Returns
        -------
        bool : Whether the node type has text features.
        """
        return ntype in self._lm_feat_ntypes

    def has_edge_lm_feats(self, etype):
        """ Test if the specified edge type has text features.

        Parameters
        ----------
        etype : (str, str, str)
            The edge type.

        Returns
        -------
        bool : Whether the edge type has text features.
        """
        return etype in self._lm_feat_etypes

    def get_node_feats(self, input_nodes, nfeat_fields, device='cpu'):
        """ Get the node features of the given input nodes. The feature fields are defined
        in ``nfeat_fields``.

        Parameters
        ----------
        input_nodes : Tensor or dict of Tensors
            The input node IDs.
        nfeat_fields : str or dict of [str ...]
            The node feature fields to be extracted.
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
                                   feat_field=nfeat_fields)

    def get_edge_feats(self, input_edges, efeat_fields, device='cpu'):
        """ Get the edge features of the given input edges. The feature fields are defined
        in ``efeat_fields``.

        Parameters
        ----------
        input_edges : Tensor or dict of Tensors
            The input edge IDs.
        efeat_fields: str or dict of [str ..]
            The edge feature fields to be extracted.
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
                                        feat_field=efeat_fields)

    def _check_ntypes(self, ntypes):
        """ Check the input ntype(s) and convert it into list of strs

        Parameters
        __________
        ntypes: str or list of str
            List of node types

        Return
        ------
        list: list of node types
        """
        assert ntypes is not None, \
            "Prediction ntype(s) must be provided."
        if isinstance(ntypes, str):
            ntypes = [ntypes]
        assert isinstance(ntypes, list), \
                "Prediction ntypes have to be a string or a list of strings."

        if ntypes == [DEFAULT_NTYPE]:
            # DGL Graph edge type is not canonical. It is just list[str].
            assert self._g.ntypes == [DEFAULT_NTYPE] and \
                   self._g.etypes == [DEFAULT_ETYPE[1]], \
                f"It is required to be a homogeneous graph when target_ntype is not provided " \
                f"or is set to {DEFAULT_NTYPE} on node tasks, expect node type " \
                f"to be {[DEFAULT_NTYPE]} and edge type to be {[DEFAULT_ETYPE[1]]}, " \
                f"but get {self._g.ntypes} and {self._g.etypes}"

        return ntypes

    def _check_node_mask(self, ntypes, masks):
        """ Check the node mask(s) and convert it into list of strs

        Parameters
        __________
        ntypes: str or list of str
            List of node types
        masks: str or list of str
            The node features field storing the masks.

        Return
        ------
        list: list of mask fields
        """
        if isinstance(ntypes, str):
            # ntypes is a string, convert it into list
            ntypes = [ntypes]

        if masks is None or isinstance(masks, str):
            # Mask is a string or None
            # All the masks are using the same name
            masks = [masks] * len(ntypes)

        assert len(ntypes) == len(masks), \
            "Expecting the number of ntypes matches the number of mask fields, " \
            f"But get {len(ntypes)} and {len(masks)}." \
            f"The node types are {ntypes} and the mask fileds are {masks}"
        return masks

    def get_unlabeled_node_set(self, train_idxs, mask="train_mask"):
        """ Get node indexes not having the given mask in the training set.

        Parameters
        __________
        train_idxs: dict of Tensor
            The training set.
        mask: str or list of str
            The node feature fields storing the training mask.
            Default: "train_mask".

        Returns
        -------
        dict of Tensors : The returned node indexes
        """
        g = self.g
        pb = g.get_partition_book()
        unlabeled_idxs = {}
        num_unlabeled = 0
        ntypes = list(train_idxs.keys())
        masks = self._check_node_mask(ntypes, mask)

        for ntype, msk in zip(ntypes, masks):
            unlabeled_mask = flip_node_mask(g.nodes[ntype].data[msk],
                                            train_idxs[ntype])
            node_trainer_ids = g.nodes[ntype].data['trainer_id'] \
                if 'trainer_id' in g.nodes[ntype].data else None

            unlabeled_idx = dgl.distributed.node_split(unlabeled_mask,
                                                       pb, ntype=ntype, force_even=True,
                                                       node_trainer_ids=node_trainer_ids)
            assert unlabeled_idx is not None, "There is no unlabeled data."
            num_unlabeled += len(unlabeled_idx)
            unlabeled_idxs[ntype] = unlabeled_idx
        logging.info('part %d, unlabeled: %d', get_rank(), num_unlabeled)
        return unlabeled_idxs

    def get_node_train_set(self, ntypes, mask="train_mask"):
        """ Get the training set for the given node types under the given mask.

        Parameters
        __________
        ntypes: str or list of str
            Node types to get the training set.
        mask: str or list of str
            The node feature fields storing the training mask.
            Default: "train_mask".

        Returns
        -------
        dict of Tensors : The returned training node indexes.
        """
        g = self._g
        pb = g.get_partition_book()
        train_idxs = {}
        num_train = 0
        ntypes = self._check_ntypes(ntypes)
        masks = self._check_node_mask(ntypes, mask)

        for ntype, msk in zip(ntypes, masks):
            if msk in g.nodes[ntype].data:
                node_trainer_ids = g.nodes[ntype].data['trainer_id'] \
                    if 'trainer_id' in g.nodes[ntype].data else None

                train_idx = dgl.distributed.node_split(g.nodes[ntype].data[msk],
                                                       pb, ntype=ntype, force_even=True,
                                                       node_trainer_ids=node_trainer_ids)
                assert train_idx is not None, \
                    f"The training set of the {ntype} nodes is " \
                    "empty or the number of training nodes is smaller than " \
                    f"the number of trainers {get_world_size()}" \
                    "Please check your training data and make sure " \
                    "the number of trainers (GPUs, if using GPU-supported " \
                    "machines) is smaller than the number of training data."
                num_train += len(train_idx)
                train_idxs[ntype] = train_idx

                logging.debug('part %d | ntype %s, mask %s | train: %d',
                              get_rank(), ntype, msk, len(train_idx))
            else:
                # Train mask may not exist for certain node types
                logging.debug('part %d | ntype %s, mask %s | train: 0',
                            get_rank(), ntype, msk)
        logging.info('part %d, train %d', get_rank(), num_train)
        return train_idxs

    def _get_node_set(self, ntypes, mask):
        """ called by get_node_val_set and get_node_test_set
        """
        g = self._g
        pb = g.get_partition_book()
        idxs = {}
        num_data = 0
        ntypes = self._check_ntypes(ntypes)
        masks = self._check_node_mask(ntypes, mask)

        for ntype, msk in zip(ntypes, masks):
            if msk in g.nodes[ntype].data:
                idx = dgl.distributed.node_split(g.nodes[ntype].data[msk],
                                                     pb, ntype=ntype, force_even=True)
                # If there is no validation/test data, idx is None.
                idx = [] if idx is None else idx
                num_data += len(idx)
                # If there are validation/test data globally, we should add them to the dict.
                total_num_idx = dist_sum(len(idx))
                if total_num_idx > 0:
                    if total_num_idx >= get_world_size():
                        # The size of the validation or test set is larger
                        # than the world size. Each validation/test dataloader
                        # will not be empty
                        idxs[ntype] = idx
                    else:
                        # There is not enough validation or test data.
                        # One or more validation/test dataloader will be
                        # empty, which will cause an evaluation error.
                        #
                        # To avoid the error, force each trainer or
                        # inferencer to use the entire validation
                        # or test set.
                        idx = th.nonzero(g.nodes[ntype].data[msk][ \
                            th.arange(g.num_nodes(ntype))]).reshape(-1,) # 1D tensor
                        idxs[ntype] = idx
                        logging.warning("Since the total number of validation/test data"
                                        "of %s, which is %d, is less than the number of "
                                        "workers %d, we will force each worker to do "
                                        "validation or testing on the entire "
                                        "validation/test set.",
                                        ntype, total_num_idx, get_world_size())

                logging.debug('part %d | ntype %s, mask %s | val/test: %d',
                          get_rank(), ntype, msk, len(idx))
        return idxs, num_data

    def get_node_val_set(self, ntypes, mask="val_mask"):
        """ Get the validation set for the given node types under the given mask.

        Parameters
        __________
        ntypes: str or list of str
            Node types to get the validation set.
        mask: str or list of str
            The node feature fields storing the validation mask.
            Default: "val_mask".

        Returns
        -------
        dict of Tensors : The returned validation node indexes.
        """
        idxs, num_data = self._get_node_set(ntypes, mask)
        logging.info('part %d, val %d', get_rank(), num_data)

        return idxs

    def get_node_test_set(self, ntypes, mask="test_mask"):
        """ Get the test set for the given node types under the given mask.

        Parameters
        __________
        ntypes: str or list of str
            Node types to get the test set.
        mask: str or list of str
            The node feature fields storing the test mask.
            Default: "test_mask".

        Returns
        -------
        dict of Tensors : The returned test node indexes.
        """
        idxs, num_data = self._get_node_set(ntypes, mask)
        logging.info('part %d, test %d', get_rank(), num_data)

        return idxs

    def get_node_infer_set(self, ntypes, mask="test_mask"):
        """ Get inference node set for the given node types under the given mask.

        If the mask exists in ``g.nodes[ntype].data``, the inference set
        is collected based on the mask.
        If not exist, the entire node set are treated as the inference set.

        Parameters
        __________
        ntypes: str or list of str
            Node types to get the inference set.
        mask: str or list of str
            The node feature fields storing the inference mask.
            Default: "test_mask".

        Returns
        -------
        dict of Tensors : The returned inference node indexes.
        """
        g = self._g
        pb = g.get_partition_book()
        infer_idxs = {}
        ntypes = self._check_ntypes(ntypes)
        masks = self._check_node_mask(ntypes, mask)

        for ntype, msk in zip(ntypes, masks):
            node_trainer_ids = g.nodes[ntype].data['trainer_id'] \
                if 'trainer_id' in g.nodes[ntype].data else None
            if msk in g.nodes[ntype].data:
                # We only do inference on a subset of nodes
                # according to the mask
                infer_idx = dgl.distributed.node_split(g.nodes[ntype].data[msk],
                                                       pb, ntype=ntype, force_even=True,
                                                       node_trainer_ids=node_trainer_ids)
                logging.info("%s contains %s, we will do inference based on the mask",
                             ntype, msk)
            else:
                # We will do inference on the entire edge set
                logging.info("%s does not contains %s" + \
                        "We will do inference on the entire node set.", ntype, msk)
                infer_idx = dgl.distributed.node_split(
                    th.full((g.num_nodes(ntype),), True, dtype=th.bool),
                    pb, ntype=ntype, force_even=True,
                    node_trainer_ids=node_trainer_ids)
            infer_idxs[ntype] = infer_idx

        return infer_idxs

    def _check_etypes(self, etypes):
        """ Check the input etype(s) and convert it into list of tuples

        Parameters
        __________
        etypes: tuples or list of tuples
            Edge types

        Return
        list: list of edge types
        ------
        """
        assert isinstance(etypes, (tuple, list)), \
            "Prediction etypes have to be a tuple or a list of tuples."
        if isinstance(etypes, tuple):
            etypes = [etypes]

        if etypes == [DEFAULT_ETYPE]:
            # DGL Graph edge type is not canonical. It is just list[str].
            assert self._g.ntypes == [DEFAULT_NTYPE] and \
                   self._g.etypes == [DEFAULT_ETYPE[1]], \
                f"It is required to be a homogeneous graph when target_etype is not provided " \
                f"or is set to {DEFAULT_ETYPE} on edge tasks, expect node type " \
                f"to be {[DEFAULT_NTYPE]} and edge type to be {[DEFAULT_ETYPE[1]]}, " \
                f"but get {self._g.ntypes} and {self._g.etypes}"
        return etypes

    def _check_edge_mask(self, etypes, masks):
        """ Check the edge mask(s) and convert it into list of strs

        Parameters
        __________
        etypes: tuple or list of tuple
            List of edge types
        masks: str or list of str
            The edge feature fields storing the masks.

        Return
        ------
        list: list of mask fields
        """
        if isinstance(etypes, tuple):
            # etypes is a tuple of strings, convert it into list
            etypes = [etypes]

        if masks is None or isinstance(masks, str):
            # Mask is a string or None
            # All the masks are using the same name
            masks = [masks] * len(etypes)

        assert len(etypes) == len(masks), \
            "Expecting the number of etypes matches the number of mask fields, " \
            f"But get {len(etypes)} and {len(masks)}." \
            f"The edge types are {etypes} and the mask fileds are {masks}"

        return masks

    def _exclude_reverse_etype(self, etypes, reverse_edge_types_map=None):
        if reverse_edge_types_map is None:
            return etypes
        etypes = set(etypes)
        for rev_etype in list(reverse_edge_types_map.values()):
            etypes.remove(rev_etype)
        return list(etypes)

    def get_edge_train_set(self, etypes=None, mask="train_mask",
                           reverse_edge_types_map=None):
        """ Get the training set for the given edge types under the given mask.

        Parameters
        __________
        etypes: list of str
            List of edge types to get the training set.
            If set to None, all the edge types are included.
            Default: None.
        mask: str or list of str
            The edge feature fields storing the training mask.
            Default: "train_mask".
        reverse_edge_types_map: dict of tupeles
            A map for reverse edge types in the format of {(edge type):(reversed edge type)}.
            Default: None.

        Returns
        -------
        dict of Tensors : The returned training edge indexes.
        """
        g = self._g
        pb = g.get_partition_book()
        train_idxs = {}
        num_train = 0
        etypes = self._exclude_reverse_etype(g.canonical_etypes, reverse_edge_types_map) \
            if etypes is None else self._check_etypes(etypes)
        masks = self._check_edge_mask(etypes, mask)

        for canonical_etype, msk in zip(etypes, masks):
            if msk in g.edges[canonical_etype].data:
                train_idx = dgl.distributed.edge_split(
                    g.edges[canonical_etype].data[msk],
                    pb, etype=canonical_etype, force_even=True)
            else:
                # If there are no training masks, we assume all edges can be used for training.
                # Therefore, we use a more memory efficient way to split the edge list.
                # TODO(zhengda) we need to split the edges properly to increase the data locality.
                train_idx = split_full_edge_list(g, canonical_etype, get_rank())

            assert train_idx is not None, \
                f"The training set of the {canonical_etype} edges is " \
                "empty or the number of training edges is smaller than " \
                f"the number of trainers {get_world_size()}" \
                "Please check your training data and make sure " \
                "the number of trainers (GPUs, if using GPU-supported " \
                "machines) is smaller than the number of training data."
            num_train += len(train_idx)
            train_idxs[canonical_etype] = train_idx

            logging.debug('part %d | etype %s, mask %s | train: %d',
                          get_rank(), canonical_etype, mask, len(train_idx))
        logging.info('part %d, train %d', get_rank(), num_train)
        return train_idxs

    def _get_edge_set(self, etypes, mask, reverse_edge_types_map):
        """ called by get_edge_val_set and get_edge_test_set
        """
        g = self._g
        pb = g.get_partition_book()
        idxs = {}
        num_data = 0
        etypes = self._exclude_reverse_etype(g.canonical_etypes, reverse_edge_types_map) \
            if etypes is None else self._check_etypes(etypes)
        masks = self._check_edge_mask(etypes, mask)

        for canonical_etype, msk in zip(etypes, masks):
            # user must provide validation/test mask
            if msk in g.edges[canonical_etype].data:
                idx = dgl.distributed.edge_split(
                    g.edges[canonical_etype].data[msk],
                    pb, etype=canonical_etype, force_even=True)
                idx = [] if idx is None else idx
                num_data += len(idx)
                # If there are validation data globally, we should add them to the dict.
                total_num_idx = dist_sum(len(idx))
                if total_num_idx > 0:
                    if total_num_idx >= get_world_size():
                        # The size of the validation or test set is larger
                        # than the world size. Each validation/test dataloader
                        # will not be empty
                        idxs[canonical_etype] = idx
                    else:
                        # There is not enough validation or test data.
                        # One or more validation/test dataloader will be
                        # empty, which will cause an evaluation error.
                        #
                        # To avoid the error, force each trainer or
                        # inferencer to use the entire validation
                        # or test set.
                        idx = th.nonzero(g.edges[canonical_etype].data[msk][\
                            th.arange(g.num_edges(canonical_etype))]).reshape(-1,) # 1D tensor
                        idxs[canonical_etype] = idx
                        logging.warning("Since the total number of validation/test data"
                                        "of %s, which is %d, is less than the number of "
                                        "workers %d, we will force each worker to do "
                                        "validation or testing on the entire "
                                        "validation/test set.",
                                        canonical_etype, total_num_idx, get_world_size())

                logging.debug('part %d | etype %s, mask %s | val/test: %d',
                              get_rank(), canonical_etype, msk, len(idx))
        return idxs, num_data

    def get_edge_val_set(self, etypes=None, mask="val_mask",
                         reverse_edge_types_map=None):
        """ Get the validation set for the given edge types under the given mask.

        Parameters
        __________
        etypes: list of str
            List of edge types to get the val set.
            If set to None, all the edge types are included.
        mask: str or list of str
            The edge feature field storing the val mask.
            Default: "val_mask".
        reverse_edge_types_map: dict
            A map for reverse edge types in the format of {(edge type):(reversed edge type)}.
            Default: None.

        Returns
        -------
        dict of Tensors : The returned validation edge indexes.
        """
        idxs, num_data = self._get_edge_set(etypes, mask, reverse_edge_types_map)
        logging.info('part %d, val %d', get_rank(), num_data)

        return idxs

    def get_edge_test_set(self, etypes=None, mask="test_mask",
                          reverse_edge_types_map=None):
        """ Get the test set for the given edge types under the given mask.

        Parameters
        __________
        etypes: list of str
            List of edge types to get the test set.
            If set to None, all the edge types are included.
        mask: str or list of str
            The edge feature field storing the test mask.
            Default: "test_mask".
        reverse_edge_types_map: dict
            A map for reverse edge types in the format of {(edge type):(reversed edge type)}.
            Default: None.

        Returns
        -------
        dict of Tensors : The returned test edge indexes.
        """
        idxs, num_data = self._get_edge_set(etypes, mask, reverse_edge_types_map)
        logging.info('part %d, test %d', get_rank(), num_data)

        return idxs

    def get_edge_infer_set(self, etypes=None, mask="test_mask", reverse_edge_types_map=None):
        """ Get the inference set for the given edge types under the given mask.

        If the mask exists in ``g.edges[etype].data``, the inference set
        is collected based on the mask.
        If not exist, the entire edge set are treated as the inference set.

        Parameters
        __________
        etypes: list of str
            List of edge types to get the inference set.
            If set to None, all the edge types are included.
            Default: None.
        mask: str or list of str
            The edge feature field storing the inference mask.
            Default: "test_mask".
        reverse_edge_types_map: dict
            A map for reverse edge types in the format of {(edge type):(reversed edge type)}.
            Default: None.

        Returns
        -------
        dict of Tensors : The returned inference edge indexes.
        """
        g = self._g
        pb = g.get_partition_book()
        infer_idxs = {}
        # If etypes is None, we use all edge types.
        etypes = self._exclude_reverse_etype(g.canonical_etypes, reverse_edge_types_map) \
            if etypes is None else self._check_etypes(etypes)
        masks = self._check_edge_mask(etypes, mask)

        for canonical_etype, msk in zip(etypes, masks):
            if msk in g.edges[canonical_etype].data:
                # mask exists
                test_idx = dgl.distributed.edge_split(
                    g.edges[canonical_etype].data[msk],
                    pb, etype=canonical_etype, force_even=True)
                # If there are test data globally, we should add them to the dict.
                if test_idx is not None and dist_sum(len(test_idx)) > 0:
                    infer_idxs[canonical_etype] = test_idx
            else:
                # mask does not exist
                # we will do inference on the entire edge set
                if get_rank() == 0:
                    logging.info("We will do inference on the entire edge set of %s.",
                            str(canonical_etype))
                infer_idx = dgl.distributed.edge_split(
                    th.full((g.num_edges(canonical_etype),), True, dtype=th.bool),
                    pb, etype=canonical_etype, force_even=True)
                infer_idxs[canonical_etype] = infer_idx

        return infer_idxs

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
