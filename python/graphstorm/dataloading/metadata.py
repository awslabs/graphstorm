"""
    Copyright Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Graph Metadata classes and graph from metadata classes.
"""

import logging
from abc import ABC, abstractmethod

import torch as th
from dgl.distributed.constants import DEFAULT_ETYPE, DEFAULT_NTYPE

METADATA_VERSION = 'gsgraph_metadata_v0.1'
HOMO_GRAPH_TYPE = 'homogeneous'
HETE_GRAPH_TYPE = 'heterogeneous'
SUPPORT_GPAPH_TYPES = [HOMO_GRAPH_TYPE, HETE_GRAPH_TYPE]

# ============ Graph Metadata Classes ============ #
class GSGraphMetadata():
    """ Metadata class for saving and retrieving graph information.

    .. versionadded:: 0.5.0
        Add ``GSGraphMetadata`` in 0.5.0 to build lightweight graph APIs that store metadata of
        graphs.

    Graph metadata contains general information of a graph, such as graph types, node types,
    edge types, and features. Such information could be used to generate lightweight graph
    instances without read in actual data, i.e., DistDGLGraph or DGLGraph, or to work as a
    source for graph metadata retrieving.

    In addition, the graph metadata is agnostic to graph libraries, e.g., DGL and PyG.

    Parameters
    ----------
    gtype: str
        The type of the graph. Can only be "homogeneous" or "heterogeneous".
    ntypes: list of str or str
        The node types that are in the format of a list, e.g., ["ntype1", "ntype2", ...], or
        a string for one node type only.
    etypes: list of tuples or a tuple
        The canonical edge types (three strings in a tuple) that are in the format of a list
        of tuples, e.g., [("ntype1", "etype1", "ntype2"), ...], or a tuple for one edge type only.
    nfeat_dims: dict of dict, optional
        The node feature dimensions that are in the format of dictionary whose keys are node types,
        and keys are dictionaries too. These dictionaries' keys are feature names, and values are
        the dimensions of the corresponding features, e.g., {'ntype1': {'feat1': [14],
        'feat2':[12]}, 'ntype2': {'feat3': [4, 7]}}. Default is None.
    efeat_dims: dict of dict, optional
        The edge feature dimensions that are in the format of dictionary whose keys are canonical
        edge types, and keys are dictionaries too. These dictionaries' keys are feature names,
        and values are the dimensions of the corresponding features, e.g.,
        {("ntype1", "etype1", "ntype2"): {'feat1': [4], 'feat2':[7]},
         ("ntype2", "etype2", "ntype3"): {'feat': [7, 14]}}. Default is None.

    Note: The format of feature dimensions is a list of integers excluding the number of samples
          (the number of rows). For example, a node feature tensor has shape (100, 4, 7) where
          the first dimension, 100, is the number of nodes. Metadata will only use the 2nd and
          3rd dimensions and store them in the list, [4, 7], for this feature.
    """
    def __init__(self,
                 gtype,
                 ntypes,
                 etypes,
                 nfeat_dims=None,
                 efeat_dims=None):
        assert gtype in SUPPORT_GPAPH_TYPES, (
            f'Graph types can only be in {SUPPORT_GPAPH_TYPES}, but got {gtype}.')
        self._gtype = gtype

        assert (isinstance(ntypes, list) and all(isinstance(ntype, str) for ntype in ntypes)) \
            or isinstance(ntypes, str), (
                f'Node types should be in a list of strings or a single string, but got {ntypes}.')
        if isinstance(ntypes, str):
            self._ntypes = [ntypes]
        else:
            self._ntypes = ntypes

        assert (isinstance(etypes, list) and all(isinstance(can_etype, tuple) for can_etype in \
            etypes) and all(len(can_etype)==3 for can_etype in etypes)) or \
            (isinstance(etypes, tuple) and len(etypes)==3), (
                'Edge types should be in a list of tuples each of which '
                'has three strings to indicate source node type, edge type, '
                f'destination node type, or in a single tuple, but got {etypes}.')

        if isinstance(etypes, tuple):
            self._etypes = [etypes]
        else:
            self._etypes = etypes

        # sanity check of homogeneous graphs
        if self._gtype == HOMO_GRAPH_TYPE:
            assert len(self._ntypes) == 1 and len(self._etypes) == 1, (
                'For a homogeneous graph, number of node types and edge types should be 1, but '
                f'got {len(self._ntypes)} node types, and {len(self._etypes)} edge types.')

        # sanity check if node types in edge types exist in node type list
        for src_ntype, _, dst_ntype in self._etypes:
            assert (src_ntype in self._ntypes) and ((dst_ntype in self._ntypes)), (
                f'Some node types {src_ntype} and {dst_ntype} do not exist in the given node type '
                f'list: {self._ntypes}.')

        if nfeat_dims is not None:
            assert isinstance(nfeat_dims, dict), ('The node feature dimensions should be a ' \
                    f'dictionary, but got {nfeat_dims}.')
            for ntype, feat_dims in nfeat_dims.items():
                assert isinstance(ntype, str), ('The key of node feature dimensions should be ' \
                    f'a string, but got {ntype}.')
                assert isinstance(feat_dims, dict), ('The value of node feature dimension of ' \
                    f'{ntype} should be a dictionary, but got {feat_dims}.')
                for feat_name, dim_list in feat_dims.items():
                    assert isinstance(feat_name, str), ('The feature dimension object ' \
                        'should be a dictionary, whose keys are strings, but got ' \
                        f'the key {feat_name} for the {ntype} node.')
                    assert isinstance(dim_list, list), ('The feature dimension object ' \
                        'should be a dictionary, whose values are lists, but got the ' \
                        f'value {dim_list} for the {ntype} node.')
            # need to make sure the input argument format is correct first, and then can test
            # the contents
            assert all(ntype in self._ntypes for ntype in nfeat_dims.keys()), ('Some node ' \
                f'types in node feature dimensions: {nfeat_dims} are not in the node type ' \
                f'list: {ntypes}.')
        self._nfeat_dims = nfeat_dims

        if efeat_dims is not None:
            assert isinstance(efeat_dims, dict), ('The edge feature dimensions should be a ' \
                f'dictionary, but got {efeat_dims}.')
            for etype, feat_dims in efeat_dims.items():
                assert isinstance(etype, tuple) and len(etype)==3, ('The key of edge feature ' \
                    'dimension dictionary should be a tuple, which includes three ' \
                    'strings to indicate source node type, relation type, destination node ' \
                    f'type ), but got {etype}.')
                assert isinstance(feat_dims, dict), ('The value of edge feature dimension ' \
                    f'dictionary should be a dictionary, but got {feat_dims}.')
                for feat_name, dim_list in feat_dims.items():
                    assert isinstance(feat_name, str), ('The feature dimension object ' \
                        'should be a dictionary, whose keys are strings, but got the ' \
                        f'key {feat_name} for the {etype} edge.')
                    assert isinstance(dim_list, list), ('The feature dimension object ' \
                        'should be a dictionary, whose values are lists, but got the ' \
                        f'value {dim_list} for the {etype} edge.')
            # need to make sure the input argument format is correct first, and then can test
            # the contents
            assert all(etype in self._etypes for etype in efeat_dims.keys()), (
                f'Some edge types in edge feature dimensions: {efeat_dims} are not in the ' \
                f'edge type list: {etypes}.')
        self._efeat_dims = efeat_dims

    # getters
    def is_homogeneous(self):
        """ Check if the graph metadata is for a homogeneous graph.

        Returns
        -------
        bool: if the graph metadata is for a homogeneous graph.
        """
        return self._gtype == HOMO_GRAPH_TYPE

    def get_ntypes(self):
        """ Get node types.

        Returns
        -------
        list: graph node types in a list of strings.
        """
        return self._ntypes

    def has_ntype(self, ntype):
        """ Check if a node type exists in the graph metadata.

        Parameters
        ----------
        ntype: str
            The node type name.

        Returns
        -------
        bool: True if the graph metadata contains the given node type, false otherwise.
        """
        return ntype in self._ntypes

    def get_etypes(self):
        """ Get edge types.

        Returns
        -------
        list
            graph node types in a list of 3-element tuples.
        """
        return self._etypes

    def has_etype(self, etype):
        """ Check if an edge type exists in the graph metadata.

        Parameters
        ----------
        etype: (str, str, str)
            The edge type name in a three element tuple.

        Returns
        -------
        bool: True if the graph metadata contains the given edge type, false otherwise.
        """
        return etype in self._etypes

    def get_nfeat_all_dims(self, ntype):
        """ Get all feature dimensions of the given node type.

        Parameters
        ----------
        ntype: str
            The node type name.

        Returns
        -------
        dict or None: the feature dimensions of the given ntype in the format of a dictionary, or
                      None if either the graph metadata has no node feature or the given ntype has
                      no feature.
        """
        if self._nfeat_dims is not None:
            nfeat_all_dims = self._nfeat_dims.get(ntype, None)
        else:
            nfeat_all_dims = None

        return nfeat_all_dims

    def get_efeat_all_dims(self, etype):
        """
        Parameters
        ----------
        etype: (str, str, str)
            The edge type name in a three element tuple.

        Returns
        -------
        list or None: the feature dimensions of the given etype in the format of a dictionary, or
                      None if either the graph metadata has no edge feature or the given etype has
                      no feature.
        """
        if self._efeat_dims is not None:
            efeat_all_dims = self._efeat_dims.get(etype, None)
        else:
            efeat_all_dims = None

        return efeat_all_dims

    # output functions
    def to_dict(self):
        """ Convert the graph metadata into a dictionary.

        Convert this metadata instance into a dictionary, which can be used to persist graph
        metadata in various formats, e.g, JSON, YAML, and etc.

        The dictionary will be like:
        {
            "version": "metadata_v0.1",
            "graph_type": "heterogeneous",
            "nodes": [
                {
                    "node_type": "ntype1",
                    "features": [{
                        "feat_name": "feat1",
                        "feat_dim": [4, 7]
                    },
                                {
                        "feat_name": "feat1",
                        "feat_dim": [32]
                    }
                    ]
                },
                {
                    "node_type": "ntype2",
                    "features": []
                }
            ],
            "edges": [
                {
                    "source_node_type": "ntype1",
                    "edge_type": "etype1",
                    "destination_node_type": "ntype2",
                    "features": [
                        {
                            "feat_name": "feat",
                            "feat_dim": [3]
                        }
                    ]
                }
            ]
        }

        Returns
        -------
        metadata_dict: dict
            The hierarchy structure of the graph metadata like the example above.
        """
        metadata_dict = {"version": METADATA_VERSION}
        metadata_dict = {"graph_type": self._gtype}
        nodes = []
        for ntype in self._ntypes:
            node = {"node_type": ntype}
            if self._nfeat_dims:
                if ntype in self._nfeat_dims:
                    nfeats = []
                    for feat_name, feat_dim in self._nfeat_dims[ntype].items():
                        nfeat_dim = {
                            "feat_name": feat_name,
                            "feat_dim": feat_dim
                            }
                        nfeats.append(nfeat_dim)
                    node['features'] = nfeats
            else:
                node['features'] = []

            nodes.append(node)

        metadata_dict["nodes"] = nodes

        edges = []
        for src_ntype, etype, dst_ntype in self._etypes:
            edge = {
                "source_node_type": src_ntype,
                "etype": etype,
                "destination_node_type": dst_ntype
            }
            if self._efeat_dims:
                if (src_ntype, etype, dst_ntype) in self._efeat_dims:
                    efeats = []
                    for feat_name, feat_dim in \
                        self._efeat_dims[(src_ntype, etype, dst_ntype)].items():
                        efeat_dim = {
                            "feat_name": feat_name,
                            "feat_dim": feat_dim
                            }
                        efeats.append(efeat_dim)
                    edge['features'] = efeats
            else:
                edge['features'] = []

            edges.append(edge)

        metadata_dict["edges"] = edges

        return metadata_dict

    def __repr__(self) -> str:
        """Formal object representation for debugging

        Returns
        -------
        str:
            A reader-friendly repsentation string of contents of an GSGraphMetadata instance.
        """
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def __str__(self) -> str:
        """Informal object representation for readability

        Display a dictionary of the metadata attributes.

        Returns
        -------
        str:
            A string with dictionary foramt, representing contents of an GSGraphMetadata instance.
        """
        metadata_dict = self.to_dict()
        return str(metadata_dict)


# ============ Metadata Graph Classes ============ #
class GSGraphFromMetadata(ABC):
    """ Abstract class as the base of graphs built from metadata.

    .. versionadded:: 0.5.0
        Add ``GSGraphFromMetadata`` in 0.5.0 to support lightweight graph construction using
        graph metadata only.

    Parameters
    ----------
    graph_metadata: GSGraphMetadata
        An instance of GSGraphMetadata class.
    """
    def __init__(self, graph_metadata: GSGraphMetadata):
        self._graph_metadata = graph_metadata

    @property
    @abstractmethod
    def ntypes(self):
        """ Get the node type list property
        """

    @property
    @abstractmethod
    def etypes(self):
        """ Get the edge type list property.
        """

    def gtype(self):
        """ Get the metadata graph type.

        Returns
        -------
        str:
            The graph type. One of 'homogeneous' or 'heterogeneous'.
        """
        return self._graph_metadata._gtype

    @abstractmethod
    def device(self):
        """ Get the device where metadata graph is located.
        """


class GSDglGraphFromMetadata(GSGraphFromMetadata):
    """ A graph from metadata implementation for simulating DGLGraph APIs

    .. versionadded:: 0.5.0
        Add ``GSDglGraphFromMetadata`` in 0.5.0 to support lightweight DGL graph construction
        using graph metadata only.

    """
    def __init__(self,
                 graph_metadata,
                 device='cpu'):
        super().__init__(graph_metadata)
        self._device = device

        # check homogeneous and convert node and edge node type string to DGL's default name
        if self._graph_metadata.is_homogeneous():
            # convert node type and edge type to DGL's default name, `_N` and (`_N`, `_E`, `_N`)
            homo_ntype = self._graph_metadata.get_ntypes()[0]
            homo_etype = self._graph_metadata.get_etypes()[0]
            self._ntypes = [DEFAULT_NTYPE]
            self._etypes = [DEFAULT_ETYPE]
        else:
            self._ntypes = self._graph_metadata.get_ntypes()
            self._etypes = self._graph_metadata.get_etypes()

        # create internal DGL nodes and edges data properties
        self._nodes = {}
        self._edges = {}

        for ntype in self._ntypes:
            self._nodes[ntype] = DglDataViewSimulation()

            # if the metadata include node feature information, create an empty tensor
            # according to the feature dimensions.
            if self._graph_metadata.is_homogeneous():
                if self._graph_metadata.get_nfeat_all_dims(homo_ntype) is not None:
                    nfeat_all_dims = self._graph_metadata.get_nfeat_all_dims(homo_ntype)

                    for nfeat_name, dims in nfeat_all_dims.items():
                        dims.insert(0, 0)
                        self._nodes[ntype].data[nfeat_name] = th.empty(dims, device=device)
            else:
                if self._graph_metadata.get_nfeat_all_dims(ntype) is not None:
                    nfeat_all_dims = self._graph_metadata.get_nfeat_all_dims(ntype)

                    for nfeat_name, dims in nfeat_all_dims.items():
                        dims.insert(0, 0)
                        self._nodes[ntype].data[nfeat_name] = th.empty(dims, device=device)

        for can_etype in self._etypes:
            self._edges[can_etype] = DglDataViewSimulation()

            # if the metadata include edge feature information, create an empty tensor
            # according to the feature dimensions.
            if self._graph_metadata.is_homogeneous():
                if self._graph_metadata.get_efeat_all_dims(homo_etype) is not None:
                    efeat_all_dims = self._graph_metadata.get_efeat_all_dims(homo_etype)
                    for efeat_name, dims in efeat_all_dims.items():
                        dims.insert(0, 0)
                        self._edges[can_etype].data[efeat_name] = th.empty(dims, device=device)
            else:
                if self._graph_metadata.get_efeat_all_dims(can_etype) is not None:
                    efeat_all_dims = self._graph_metadata.get_efeat_all_dims(can_etype)
                    for efeat_name, dims in efeat_all_dims.items():
                        dims.insert(0, 0)
                        self._edges[can_etype].data[efeat_name] = th.empty(dims, device=device)


    # Implementations of abstract methods
    @property
    def ntypes(self):
        """ Get the node type list.
        """
        return self._ntypes

    @property
    def etypes(self):
        """ Get the edge type list.

        In DGL, an edge type is a string to specify the type of edge. But this string could be
        same across various edge types. So it is recommended to use canonical edge type to give
        a unique edge type identifier.
        """
        etypes = []
        for can_etype in self._etypes:
            etypes.append(can_etype[1])

        return etypes

    def device(self):
        """ Get the device where metadata graph is located.

        """
        return self._device

    # Implementations of DGL specific APIs
    @property
    def canonical_etypes(self):
        """ Get the canonical edge type list.

        A canonical edge type is a three element tuple, where the first element is the source
        node type, the second element is the edge type, and the third element is the destination
        node type.
        """
        return self._etypes

    @property
    def nodes(self):
        """ Provide an interface of metadata graph nodes

        DGL's `nodes` property is a NodeView class for node data retrieval. This metadata DGL
        graph cannot provide the NodeView class, but can provide its node type data retrieval
        interface, i.e., g.nodes[ntype].data and g.nodes[ntype].data[nfeat_name].

        Returns
        ------
        dict : the dictionary of nodes and their DataView classes.
        """

        logging.warning(('This %s is a metadata graph that simulates a DGL graph without '
            'actual graph structure, i.e., nodes and edges, but meta information, e.g., node '
            'types, edge types, and node or edge feature names. You can use the '
            '".nodes[ntype].data" or ".nodes[ntype].data[efeat_name]" interface to retrieve '
            'node data information.'),
            self.__class__.__name__)
        return self._nodes

    @property
    def edges(self):
        """ Provide an interface of metadata graph edges

        DGL's `edges` property is an EdgeView class for edge data retrieval. This metadata DGL
        graph cannot provide the EdgeView class, but can provide its edge type data retrieval
        interface, i.e., g.edges[etype].data and g.edges[etype].data[efeat_name].

        Returns
        -------
        dict : the dictionary of edges and their DataView classes.
        """
        logging.warning(('This %s is a metadata graph that simulates a DGL graph without '
            'actual graph structure, i.e., nodes and edges, but meta information, e.g., node '
            'types, edge types, and node or edge feature names. You can use the '
            '".edges[etype].data" or ".edges[etype].data[efeat_name]" interface to retrieve '
            'edge data information.'),
            self.__class__.__name__)
        return self._edges

    def is_homogeneous(self):
        """ Check if the metadata graph is a homogeneous one.
        """
        return self._graph_metadata.is_homogeneous()

    @property
    def ndata(self):
        """ Provide an interface for retrieving node data in a homogeneous metadata graph.

        This interface is for homogeneous graphs only, and the node type should be `_N`, which is
        defined by dgl's DEFAULT_NTYPE constant.
        """
        if not self.is_homogeneous():
            logging.warning(
                'Only homogeneous metadata graphs support the "ndata" interface, but got a '
                'heterogeneous metadata graph.')
            return {}
        else:
            return self._nodes[DEFAULT_NTYPE].data

    @property
    def edata(self):
        """ Provide an interface for retrieving edge data in a homogeneous metadata graph.

        This interface is for homogeneous graphs only, and the edge type should be
        (`_N`, `_E`, `_N`), which is defined by DGL constants.
        """
        if not self.is_homogeneous():
            logging.warning(
                'Only homogeneous metadata graphs support the "edata" interface, but got a '
                'heterogeneous metadata graph.')
            return {}
        else:
            return self._edges[DEFAULT_ETYPE].data


class DglDataViewSimulation():
    """ Simulation of the DGL NodeView and EdgeView class.

    .. versionadded:: 0.5.0
        Add ``DglDataViewSimulation`` in 0.5.0 to support real-time inference on Amazon SageMaker
        endpoints.

    This class provides a simple interface of DGL NodeView/EdgeView class, i.e., it only provides
    the `data` interface for retrieving node/edge feature data.

    The `data` property is a dictionary to host feature names with their corresponding tensors.
    """
    def __init__(self):
        self._data = {}

    @property
    def data(self):
        """ Get the data property for setting and getting
        """
        return self._data


class GSDglDistGraphFromMetadata(GSDglGraphFromMetadata):
    """ A metadata graph implementation for DGL distributed graph APIs.

    .. versionadded:: 0.5.0
        Add ``GSMetadataDglDistGraph`` in 0.5.0 to support lightweight DGL distributed graph
        construction using graph metadata only.

    This metadata graph only provides two DGL distributed graph APIs, i.e.,

    - get_node_partition_policy(ntype)
    - get_partition_book()

    However, as DGL distributed graph requires actual graph structure data, which the metadata
    cannot provide. Therefore, this metadata graph will raise NotImplementedError to let
    callers know this metadata graph cannot be used for DGL distributed graph.
    """
    # provides the two distributed graph APIs
    def get_node_partition_policy(self, ntype):
        """ Retrieve the node partition policy of the given node type.

        Not implemented.
        """
        raise NotImplementedError(
            f'The {self.__class__.__name__} class is a metadata graph to simulated a DGL '
            'distributed graph for special use cases, e.g., real-time inference model reloading. '
            f'It does not provide the "get_node_partition_policy" with given {ntype} interface.')

    def get_partition_book(self):
        """ Retrieve the overall node partition policy of distributed graph.

        Not implemented.
        """
        raise NotImplementedError(
            f'The {self.__class__.__name__} class is a metadata graph to simulated a DGL '
            'distributed graph for special use cases, e.g., real-time inference model reloading. '
            'It does not provide the "get_partition_book" interface.')

# ============ Metadata Graph Utilities ============ #
def config_json_sanity_check(config_json):
    """ Function to check configuration JSON object for metadata creation.

    .. versionadded:: 0.5.0
        Add ``config_json_sanity_check`` in 0.5.0 to support real-time inference.

    This function extends gconstruct's verify_configs() function in three major perspectives,
    and it is more rigorous than the verify_configs() to make sure contents are correct and
    can be converted to metadata.
    1. Check JSON from both gconstruct and GSProcessing.
    2. Require "version" field.
    3. Require "feature_dim" for any element in the "features" field if has "features".

    JSON from GS gconstruct:
        {
            "version": "gconstruct-runtime-v0.1",
            "is_homogeneous": bool,
            "add_reverse_edges": bool,
            "nodes":[
                {
                    "node_id_col": str,
                    "node_type": str,
                    "features": [
                        {
                            "feature_col": str
                            "feature_name": str
                            "feature_dim": list[int]
                        }
                    ]
                }
            ]
            "edges":[
                {
                    "source_id_col": str,
                    "dest_id_col": str,
                    "relation": [str, str, str],
                    "features": [
                        {
                            "feature_col": str
                            "feature_name": str
                            "feature_dim": list[int]
                        }
                    ]
                }
            ]
        }
    JSON from GSProcessing:
        {
            "version": "gsprocessing-runtime-v0.4.1",
            "graph": {
                "is_homogeneous": bool,
                "add_reverse_edges": bool,
                "nodes": [
                    {
                        "type": str,
                        "column": str,
                        "data": {
                            "format": str,
                            "files": list[str]
                        },
                        "features": [
                            {
                                "column": str,
                                "name": str,
                                "dim": list[int],
                            }
                        ]
                    }
                ]
                "edges": [
                    {
                        "source": {"column": str, "type": str},
                        "dest": {"column": str, "type": str},
                        "relation": {"type": str},
                        "data": {
                            "format": str,
                            "files": list[str]
                        },
                        "features": [
                            {
                                "column": str,
                                "name": str,
                                "dim": list[int],
                            }
                        ]
                    }
                ]
            }
        }

    Note: this function assumes the JSON object should come from GS own code, and the "version"
          field should exist, which is used to identify if the JSON object is from gconstruct or
          GSProcessing.

    If the config JSON object missing required fields, will raise AssertionError.
    """
    assert 'version' in config_json, (
        'A "version" field must be defined in the configuration JSON object.')
    config_version = config_json['version']
    assert 'runtime' in config_version, (
        'The value of the "version" field must contain "runtime" to identify '
        'it is a runtime version.')

    if config_version.startswith('gconstruct'):
        assert 'is_homogeneous' in config_json, (
            'A "is_homogeneous" field must be defined in the configuration JSON object.')
        is_homo = config_json['is_homogeneous']
        assert is_homo in [True, "True", 'true', False, "False", 'false'], (
            'The value of "is_homogeneous" can only be "True", "true", "False", or '
            f'"false", but got {is_homo}.')

        assert "add_reverse_edges" in config_json, (
            'An "add_reverse_edges" field must be defined in the configuration JSON object.'
        )
        add_reverse_edges = config_json["add_reverse_edges"]
        assert add_reverse_edges in [True, "True", 'true', False, "False", 'false'], (
            'The value of "add_reverse_edges" can only be "True", "true", "False", or '
            f'"false", but got {add_reverse_edges}.')

        assert 'nodes' in config_json, (
            'A "nodes" field must be defined in the configuration JSON object.')
        assert len(config_json['nodes']) > 0, 'Need at least one node in the \"nodes\" object.'
        ntypes = []
        for node_obj in config_json['nodes']:
            assert 'node_type' in node_obj, (
                'A "node_type" field must be defined in a node object under the "nodes" field.')
            ntypes.append(node_obj['node_type'])

            if 'features' in node_obj:
                for feat_obj in node_obj['features']:
                    assert 'feature_name' in feat_obj, (
                        'A "feature_name" field must be defined in a feature object.')
                    assert 'feature_dim' in feat_obj, (
                        'A "feature_dim" field must be defined in a feature object.')
                    feat_dim = feat_obj['feature_dim']
                    assert isinstance(feat_dim, list), (
                        f'Values of "feature_dim" field must be a list, but got {feat_dim}.')
        # check duplicates in node types
        assert len(ntypes) == len(set(ntypes)), (
            f'There are duplicated node types in the nodes object: {ntypes}.')

        assert 'edges' in config_json, (
            'An "edges" field must be defined in the configuration JSON object.')
        assert len(config_json['edges']) > 0, 'Need at least one edge in the \"edges\" object.'
        etypes = []
        for edge_obj in config_json['edges']:
            assert 'relation' in edge_obj, (
                'A "relation" field must be defined in an edge object.')
            etypes.append(tuple(edge_obj['relation']))

            if 'features' in edge_obj:
                for feat_obj in edge_obj['features']:
                    assert 'feature_name' in feat_obj, (
                        'A "feature_name" field must be defined in a feature object.')
                    assert 'feature_dim' in feat_obj, (
                        'A "feature_dim" field must be defined in a feature object.')
                    feat_dim = feat_obj['feature_dim']
                    assert isinstance(feat_dim, list), (
                        f'Values of "feature_dim" field must be a list, but got {feat_dim}.')
        # check duplicates in edge types
        assert len(etypes) == len(set(etypes)), (
            f'There are duplicated edge types in the edges object: {etypes}.')
    elif config_version.startswith('gsprocessing'):
        assert 'graph' in config_json, (
            'A "graph" field must be defined in the configuration JSON object.')
        graph_obj = config_json['graph']

        assert 'is_homogeneous' in graph_obj, (
            'An "is_homogeneous" field must be defined in the configuration JSON object.')
        is_homo = graph_obj['is_homogeneous']
        assert is_homo in [True, "True", 'true', False, "False", 'false'], (
            'The value of "is_homogeneous" can only be "True", "true", "False", or '
            f'"false", but got {is_homo}.')

        assert 'add_reverse_edges' in graph_obj, (
            'An "add_reverse_edges" field must be defined in the configuration JSON object.')
        add_reverse_edges = graph_obj['add_reverse_edges']
        assert add_reverse_edges in [True, "True", 'true', False, "False", 'false'], (
            'The value of "add_reverse_edges" can only be "True", "true", "False", or '
            f'"false", but got {add_reverse_edges}.')

        assert 'nodes' in graph_obj, (
            'A "nodes" field must be defined in the graph object.')
        assert len(graph_obj['nodes']) > 0, 'Need at least one node in the \"nodes\" object.'

        ntypes = []
        for node_obj in graph_obj['nodes']:
            assert 'type' in node_obj, (
                'A "type" field must be defined in the node object.')
            ntypes.append(node_obj['type'])

            if 'features' in node_obj:
                for feat_obj in node_obj['features']:
                    assert 'name' in feat_obj, (
                        'A "name" field must be defined in a feature object.')
                    assert 'dim' in feat_obj, (
                        'A "dim" field must be defined in a feature object.')
                    feat_dim = feat_obj['dim']
                    assert isinstance(feat_dim, list), (
                        f'Values of "dim" field must be a list, but got {feat_dim}.')
        # check duplicates in node types
        assert len(ntypes) == len(set(ntypes)), (
            'There are duplicated node types in the nodes object: {ntypes}.')

        assert 'edges' in graph_obj, (
            'An "edges" field must be defined in the graph object.')
        assert len(graph_obj['edges']) > 0, 'Need at least one edge in the \"edges\" object.'

        etypes = []
        for edge_obj in graph_obj['edges']:
            assert 'source' in edge_obj, (
                'A "source" field must be defined in the edge object.')
            assert 'type' in edge_obj['source'], (
                'A "type" field must be defined in the source object.')

            assert 'dest' in edge_obj, (
                'A "dest" field must be defined in the edge object.')
            assert 'type' in edge_obj['dest'], (
                'A "type" field must be defined in the dest object.')

            assert 'relation' in edge_obj, (
                'A "relation" field must be defined in the edge object.')
            assert 'type' in edge_obj['relation'], (
                'A "type" field must be defined in the relation object.')

            etypes.append((edge_obj['source']['type'], edge_obj['relation']['type'], \
                           edge_obj['dest']['type']))

            if 'features' in edge_obj:
                for feat_obj in edge_obj['features']:
                    assert 'name' in feat_obj, (
                        'A "name" field must be defined in a feature object.')
                    assert 'dim' in feat_obj, (
                        'A "dim" field must be defined in a feature object.')
                    feat_dim = feat_obj['dim']
                    assert isinstance(feat_dim, list), (
                        f'Values of "dim" field must be a list, but got {feat_dim}.')
        # check duplicates in edge types
        assert len(etypes) == len(set(etypes)), (
            'There are duplicated edge types in the edges object: {etypes}.')
    else:
        raise NotImplementedError(
            'GSGraphMetadata can only be loaded from the JSON file generated by GraphStorm '
            'gconstruct or GSProcessing.')


def load_metadata_from_json(config_json):
    """ Create a GSGraphMetadata from a JSON object.

    .. versionadded:: 0.5.0
        Add ``load_metadata_from_json`` in 0.5.0 to support real-time inference on Amazon
        SageMaker endpoints.

    This utility function convert the metadata JSON object to a GSGraphMetadata instance. The
    metadata JSON object comes from the GS gconstruct or GSProcessing command, which outputs the
    graph information after graph construction and feature transformation.

    Below are example contents of GS construction (v0.1) and GSProcessing on (v0.4.1).

    JSON from GS gconstruct:
        {
            "version": "gconstruct-runtime-v0.1",
            "is_homogeneous": bool,
            "add_reverse_edges": bool,
            "nodes":[
                {
                    "node_id_col": str,
                    "node_type": str,
                    "features": [
                        {
                            "feature_col": str
                            "feature_name": str
                            "feature_dim": list[int]
                        }
                    ]
                }
            ]
            "edges":[
                {
                    "source_id_col": str,
                    "dest_id_col": str,
                    "relation": [str, str, str],
                    "features": [
                        {
                            "feature_col": str
                            "feature_name": str
                            "feature_dim": list[int]
                        }
                    ]
                }
            ]
        }
    JSON from GSProcessing:
        {
            "version": "gsprocessing-runtime-v0.4.1",
            "graph": {
                "is_homogeneous": bool,
                "add_reverse_edges": bool,
                "nodes": [
                    {
                        "type": str,
                        "column": str,
                        "data": {
                            "format": str,
                            "files": list[str]
                        },
                        "features": [
                            {
                                "column": str,
                                "name": str,
                                "dim": list[int],
                            }
                        ]
                    }
                ]
                "edges": [
                    {
                        "source": {"column": str, "type": str},
                        "dest": {"column": str, "type": str},
                        "relation": {"type": str},
                        "data": {
                            "format": str,
                            "files": list[str]
                        },
                        "features": [
                            {
                                "column": str,
                                "name": str,
                                "dim": list[int],
                            }
                        ]
                    }
                ]
            }
        }

    Parameters
    ----------
    config_json: dict
        A Python dict containing graph configuration information. The JSON object may come from
        either GraphStorm gconstruct or GSProcessing commands as a configuration JSON file.

    Returns
    --------
    gs_metadata: GSGraphMetadata
        A GraphStorm GSGraphMetadata class instance.
    """
    # do sanity check first, making sure no errors in the JSON object
    config_json_sanity_check(config_json)

    # check which tool generates this JSON object
    config_version = config_json['version']

    if config_version.startswith('gconstruct'):
        is_homo = config_json['is_homogeneous'] in ['True', 'true', True]

        # parse node types
        ntypes = []
        nfeat_dims = {}
        for node_obj in config_json['nodes']:
            ntypes.append(node_obj['node_type'])
            # extract feature name and dimensions if have
            if 'features' in node_obj:
                feat_dims = {}
                for feat_obj in node_obj['features']:
                    feat_dims[feat_obj['feature_name']] = feat_obj['feature_dim']
                nfeat_dims[node_obj['node_type']] = feat_dims

        # parse edge types
        etypes = []
        efeat_dims = {}
        add_reverse_edges = config_json['add_reverse_edges'] in ['True', 'true', True]
        for edge_obj in config_json['edges']:
            etypes.append(tuple(edge_obj['relation']))  # convert a list to tuple as can_etype
            # extract feature name and dimensions if have
            if 'features' in edge_obj:
                feat_dims = {}
                for feat_obj in edge_obj['features']:
                    feat_dims[feat_obj['feature_name']] = feat_obj['feature_dim']
                efeat_dims[tuple(edge_obj['relation'])] = feat_dims
            if add_reverse_edges and not is_homo:
                reverse_relation = [edge_obj['relation'][2],
                                    edge_obj['relation'][1] + "-rev",
                                    edge_obj['relation'][0]]
                etypes.append(tuple(reverse_relation))
                if 'features' in edge_obj:
                    feat_dims = {}
                    for feat_obj in edge_obj['features']:
                        feat_dims[feat_obj['feature_name']] = feat_obj['feature_dim']
                    efeat_dims[tuple(reverse_relation)] = feat_dims
    else:
        graph_obj = config_json['graph']
        is_homo = graph_obj['is_homogeneous'] in ['True', 'true', True]

        # parse node types
        ntypes = []
        nfeat_dims = {}
        for node_obj in graph_obj['nodes']:
            ntypes.append(node_obj['type'])
            # extract feature name and dimensions if they exist
            if 'features' in node_obj:
                feat_dims = {}
                for feat_obj in node_obj['features']:
                    feat_dims[feat_obj['name']] = feat_obj['dim']
                nfeat_dims[node_obj['type']] = feat_dims

        # parse edge types
        etypes = []
        efeat_dims = {}
        add_reverse_edges = graph_obj['add_reverse_edges'] in ['True', 'true', True]
        for edge_obj in graph_obj['edges']:
            src_ntype = edge_obj['source']['type']
            dst_ntype = edge_obj['dest']['type']
            etype = edge_obj['relation']['type']
            etypes.append((src_ntype, etype, dst_ntype))  # create a tuple as can_etype
            # extract feature name and dimensions if they exist
            if 'features' in edge_obj:
                feat_dims = {}
                for feat_obj in edge_obj['features']:
                    feat_dims[feat_obj['name']] = feat_obj['dim']
                efeat_dims[(src_ntype, etype, dst_ntype)] = feat_dims
            if add_reverse_edges and not is_homo:
                etypes.append((dst_ntype, etype + "-rev", src_ntype))
                if 'features' in edge_obj:
                    feat_dims = {}
                    for feat_obj in edge_obj['features']:
                        feat_dims[feat_obj['name']] = feat_obj['dim']
                    efeat_dims[(dst_ntype, etype + "-rev", src_ntype)] = feat_dims

    # create the metadata instance
    if is_homo:
        gtype = HOMO_GRAPH_TYPE
    else:
        gtype = HETE_GRAPH_TYPE

    graph_metadata = GSGraphMetadata(gtype=gtype,
                                     ntypes=ntypes,
                                     etypes=etypes,
                                     nfeat_dims=nfeat_dims,
                                     efeat_dims=efeat_dims)
    return graph_metadata
