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

    MetadataSchema class is designed to encapsulate information present in
    the metadata schema for the input dataset.
"""
import json
import os

import numpy as np

from .constants import (
    STR_DATA,
    STR_EDGE_DATA,
    STR_EDGE_TYPE,
    STR_EDGES,
    STR_FORMAT,
    STR_FORMAT_DELIMITER,
    STR_GRAPH_NAME,
    STR_NAME,
    STR_NODE_DATA,
    STR_NODE_TYPE,
    STR_NUM_EDGES_PER_TYPE,
    STR_NUM_NODES_PER_TYPE,
)


class MetadataSchema:
    """MetadataSchema class is designed to capture and encapsulate all the
    information present in the metadata schema file of the input graph
    dataset. This class reads the input metadata schema from the input
    file and builds necessary data structures which are used during the
    normal processing of the data pre-processing pipeline.

    Metadata schema of the input graph contains the following key-values,
    some of which are mandatory and others are optional.
        *``graph_name``[mandatory], a string for the graph name,
        * ``num_nodes_per_type``[mandatory], a list of integers indicating
                the no. of nodes for a given node type,
        * ``num_edges_per_type``[mandatory], a list of integers indicating
                the no. of edges for a given edge type,
        * ``node_type``[mandatory], a list of strings each of which is the
                name of a node type in the graph,
        * ``edge_type``[mandatory], a list of strings each of which is the
                name of a edge type in the graph,
        * ``edges``[mandatory], a dictionary which contains the metadata
                for the edges in the graph. This dictionary has the following
                key-value pairs,
                { ``format``: {
                                ``name``: string,
                                    # valid values are [``csv``, ``numpy``]
                                ``delimiter``: string,
                                    # valid delimiter charactor.
                              }
                  ``data``: [file1, file2, file3, ....]
                        # list of file paths in which edges are stored.
                }
        * ``node_data``[optional], a dictionary which contains the metadata
                for the features associated with node types from the graph.
                The structure of this dictionary is as follows:
                {
                    ntype1: {
                        feature1: {
                            ``format``: {
                                ``name``: string,
                                    # valid values are [``numpy``, ``parquet``]
                                ``delimiter``: string
                                    # valid delimiter
                                }
                            ``data``: [file1, file2, file3, ....]
                                # list of file paths in which features for
                                # node type ntype1 are located.
                            }
                        feature2: {
                            ...
                            },
                        ...
                        },
                    ntype2: {
                        ...
                        },
                    ...

                }
            * ``edge_data``[optional], a dictionary which contains the metadata
                    for the features associated with edge types from the graph.
                    This dictionary is identical to the ``node_data``
                    dicionary, but it will describe the metadata which will
                    be used to locate files for edge features for edge types
                    from the input graph.

    A valid metadata.json file should adhere to the following rules.
        * ``graph_name`` should be a valid string with non-zero length,
        * ``num_nodes_per_type`` should be a list of positive non-zero integers
                and the length of list should be more than 0,
        * ``num_edges_per_type`` should be a list of positive non-zero integers
                and the length of list should be more than 0,
        * ``node_type`` should ba list of valid non-None strings of positive
                length, length of this list should be equal to the
                ``num_nodes_per_type`` list,
        * ``edge_type`` should ba list of valid non-None strings of positive
                length, length of this list should be equal to the
                ``num_edges_per_type`` list,
        * ``edges`` should have the same no. of keys are the no. of edge types
                present in the ``edge_type`` list,
        * ``node_data``, if present, can contain keys whose count is less than
                or equal to the node types present in the ``node_type`` list,
                Each node type in this dictionary is expected to non-zero
                features,
        * ``edge_data``, if present, follows the same rules as described for
                ``node_data`` in the earlier bullet point.

    Examples:
    ---------
    Creating and initializing the an instance of this class during the pipelines
    initialization process.
    >>> base_dir = "/home/ubuntu/graph-dataset"
    >>> file_path = "metadata.json"
    >>> metadata_obj = Metadata(file_name, base_dir)

    Once the object of this class is created it can be stored for retrieving any
    information from the Metadata file when needed.
    >>> graph_ntypes = metadata_obj.ntypes
    >>> print(graph_ntypes)
    >>> ["ntype1", "ntype2"]

    In the above code snipped ``ntypes`` property is invoked to get a list of
    node types present in the input graph, whose return value is a list of
    strings. The above code snippet assumes that the input graph has two
    node types namely ``ntype1`` and ``ntype2``.
    """

    def __init__(self, file_name, base_dir, init_maps=True):
        self._basedir = None
        self._metadata_path = None
        self._data = None
        self._global_nid_offsets = None
        self._global_eid_offsets = None
        self._ntypes = None
        self._ntype_id_map = None
        self._id_ntype_map = None
        self._etype_id_map = None
        self._id_etype_map = None
        self._etypes = None
        self._ntype_features = None
        self._etype_features = None
        self._ntype_feature_files = None
        self._etype_feature_files = None
        self._etype_files = None

        self.__load_json(file_name, base_dir)
        if init_maps:
            self.__init_maps()
            self.__validate_metadata()

    ############################## Private Methods #############################

    def __check_ifexists(self, data_files, file_type):
        """Helper function to verify the presence of a set of filenames.

        Argument
        --------
        data_files: list of str
            A set of file names, either in absolute or in relative formats.
        src_type: str
            Either node type or edge type from the input graph.
        """
        for file_name in data_files:
            file_path = file_name
            if not os.path.isabs(file_path):
                file_path = os.path.join(self._basedir, file_path)
            assert os.path.isfile(
                file_path
            ), f"File: {file_path} does not exist for {file_type}."

    def __init_maps(self):
        """Initialization function to build all the necessary data structures
        used during typical processing of the pipeline.
        """
        self._init_global_nid_offsets()
        self._init_global_eid_offsets()
        self._init_ntype_maps()
        self._init_etype_maps()
        self._init_ntype_features()
        self._init_etype_features()
        self._init_etype_files()
        self._init_ntype_feature_files()
        self._init_etype_feature_files()

    def __load_json(self, file_name, base_dir):
        """Helper function to read the metadata schema file of the input
        graph dataset. ``file_name`` input argument can be either in absolute
        or relative path formats. ``base_dir`` is used as the root directory
        for the input graph dataset where all the files (of the dataset) are
        expected to be located on the disk.

        Argument
        --------
        file_name: string
            Filename of the input metadata schema file. This can be either
            in absolute or relative path formats. If in relative format,
            ``base_dir`` is used as the root directory.
        base_dir: string
            Base directory, if provided, is used as the root directory for
            the input graph dataset
        """
        self._basedir = None
        if base_dir is None:
            self._basedir = os.getcwd()
        else:
            self._basedir = base_dir

        self._metadata_path = os.path.join(
            os.path.join(self._basedir, file_name)
        )
        with open(self._metadata_path, "r", encoding="utf-8") as handle:
            self._data = json.load(handle)

    def __validate_metadata(self):
        """Helper function to validate the input metadata schema contents.
        Following set of rules are encoforced to ensure the correctness of
        the input metadata schema file.
           * ``graph_name`` should be a valid string,
           * ``num_nodes_per_type`` should be a valid list of integers
           * ``num_edges_per_type`` should be a valid list of integers
           * ``node_type`` is a list whose length should be same as the
                length of the list ``num_nodes_per_type``
           * ``edge_type`` is a list whose length should be same as the
                length of the list ``num_edges_per_type``
           * No. of node types with node features should be less then
                total no. of node types
           * No. of edge types with edge features should be less than
                total no. of edge types
           * Node type with node features should be a valid node type,
                valid node types are in the ``node_type`` list
           * Edge type with edge features should be a valie edge type,
                valid edge types are in the ``edge_type`` list
        """
        assert isinstance(self._data[STR_GRAPH_NAME], str), "Invalid graph name"
        assert len(self._data[STR_GRAPH_NAME]) > 0, "Invalid graph name"

        assert (
            len(self._data[STR_NUM_NODES_PER_TYPE]) > 0
        ), "Invalid number of nodes"
        assert (
            len(self._data[STR_NUM_EDGES_PER_TYPE]) > 0
        ), "Invalid number of edges"

        assert (
            len(self._data[STR_NUM_NODES_PER_TYPE]) > 0
        ), "Invalid number of nodes"
        assert (
            len(self._data[STR_NUM_EDGES_PER_TYPE]) > 0
        ), "Invalid number of edges"

        for idx, num_nodes in enumerate(self._data[STR_NUM_NODES_PER_TYPE]):
            assert (
                num_nodes > 0
            ), f"No. of nodes at index: {idx} is not a valid value."

        for idx, num_edges in enumerate(self._data[STR_NUM_EDGES_PER_TYPE]):
            assert (
                num_edges > 0
            ), f"No. of edges at index: {idx} is not a valie value."

        assert len(self._data[STR_NODE_TYPE]) == len(
            self._data[STR_NUM_NODES_PER_TYPE]
        ), "No. of node types does not match with No. of nodes per node type."
        assert len(self._data[STR_EDGE_TYPE]) == len(
            self._data[STR_NUM_EDGES_PER_TYPE]
        ), "No. of edge types does not match with No. of edges per edge type."

        assert len(self._ntype_features) <= len(
            self._data[STR_NUM_NODES_PER_TYPE]
        ), "Node types with features does not match node type counts."
        assert len(self._etype_features) <= len(
            self._data[STR_NUM_EDGES_PER_TYPE]
        ), "Edge types with features does not match edge type counts."

        # Check for each existing node feature, corresponding node type is valid
        for key, _ in self._ntype_feature_files.items():
            ntype = key[0]
            assert (
                ntype in self.ntypes
            ), f"Node Type: {ntype} is not present in the list of node types."

        # Check for each existing edge feature, corresponding edge type is valid
        for key, _ in self._etype_feature_files.items():
            etype = key[0]
            assert (
                etype in self.etypes
            ), f"Edge Type: {etype} is not present in the list of edge types"

    ########################## Prrotected Methods ##############################

    def _init_global_eid_offsets(self):
        """Function to initialize the global_edge_id offsets for all the
        edge types present in the input graph.

        global_edge_ids, or global_eids, are unique ids assigned to each
        edge of the input graph. These ids are expected to be unique ids
        across the input graph.

        The range of global_edge_ids for each edge type in the graph are
        stored in a dictionary where key-value pairs are etypes and a tuple
        which indicates the starting and ending value (exlcusive) of the
        global_edge_ids. For instance,
            etype1: (a, b) where
            etype1 is a string describing the edge type
            and [a, b) is the range of global_edge_ids for the edge type
            `etype1`.
        """
        etypes = self._data[STR_EDGE_TYPE]
        etype_counts = self._data[STR_NUM_EDGES_PER_TYPE]
        for count in etype_counts:
            assert (
                isinstance(count, int) and count > 0
            ), "Invalid value for No. of nodes per type"

        prefix_sum = np.cumsum([0] + etype_counts)
        starts = prefix_sum[:-1]
        ends = prefix_sum[1:]
        ranges = zip(starts, ends)

        self._global_eid_offsets = dict(zip(etypes, ranges))

    def _init_global_nid_offsets(self):
        """Function to initialize the global_node_id offsets for all the
        node types present in the input graph.

        global_node_ids, or global_nids, are unique ids assigned to each
        node of the input graph. These ids are expected to be unique ids
        across the input graph.

        The range of global_node_ids for each node type in the graph are
        stored in a dictionary where key-value pairs are ntypes and a tuple
        which indicates the starting and ending value (exclusive) of the
        global_node_ids. For instance,
            ntype1: [a, b) are the key-value pairs.
            where ntype1 is a string describing the node type
            and [a, b) is the range of global_node_ids for the node type
            `ntype1`.
        """
        ntypes = self._data[STR_NODE_TYPE]
        ntype_counts = self._data[STR_NUM_NODES_PER_TYPE]
        for count in ntype_counts:
            assert (
                isinstance(count, int) and count > 0
            ), "Invalid value for No. of nodes per type"

        prefix_sum = np.cumsum([0] + ntype_counts)
        starts = prefix_sum[:-1]
        ends = prefix_sum[1:]
        ranges = zip(starts, ends)
        self._global_nid_offsets = dict(zip(ntypes, ranges))

    def _init_etype_maps(self):
        """Initialization function to build unique id maps for edge types of
        the input graph. This function creates a list of edge types present in
        the graph, and unique-id <-> edge type and reverse maps.
        """
        self._etypes = self._data[STR_EDGE_TYPE]
        self._etype_id_map = {
            etype_name: idx for idx, etype_name in enumerate(self._etypes)
        }
        self._id_etype_map = dict(enumerate(self._etypes))

    def _init_etype_features(self):
        """Initialization function to build a map in which key are edge types
        in the graph and values are a list of features for the corresponding
        edge type, if present. For instance,
            etype1: [feature1, feature2, ....] where
            etype1 is a edge type from the graph and
            [feature1, feature2, ...] is a list of features associated with
            the etype1 edge type.
        """
        self._etype_features = {}
        edge_data = self._data.get(STR_EDGE_DATA, {})
        for etype in self._data[STR_EDGE_TYPE]:
            features = edge_data.get(etype, {})
            if len(features) > 0:
                self._etype_features[etype] = list(features.keys())

    def _init_etype_feature_files(
        self,
    ):
        """Initialization function to build a map to capture the files names
        and their format. For instance,
            (etype, feature_name): (file_type, delimiter, [file1, file2, ...]
            where ``etype`` is the edge type,
                  ``feature_name`` is a feature associated with ``etype`` node
                  type,
                  ``file_type`` is the type of the file numpy and parquet are
                  supported file formats,
                  ``delimiter`` is the delimiter used when these set of files
                  are created, and
                  ``[file1, file2, ...]`` is a list of files in which features
                  are stored for this edge type.
        """
        self._etype_feature_files = {}
        edge_data = self._data.get(STR_EDGE_DATA, {})
        for etype, etype_info in edge_data.items():
            for feature_name, feature_info in etype_info.items():
                data_files = feature_info.get(STR_DATA, [])
                file_type = None
                delimiter = None
                if len(data_files) > 0:
                    file_type = feature_info[STR_FORMAT][STR_NAME]
                    delimiter = feature_info[STR_FORMAT].get(
                        STR_FORMAT_DELIMITER, ","
                    )

                self.__check_ifexists(data_files, f"{etype}/{feature_name}")
                self._etype_feature_files[(etype, feature_name)] = (
                    file_type,
                    delimiter,
                    data_files,
                )

    def _init_etype_files(
        self,
    ):
        """Initialization function to build a map for the edge types and the
        corresponding information about the set of associated files. For
        instance, etype: (file_type, delimiter, [file1, file2, ...])
            where ``etype`` is an edge type in the input graph,
                  ``file_type`` is type of the file, numpy and csv are the
                  supported file formats,
                  ``delimiter`` is the delimiter used in these files, and
                  ``[file1, file2, ...]`` is a list of files in which the
                  edges are stored for the corresponding edge type.
        """
        self._etype_files = {}
        for etype, etype_info in self._data[STR_EDGES].items():
            file_type = etype_info[STR_FORMAT][STR_NAME]
            delimiter = etype_info[STR_FORMAT][STR_FORMAT_DELIMITER]
            data_files = etype_info[STR_DATA]

            self.__check_ifexists(data_files, etype)
            self._etype_files[etype] = (file_type, delimiter, data_files)

        assert len(self._etype_files) == len(self._etypes), (
            f"In the metadata file there are some edges"
            f" for which there are no corresponding edge files."
            f" etypes = {self._etypes} are all the edge files. "
            f" etypes, for which edge files are present: "
            f" {self.etype_files.keys()}"
        )

    def _init_ntype_maps(self):
        """Initialization function to build unique id maps for node types of
        the input graph. This function creates a list of node types present in
        the graph, and unique-id <-> node type and reverse maps.
        """
        self._ntypes = self._data[STR_NODE_TYPE]
        self._ntype_id_map = {
            ntype_name: idx for idx, ntype_name in enumerate(self._ntypes)
        }
        self._id_ntype_map = dict(enumerate(self._ntypes))

    def _init_ntype_features(self):
        """Initilization function to build a map in which keys are node types
        in the graph and values are a lists of features for the corresponding
        node type, if present. For instance,
            ntype1: [feature1, feature2, ...] where,
            ntype1 is a node type from the graph and
            [feature1, feature2, ....] is a list of features associated with
            the ntype1 node type.
        """
        self._ntype_features = {}
        node_data = self._data.get(STR_NODE_DATA, {})
        for ntype in self._data[STR_NODE_TYPE]:
            features = node_data.get(ntype, {})
            if len(features) > 0:
                self._ntype_features[ntype] = list(features.keys())

    def _init_ntype_feature_files(self):
        """Initialization function to build a map to capture the files names
        and their format. For instance,
            (ntype, feature_name): (file_type, delimiter, [file1, file2, ...]
            where ``ntype`` is the node type,
                  ``feature_name`` is a feature associated with ``ntype`` node
                  type,
                  ``file_type`` is the type of the file numpy and parquet are
                  supported file formats,
                  ``delimiter`` is the delimiter used when these set of files
                  are created, and
                  ``[file1, file2, ...]`` is a list of files in which features
                  are stored for this node type.
        """
        self._ntype_feature_files = {}
        node_data = self._data.get(STR_NODE_DATA, {})
        for ntype, ntype_info in node_data.items():
            for feature_name, feature_info in ntype_info.items():
                data_files = feature_info.get(STR_DATA, [])
                file_type = None
                delimiter = None
                if len(data_files) > 0:
                    file_type = feature_info[STR_FORMAT][STR_NAME]
                    delimiter = feature_info[STR_FORMAT].get(
                        STR_FORMAT_DELIMITER, ","
                    )

                self.__check_ifexists(data_files, f"{ntype}/{feature_name}")
                self._ntype_feature_files[(ntype, feature_name)] = (
                    file_type,
                    delimiter,
                    data_files,
                )

    ############################## Public Methods ##############################

    def get_etype_info(self, etype):
        """Retrieves files and their format information with a given edge
        type. (``file_type``, ``delimiter``, ``[file1, file2, ...]``) is
        returned for the associated edge type. File types (numpy, csv
        supported formats), delimter used in these files and a list of file
        names in which edges are stored for the edge type.

        Argument
        --------
        etype: str
            Edge type from the input graph.

        Returns
        -------
        tuple: (str, str, list(str))
            File type (a valid file type which are ``numpy`` or ``csv``),
            valid delimiter and list of file paths in which edges are
            stored for the given edge type.
        """
        return self._etype_files[etype]

    def get_etype_features(self, etype):
        """Retrieves a list of edge features associated with ``etype``.

        Argument
        --------
        etype: str
            Edge type from the input graph.

        Returns
        ------
        list(str): list of strings
            List of strings describing the edge features.
        """
        return self._etype_features[etype]

    def get_etype_feature_files(self, etype, feature_name):
        """Retrieves a tuple, with the metadata for edge feature.

        Argument
        --------
        etype: str
            Edge type from the input graph.
        feature_name: str
            Edge feature name associated with the ``etype`` edge type.

        Returns
        -------
        tuple: (str, str, list(str))
            File type (valid values are ``numpy``or ``parquet``), delimiter
            (a valid delimiter) and list of paths for filenames in which
            edge features are stored.
        """
        return self._etype_feature_files[(etype, feature_name)]

    def get_ntype_features(self, ntype):
        """Retrieves a list of node features associated with ``ntype``.

        Argument
        --------
        ntype: str
            Node type from the input graph.

        Returns
        ------
        list(str): list of strings
            List of strings describing the node features.
        """
        return self._ntype_features[ntype]

    def get_ntype_feature_files(self, ntype, feature_name):
        """Retrieves a tuple, with the metadata for node feature.

        Argument
        --------
        ntype: str
            Node type from the input graph.
        feature_name: str
            Node feature name associated with the ``ntype`` node type.

        Returns
        -------
        tuple: (str, str, list(str))
            File type (valid values are ``numpy``or ``parquet``), delimiter
            (a valid delimiter) and list of paths for filenames in which
            node features are stored.
        """
        return self._ntype_feature_files[(ntype, feature_name)]

    ############################## Properties #################################

    @property
    def data(self):
        """Property to get the schema object read from the input metadata
        file.
        """
        return self._data

    @property
    def id_etype_map(self):
        """Property to get the dictionary which stores (edge_id, etype) as
        the key-value pairs.
        """
        return self._id_etype_map

    @property
    def id_ntype_map(self):
        """Property to get the node-id <-> node type map."""
        return self._id_ntype_map

    @property
    def etypes(self):
        """Property to get the list of edges types of the input graph."""
        return self._etypes

    @property
    def etype_id_map(self):
        """Property to get the dictionary which stores (etype, edge_id) as
        the key-value pairs.
        """
        return self._etype_id_map

    @property
    def etype_features(self):
        """Property to get a list of feature names for each edge type."""
        return self._etype_features

    @property
    def etype_feature_files(self):
        """Property to get the dictionary which stores file information for
        each edge type as key-value pairs.
        """
        return self._etype_feature_files

    @property
    def etype_files(self):
        """Property to get the dictionary which stores information about
        files associated with an edge type.
        """
        return self._etype_files

    @property
    def global_nid_offsets(self):
        """Property to get the dictionary which stores the node types
        and associated global_node_id offsets as a tuple
        """
        return self._global_nid_offsets

    @property
    def global_eid_offsets(self):
        """Property to get the dictionary which stores (edge type,
        (global_edge_id, global_edge_id)) as the key-value pairs. Each edge
        type is associated with a set of global edge ids.
        """
        return self._global_eid_offsets

    @property
    def metadata_path(self):
        """Property to get the path for the metadata schema file."""
        return self._metadata_path

    @property
    def ntypes(self):
        """Property to get the node types of the input graph."""
        return self._ntypes

    @property
    def ntype_id_map(self):
        """Property to get the node type <-> node-id map."""
        return self._ntype_id_map

    @property
    def ntype_features(self):
        """Property to get a list of feature names for each node type."""
        return self._ntype_features

    @property
    def ntype_feature_files(self):
        """Property to get the dictionary which stores the key, value pairs as
        (ntype, feature_name) <-> (file_type, delimiter, [file1, file2, ...].
        """
        return self._ntype_feature_files
