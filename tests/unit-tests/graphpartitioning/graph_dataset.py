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
"""
import json
import os
import tempfile

from collections import namedtuple

import numpy as np
import pyarrow as pa
import torch

from graphpartitioning import array_readwriter, constants

################### Constants used in pipelines unit tests ####################

METADATA_NAME = "metadata.json"
GRAPH_NAME = "graph_name"
NUM_NODES_PER_TYPE = "num_nodes_per_type"
NUM_EDGES_PER_TYPE = "num_edges_per_type"
NODE_TYPE = "node_type"
EDGE_TYPE = "edge_type"
NODE_DATA = "node_data"
EDGE_DATA = "edge_data"
EDGES = "edges"

FORMAT = "format"
NAME = "name"
DELIMITER = "delimiter"

NODE1 = "n1"
NODE2 = "n2"
NODE3 = "n3"

EDGE1 = f"{NODE1}:e1:{NODE2}"
EDGE2 = f"{NODE2}:e2:{NODE1}"

NFEAT1 = "nfeat1"
EFEAT1 = "efeat1"

NFEAT2 = "nfeat2"
EFEAT2 = "efeat2"

DELIMITER = "delimiter"
DATA = "data"

NUM_NODES1 = 10
NUM_NODES2 = 10

NUM_EDGES1 = 10
NUM_EDGES2 = 10

ntypes = [NODE1, NODE2]
etypes = [EDGE1, EDGE2]

############## Set of utility functions to help execute unit tests ############


def add_graph_name(schema):
    """Helper function to add ``graph_name`` to the input schema.

    Argument
    --------
    schema: dict
        Dictionary object to store graph metadata.
    """
    schema[GRAPH_NAME] = "Test"


def add_nodes(schema, val):
    """Helper function to add ``node_type`` and ``num_nodes_per_type``
    to the input schema.

    Argument
    --------
    schema: dict
        Dictionary object to store graph metadata.
    val: dict
        Dictionary which stores node types and no. of nodes for this node type
    """
    num_nodes = []
    ntypes = []
    for key, value in val.items():
        ntypes.append(key)
        num_nodes.append(value)

    schema[NUM_NODES_PER_TYPE] = num_nodes
    schema[NODE_TYPE] = ntypes


def add_edges(schema, val):
    """Helper function to add ``edge_type`` and ``num_edges_per_type`` to
    the input schema.

    Argument
    --------
    schema: dict
        Dictionary object to store graph metadata.
    val: dict
        Dictionary which stores edge types and no. of edges for this edge type.
    """
    num_edges = []
    etypes = []
    for etype, edge_count in val.items():
        num_edges.append(edge_count)
        etypes.append(etype)
    schema[NUM_EDGES_PER_TYPE] = num_edges
    schema[EDGE_TYPE] = etypes
    return etypes


def get_data(file_prefix, index):
    """Helper function to generate dummy data for edges, node features and
    edge features.

    Edges are generated with type node ids (unique node ids within the context
    of a particular node type) between 1 and 10 for both source and destination
    end points. A set of 10 edges are generated.

    Dummy node features are generated which are (10, 5) shape and np.int64.

    Dummy edge features are generated which are (10, 10) shape and
    np.float64 dtype.

    Argument
    --------
    file_prefix: str
        Prefix string to add to the filename.
    index: int
        Number used to identify the id of the file.
    """
    data = None
    if file_prefix == "edges_":
        src_ids = np.arange(10, dtype=np.int64).reshape(10, 1)
        dst_ids = np.arange(10, dtype=np.int64).reshape(10, 1)
        data = np.hstack((src_ids, dst_ids)).astype(np.int64)
        data = data.reshape(10, 2)
    elif file_prefix == "edge_feature_":
        data = np.stack(
            [np.ones((5,), dtype=np.uint8) * idx for idx in range(10)]
        )
        data = data + 100 * (index + 1)
    else:
        data = np.stack(
            [np.ones((10,), dtype=np.float64) * idx for idx in range(10)]
        )
        data = data + 1000 * (index + 1)
    return data


def process_files(file_name, file_format, file_metadata, file_count):
    """Helper function to add entries into the input schema about a
    set of files. These could be assocaited with either edges, node features
    or edge features.

    Various flags are used to control the process of generating files to
    simulate various unit testing scenarios. File creation, path of the
    filename, file prefix, base directory are some of the parameters provided
    by the invoker of this function.

    Argument
    --------
    file_name: str
        Partial name of the file.
    file_format: dict
        Dictionary which sotres delimiter and file type (csv, numpy, parquet).
    file_metadata: namedtuple
        Namedtuple object which provides root-directory, prefix-directory,
        file-creation-file, file-path-type and file-prefix are some of the
        key-value pairs stored in this parameter.
    file_count: dict
        Dicionary which stored no. of files to create for a given node/edge
        type.
    """
    root_dir = file_metadata.root_dir
    prefix_dir = file_metadata.prefix_dir
    create_files = file_metadata.create_files
    is_absolute = file_metadata.is_absolute
    file_prefix = file_metadata.file_prefix

    names = []
    for i in range(file_count):
        extension = file_format[NAME]
        filename = os.path.join(
            prefix_dir, f"{file_prefix}_{file_name}_{i}.{extension}"
        )
        if is_absolute:
            filename = os.path.join(root_dir, filename)
        names.append(filename)
        if create_files:
            local_dir = os.path.join(root_dir, prefix_dir)
            if not os.path.isdir(local_dir):
                os.makedirs(local_dir, exist_ok=True)

            data = get_data(file_prefix, i)
            filename = os.path.join(root_dir, filename)
            array_readwriter.get_array_parser(**file_format).write(
                filename, data
            )
    return names


def add_edges_files(schema, val, edge_file_count, edge_format, file_metadata):
    """Helper function to add and generate files for edges types in the
    graph.

    Argument
    --------
        schema: dict
            Dictionary which stores the metadata of the unit test graphs.
        val: dict
            Dictionary which stores no. of edges for each edge type in
            the input graph.
        edge_file_count: dict
            Dictionary which stores edge types and no. of files to generate
            for each edge type.
        edge_format: dict
            Dictionary which stores file format information for each
            etype/feature_name pair as (file_type, delimiter, list_of_files)
        file_metadata: namedtuple
            Namedtuple to control the file creation parameters, like file-
            creation-flag, path-type-of-the-file, base_dir-file-location,
            prefix-directory, and file_name.
    """
    etypes = add_edges(schema, val)
    root_dir = file_metadata.root_dir
    prefix_dir = file_metadata.prefix_dir
    create_files = file_metadata.create_files
    is_absolute = file_metadata.is_absolute
    file_prefix = file_metadata.file_prefix

    edge_data = {}
    for idx, etype in enumerate(etypes):
        edge_data[etype] = {}
        edge_format_fix = {}
        edge_format_fix[NAME] = edge_format[NAME]
        if edge_format[NAME] == "csv":
            edge_format_fix[DELIMITER] = (
                edge_format[DELIMITER] if DELIMITER in edge_format else ":"
            )
        edge_data[etype][FORMAT] = edge_format
        edge_data[etype][DATA] = process_files(
            etype, edge_format_fix, file_metadata, edge_file_count[etype]
        )

    schema[EDGES] = edge_data


def add_node_features(
    schema, val, node_features, node_file_count, file_fmt, file_metadata
):
    """Helper function to add node features to the input schema.

    Arrgument
    ---------
    schema: dict
        Dictionary which stores the metadata of the unit test graphs.
    val: dict
        Dictionary which stores no. of edges for each node type in
        the input graph.
    node_features: list of str
        Dictionary which stores node type and list of associated node
        features.
    node_file_count: dict
        Dictionary which stores node types and no. of files to generate
        for each node type.
    file_fmt: dict
        Dictionary which stores file format information for each
        etype/feature_name pair as (file_type, delimiter, list_of_files)
    file_metadata: namedtuple
        Namedtuple to control the file creation parameters, like file-
        creation-flag, path-type-of-the-file, base_dir-file-location,
        prefix-directory, and file_name.
    """
    add_nodes(schema, val)

    node_feature_data = {}
    for ntype, features in node_features.items():
        node_feature_data[ntype] = {}
        for feature in features:
            node_feature_data[ntype][feature] = {}
            node_feature_data[ntype][feature][FORMAT] = file_fmt[
                ntype + "/" + feature
            ]
            if file_fmt[ntype + "/" + feature][NAME] == "csv":
                node_feature_data[ntype][feature][DATA] = process_files(
                    ntype,
                    file_fmt[ntype + "/" + feature],
                    file_metadata,
                    node_file_count[ntype],
                )
            else:
                non_csv_file_format = {}
                non_csv_file_format[NAME] = file_fmt[ntype + "/" + feature][
                    NAME
                ]
                node_feature_data[ntype][feature][DATA] = process_files(
                    ntype,
                    non_csv_file_format,
                    file_metadata,
                    node_file_count[ntype],
                )

    schema[NODE_DATA] = node_feature_data


def add_edge_features(
    schema, val, edge_features, edge_file_count, file_fmt, file_metadata
):
    """Helper function to add edge features to the input schema.

    Argument
    ---------
    schema: dict
        Dictionary which stores the metadata of the unit test graphs.
    val: dict
        Dictionary which stores no. of edges for each edge type in
        the input graph.
    edge_features: list of str
        Dictionary which stores edge type and list of associated edge
        features.
    edge_file_count: dict
        Dictionary which stores edge types and no. of files to generate
        for each edge type.
    file_fmt: dict
        Dictionary which stores file format information for each
        etype/feature_name pair as (file_type, delimiter, list_of_files)
    file_metadata: namedtuple
        Namedtuple to control the file creation parameters, like file-
        creation-flag, path-type-of-the-file, base_dir-file-location,
        prefix-directory, and file_name.
    """
    add_edges(schema, val)

    edge_feature_data = {}
    for etype, features in edge_features.items():
        edge_feature_data[etype] = {}
        for feature in features:
            edge_feature_data[etype][feature] = {}
            edge_feature_data[etype][feature][FORMAT] = file_fmt[
                etype + "/" + feature
            ]
            if file_fmt[etype + "/" + feature][NAME] == "csv":
                edge_feature_data[etype][feature][DATA] = process_files(
                    etype,
                    file_fmt[etype + "/" + feature],
                    file_metadata,
                    edge_file_count[etype],
                )
            else:
                no_csv_fmt = {}
                no_csv_fmt[NAME] = file_fmt[etype + "/" + feature][NAME]
                edge_feature_data[etype][feature][DATA] = process_files(
                    etype, no_csv_fmt, file_metadata, edge_file_count[etype]
                )

    schema[EDGE_DATA] = edge_feature_data
