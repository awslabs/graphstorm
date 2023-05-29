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
import pytest
import torch

from graphstorm.graphpartitioning.metadata_schema import MetadataSchema
from graph_dataset import *


def test_load_json():
    """Unit test to test json load function."""
    input_dict = {}
    add_graph_name(input_dict)

    with tempfile.TemporaryDirectory() as root_dir:

        metadata_path = os.path.join(root_dir, METADATA_NAME)
        with open(metadata_path, "w") as handle:
            json.dump(input_dict, handle, indent=4)

        obj = MetadataSchema()
        obj.init(METADATA_NAME, root_dir, init_maps=False)

        assert (
            obj.data == input_dict
        ), f"Expected and Actual metadata objects does not match."


@pytest.mark.parametrize("ntypes", [[], [NODE1], [NODE1, NODE2]])
def test_ntypes(ntypes):
    """Unit Tests to test various scenarios when only nodes are in the input
    graph dataset.
    """
    input_dict = {}
    with tempfile.TemporaryDirectory() as root_dir:

        add_nodes(input_dict, {ntype: 10 for ntype in ntypes})

        filename = os.path.join(root_dir, METADATA_NAME)
        with open(filename, "w") as input_handle:
            json.dump(input_dict, input_handle, indent=4)

        obj = MetadataSchema()
        obj.init(METADATA_NAME, root_dir, init_maps=False)
        obj.init_ntype_maps()

        # Test case here.
        expected_ntype_id_map = {ntype: idx for idx, ntype in enumerate(ntypes)}
        assert (
            expected_ntype_id_map == obj.ntype_id_map
        ), f"ntype_id_map initialization failure"

        exptected_id_ntype_map = {
            idx: ntype for idx, ntype in enumerate(ntypes)
        }
        assert (
            exptected_id_ntype_map == obj.id_ntype_map
        ), f"id_ntype_map test failure"

        expected_ntypes = ntypes
        assert expected_ntypes == obj.ntypes, f"ntypes failure."

        obj.init_global_nid_offsets()
        counts = np.cumsum([0] + [10 for ntype in ntypes])
        ranges = [
            (counts[idx], counts[idx + 1]) for idx in range(len(counts) - 1)
        ]
        expected_gnid_offsets = {
            ntype: ranges[idx] for idx, ntype in enumerate(ntypes)
        }
        assert expected_gnid_offsets == obj.global_nid_offsets


@pytest.mark.parametrize("etypes", [[], [EDGE1], [EDGE1, EDGE2]])
def test_etypes(etypes):
    """Unit test to test various scenarions when only nodes and edges
    are present in the input graph schema.
    """
    ntypes = [NODE1, NODE2]
    input_dict = {}
    with tempfile.TemporaryDirectory() as root_dir:

        add_nodes(input_dict, {ntype: 10 for ntype in ntypes})
        add_edges(input_dict, {etype: 10 for etype in etypes})

        filename = os.path.join(root_dir, METADATA_NAME)
        with open(filename, "w") as input_handle:
            json.dump(input_dict, input_handle, indent=4)

        obj = MetadataSchema()
        obj.init(METADATA_NAME, root_dir, init_maps=False)
        obj.init_etype_maps()

        # Test case here.
        expected_etype_id_map = {etype: idx for idx, etype in enumerate(etypes)}
        assert (
            expected_etype_id_map == obj.etype_id_map
        ), f"etype_id_map initialization failure"

        exptected_id_etype_map = {
            idx: etype for idx, etype in enumerate(etypes)
        }
        assert (
            exptected_id_etype_map == obj.id_etype_map
        ), f"id_etype_map test failure"

        expected_etypes = etypes
        assert expected_etypes == obj.etypes, f"etypes failure."

        obj.init_global_eid_offsets()
        counts = np.cumsum([0] + [10 for etype in etypes])
        ranges = [
            (counts[idx], counts[idx + 1]) for idx in range(len(counts) - 1)
        ]
        expected_geid_offsets = {
            etype: ranges[idx] for idx, etype in enumerate(etypes)
        }
        assert expected_geid_offsets == obj.global_eid_offsets


@pytest.mark.parametrize("etypes", [[], [EDGE1], [EDGE1, EDGE2]])
@pytest.mark.parametrize("edge_fmt", ["csv", "numpy", "parquet"])
@pytest.mark.parametrize("edge_fmt_del", [" ", "/"])
@pytest.mark.parametrize("create_files", [True, False])
@pytest.mark.parametrize("prefix_dir", ["", "edges_"])
@pytest.mark.parametrize("is_absolute", [True, False])
def test_etype_files(
    etypes, edge_fmt, edge_fmt_del, create_files, prefix_dir, is_absolute
):
    """Helper function to test various scenarions with edge files.
    Files presence/absence is test rigorously.
    """
    input_dict = {}
    with tempfile.TemporaryDirectory() as root_dir:

        add_nodes(input_dict, {ntype: 10 for ntype in ntypes})

        file_metadata = namedtuple("file_metadata", "")
        file_metadata.root_dir = root_dir
        file_metadata.prefix_dir = prefix_dir
        file_metadata.create_files = create_files
        file_metadata.is_absolute = is_absolute
        file_metadata.file_prefix = "edges_"

        add_edges_files(
            input_dict,
            {etype: 10 for etype in etypes},
            {etype: idx + 1 for idx, etype in enumerate(etypes)},
            {"name": edge_fmt, "delimiter": edge_fmt_del},
            file_metadata,
        )

        filename = os.path.join(root_dir, METADATA_NAME)
        with open(filename, "w") as input_handle:
            json.dump(input_dict, input_handle, indent=4)

        obj = MetadataSchema()
        obj.init(METADATA_NAME, root_dir, init_maps=False)
        obj.init_etype_maps()

        try:
            obj.init_etype_files()
        except FileNotFoundError as exp:
            if create_files:
                raise exp
            else:
                pass


@pytest.mark.parametrize("ntypes", [[], [NODE1], [NODE1, NODE2]])
@pytest.mark.parametrize("file_type", ["parquet", "numpy"])
@pytest.mark.parametrize("delimiter", [" ", "/"])
@pytest.mark.parametrize("create_files", [True, False])
@pytest.mark.parametrize("prefix_dir", ["", "node_data"])
@pytest.mark.parametrize("is_absolute", [True, False])
def test_node_features(
    ntypes, file_type, delimiter, create_files, prefix_dir, is_absolute
):
    """Unit tests when node features are present in the input graph."""
    etypes = [EDGE1, EDGE2]
    input_dict = {}
    with tempfile.TemporaryDirectory() as root_dir:

        file_metadata = namedtuple("file_metadata", "")
        file_metadata.root_dir = root_dir
        file_metadata.prefix_dir = prefix_dir
        file_metadata.create_files = create_files
        file_metadata.is_absolute = is_absolute
        file_metadata.file_prefix = "node_feature_"

        add_node_features(
            input_dict,
            {ntype: 10 for ntype in ntypes},
            {ntype: [NFEAT1] for ntype in ntypes},
            {ntype: 5 for ntype in ntypes},
            {
                ntype + "/" + NFEAT1: {NAME: file_type, DELIMITER: delimiter}
                for ntype in ntypes
            },
            file_metadata,
        )
        add_edges(input_dict, {etype: 10 for etype in etypes})

        filename = os.path.join(root_dir, METADATA_NAME)
        with open(filename, "w") as input_handle:
            json.dump(input_dict, input_handle, indent=4)

        # Tests here.
        obj = MetadataSchema()
        obj.init(METADATA_NAME, root_dir, init_maps=False)
        obj.init_ntype_maps()

        try:
            obj.init_ntype_feature_files()
        except FileNotFoundError as exp:
            if create_files:
                raise exp
            else:
                pass


@pytest.mark.parametrize("etypes", [[], [EDGE1], [EDGE1, EDGE2]])
@pytest.mark.parametrize("file_type", ["csv", "numpy"])
@pytest.mark.parametrize("delimiter", [" ", "/"])
@pytest.mark.parametrize("create_files", [True, False])
@pytest.mark.parametrize("prefix_dir", ["", "edges"])
@pytest.mark.parametrize("is_absolute", [True, False])
def test_edge_features(
    etypes, file_type, delimiter, create_files, prefix_dir, is_absolute
):
    """Unit tests when edge features are present in the input graph."""
    ntypes = [NODE1, NODE2]
    input_dict = {}
    with tempfile.TemporaryDirectory() as root_dir:

        file_metadata = namedtuple("file_metadata", "")
        file_metadata.root_dir = root_dir
        file_metadata.prefix_dir = prefix_dir
        file_metadata.create_files = create_files
        file_metadata.is_absolute = is_absolute
        file_metadata.file_prefix = "edge_feature_"

        add_nodes(input_dict, {ntype: 10 for ntype in ntypes})
        add_edge_features(
            input_dict,
            {etype: 10 for etype in etypes},
            {etype: [EFEAT1] for etype in etypes},
            {etype: 5 for etype in etypes},
            {
                etype + "/" + EFEAT1: {NAME: file_type, DELIMITER: delimiter}
                for etype in etypes
            },
            file_metadata,
        )

        filename = os.path.join(root_dir, METADATA_NAME)
        with open(filename, "w") as input_handle:
            json.dump(input_dict, input_handle, indent=4)

        obj = MetadataSchema()
        obj.init(METADATA_NAME, root_dir, init_maps=False)
        obj.init_etype_maps()

        try:
            obj.init_etype_feature_files()
        except FileNotFoundError as exp:
            if create_files:
                raise exp
            else:
                pass


@pytest.mark.parametrize(
    "metadata",
    [
        {
            GRAPH_NAME: "Test",
            NUM_NODES_PER_TYPE: [10, 10],
            NODE_TYPE: [NODE1, NODE2],
            NUM_EDGES_PER_TYPE: [30],
            EDGE_TYPE: [EDGE1],
            EDGES: {EDGE1: {FORMAT: {NAME: "csv", DELIMITER: ":"}, DATA: []}},
            NODE_DATA: {
                NODE1: {
                    NFEAT1: {FORMAT: {NAME: "npy", DELIMITER: " "}, DATA: []}
                },
            },
            EDGE_DATA: {
                EDGE1: {
                    EFEAT1: {
                        FORMAT: {NAME: "parquet", DELIMITER: " "},
                        DATA: [],
                    }
                }
            },
        },
    ],
)
def test_generic_usecase(metadata):
    """Unit test case to test general usecase of the pipeline."""

    def generate_files(filenames):
        lines = ["This is a test file"]
        for filename in filenames:
            with open(filename, "w") as handle:
                handle.writelines(lines)

    with tempfile.TemporaryDirectory() as root_dir:

        # Create Node features.
        node_feature_files = []
        extension = metadata[NODE_DATA][NODE1][NFEAT1][FORMAT][NAME]
        for idx in range(1):
            node_feature_files.append(
                os.path.join(root_dir, f"node_feature_{idx}.{extension}")
            )
        metadata[NODE_DATA][NODE1][NFEAT1][DATA] = node_feature_files
        generate_files(node_feature_files)

        # Create edge files.
        edge_files = []
        extension = metadata[EDGES][EDGE1][FORMAT][NAME]
        for idx in range(2):
            edge_files.append(
                os.path.join(root_dir, f"edges_{idx}.{extension}")
            )
        metadata[EDGES][EDGE1][DATA] = edge_files
        generate_files(edge_files)

        # Create edge feature files.
        edge_feature_files = []
        extension = metadata[EDGE_DATA][EDGE1][EFEAT1][FORMAT][NAME]
        for idx in range(2):
            edge_feature_files.append(
                os.path.join(root_dir, f"edge_feature_{idx}.{extension}")
            )
        metadata[EDGE_DATA][EDGE1][EFEAT1][DATA] = edge_feature_files
        generate_files(edge_feature_files)

        # Create metadata file.
        filename = os.path.join(root_dir, METADATA_NAME)
        with open(filename, "w") as input_handle:
            json.dump(metadata, input_handle, indent=4)

        # Test here.
        obj = MetadataSchema()
        obj.init(METADATA_NAME, root_dir)

        # Check base directory for the dataset.
        assert obj.metadata_path == os.path.join(
            root_dir, METADATA_NAME
        ), f"Base directory does not match."

        # Check node types.
        assert obj.ntypes == [
            NODE1,
            NODE2,
        ], f"Number of node types does not match."
        assert obj.ntype_id_map == {
            NODE1: 0,
            NODE2: 1,
        }, f"ntype_id_map does not match."
        assert obj.id_ntype_map == {
            0: NODE1,
            1: NODE2,
        }, f"id_ntype_map does not match."
        assert obj.global_nid_offsets == {
            NODE1: (0, 10),
            NODE2: (10, 20),
        }, f"global_nid_offsets does not match."

        # Check edge types.
        assert obj.etypes == [EDGE1], f"Edge Types does not match."
        assert obj.etype_id_map == {EDGE1: 0}, f"etype_id_map does not match."
        assert obj.id_etype_map == {0: EDGE1}, f"id_etype_map does not match."
        assert obj.global_eid_offsets == {
            EDGE1: (0, 30)
        }, f"global_eid_offsets does not match."

        # Check node features.
        assert obj.ntype_features == {
            NODE1: [NFEAT1]
        }, "node features does not match."

        # Check node features using get method.
        assert obj.get_ntype_features(NODE1) == [
            NFEAT1
        ], f"Node feature names does not match."

        # Check edge features.
        assert obj.etype_features == {
            EDGE1: [EFEAT1]
        }, "node features does not match."

        assert obj.get_etype_features(EDGE1) == [
            EFEAT1
        ], f"Edge feature names does not match."

        # Check etype files using get method.
        assert obj.get_etype_info(EDGE1) == (
            "csv",
            ":",
            [
                os.path.join(root_dir, "edges_0.csv"),
                os.path.join(root_dir, "edges_1.csv"),
            ],
        ), f"Edge files does not match."

        # Check ntype feat files.
        assert obj.get_ntype_feature_files(NODE1, NFEAT1) == (
            "npy",
            " ",
            [os.path.join(root_dir, "node_feature_0.npy")],
        ), f"Features files for nodes does not match {obj.ntype_feature_files}"

        # Check etype feat files.
        assert obj.etype_feature_files == {
            (EDGE1, EFEAT1): (
                "parquet",
                " ",
                [
                    os.path.join(root_dir, "edge_feature_0.parquet"),
                    os.path.join(root_dir, "edge_feature_1.parquet"),
                ],
            )
        }, "Features files for nodes does not match."

        # Check edge files.
        assert obj.etype_files == {
            EDGE1: (
                "csv",
                ":",
                [
                    os.path.join(root_dir, "edges_0.csv"),
                    os.path.join(root_dir, "edges_1.csv"),
                ],
            )
        }, "Edge files for edge types does not match."

        # Check data property
        assert (
            obj.data == metadata
        ), f"Metadata contents and data property does not match."


@pytest.mark.parametrize(
    "metadata",
    [
        {
            GRAPH_NAME: "Test",
        },
        {
            GRAPH_NAME: "Test",
            NUM_NODES_PER_TYPE: [10, 10],
            NODE_TYPE: [NODE1, NODE2],
        },
        {
            GRAPH_NAME: "Test",
            NUM_NODES_PER_TYPE: [10, 10],
            NODE_TYPE: [NODE1, NODE2, NODE3],
        },
        {
            GRAPH_NAME: "Test",
            NUM_NODES_PER_TYPE: [10, 10],
            NODE_TYPE: [NODE1, NODE2],
            NUM_EDGES_PER_TYPE: [30],
            EDGE_TYPE: [EDGE1],
        },
        {
            GRAPH_NAME: "Test",
            NUM_NODES_PER_TYPE: [10, 10],
            NODE_TYPE: [NODE1, NODE2],
            NUM_EDGES_PER_TYPE: [30, 20],
            EDGE_TYPE: [EDGE1, EDGE2],
            EDGES: {EDGE1: {FORMAT: {NAME: "csv", DELIMITER: ":"}, DATA: []}},
        },
        {
            GRAPH_NAME: 123,
            NUM_NODES_PER_TYPE: [10, 10],
            NODE_TYPE: [NODE1, NODE2],
            NUM_EDGES_PER_TYPE: [30, 20],
            EDGE_TYPE: [EDGE1, EDGE2],
            EDGES: {EDGE1: {FORMAT: {NAME: "csv", DELIMITER: ":"}, DATA: []}},
        },
        {
            GRAPH_NAME: "Test",
            NUM_NODES_PER_TYPE: [10, 10],
            NODE_TYPE: [NODE1, NODE2],
            NUM_EDGES_PER_TYPE: [30],
            EDGE_TYPE: [EDGE1, EDGE2],
            EDGES: {EDGE1: {FORMAT: {NAME: "csv", DELIMITER: ":"}, DATA: []}},
        },
        {
            GRAPH_NAME: "Test",
            NUM_NODES_PER_TYPE: [10],
            NODE_TYPE: [NODE1, NODE2],
            NUM_EDGES_PER_TYPE: [30, 20],
            EDGE_TYPE: [EDGE1, EDGE2],
            EDGES: {EDGE1: {FORMAT: {NAME: "csv", DELIMITER: ":"}, DATA: []}},
        },
        {
            GRAPH_NAME: "Test",
            NUM_NODES_PER_TYPE: [10],
            NODE_TYPE: [NODE1],
            NUM_EDGES_PER_TYPE: [30],
            EDGE_TYPE: [
                EDGE1,
            ],
            EDGES: {EDGE1: {FORMAT: {NAME: "csv", DELIMITER: ":"}, DATA: []}},
            NODE_DATA: {
                NODE1: {
                    NFEAT1: {FORMAT: {NAME: "npy", DELIMITER: " "}, DATA: []}
                },
                NODE2: {
                    NFEAT1: {FORMAT: {NAME: "npy", DELIMITER: " "}, DATA: []}
                },
            },
        },
        {
            GRAPH_NAME: "Test",
            NUM_NODES_PER_TYPE: [10],
            NODE_TYPE: [NODE1],
            NUM_EDGES_PER_TYPE: [30],
            EDGE_TYPE: [
                EDGE1,
            ],
            EDGES: {EDGE1: {FORMAT: {NAME: "csv", DELIMITER: ":"}, DATA: []}},
            EDGE_DATA: {
                EDGE1: {
                    EFEAT1: {FORMAT: {NAME: "npy", DELIMITER: " "}, DATA: []}
                },
                EDGE2: {
                    EFEAT1: {FORMAT: {NAME: "npy", DELIMITER: " "}, DATA: []}
                },
            },
        },
    ],
)
def test_error_cases(metadata):
    """Unit tests to test invalid test cases with the input metadata schema."""

    def generate_files(filenames):
        lines = ["This is a test file"]
        for filename in filenames:
            with open(filename, "w") as handle:
                handle.writelines(lines)

    with tempfile.TemporaryDirectory() as root_dir:
        # Generate files, if necessary
        # Create edge files.
        if EDGES in metadata:
            if EDGE1 in metadata[EDGES]:
                edge_files = []
                extension = metadata[EDGES][EDGE1][FORMAT][NAME]
                for idx in range(2):
                    edge_files.append(
                        os.path.join(root_dir, f"edges_{idx}.{extension}")
                    )
                metadata[EDGES][EDGE1][DATA] = edge_files
                generate_files(edge_files)
        if NODE_DATA in metadata:
            if NODE1 in metadata[NODE_DATA]:
                node_files = []
                extension = metadata[NODE_DATA][NODE1][NFEAT1][FORMAT][NAME]
                for idx in range(2):
                    node_files.append(
                        os.path.join(
                            root_dir, f"node_1_features_{idx}.{extension}"
                        )
                    )
                metadata[NODE_DATA][NODE1][NFEAT1][DATA] = node_files
                generate_files(node_files)
            if NODE2 in metadata[NODE_DATA]:
                node_files = []
                extension = metadata[NODE_DATA][NODE2][NFEAT1][FORMAT][NAME]
                for idx in range(2):
                    node_files.append(
                        os.path.join(
                            root_dir, f"node_2_features_{idx}.{extension}"
                        )
                    )
                metadata[NODE_DATA][NODE2][NFEAT1][DATA] = node_files
                generate_files(node_files)
        if EDGE_DATA in metadata:
            if EDGE1 in metadata[EDGE_DATA]:
                edge_files = []
                extension = metadata[EDGE_DATA][EDGE1][EFEAT1][FORMAT][NAME]
                for idx in range(2):
                    edge_files.append(
                        os.path.join(
                            root_dir, f"edge_1_features_{idx}.{extension}"
                        )
                    )
                metadata[EDGE_DATA][EDGE1][EFEAT1][DATA] = edge_files
                generate_files(edge_files)
            if EDGE2 in metadata[EDGE_DATA]:
                edge_files = []
                extension = metadata[EDGE_DATA][EDGE2][EFEAT1][FORMAT][NAME]
                for idx in range(2):
                    edge_files.append(
                        os.path.join(
                            root_dir, f"edge_2_features_{idx}.{extension}"
                        )
                    )
                metadata[EDGE_DATA][EDGE2][EFEAT1][DATA] = edge_files
                generate_files(edge_files)

        # Create metadata file.
        filename = os.path.join(root_dir, METADATA_NAME)
        with open(filename, "w") as input_handle:
            json.dump(metadata, input_handle, indent=4)

        # Test here.
        obj = MetadataSchema()
        try:
            obj.init(METADATA_NAME, root_dir)
        except KeyError as exp:
            pass
        except ValueError as exp:
            pass


if __name__ == "__main__":
    test_load_json()
    test_load_json()
    test_ntypes(ntypes)
    test_etypes(etypes)
    test_etype_files()
    test_node_features()
    test_edge_features()
    test_generic_usecase()
    test_error_cases()
