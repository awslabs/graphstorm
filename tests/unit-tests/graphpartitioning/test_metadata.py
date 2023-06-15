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
import contextlib, pytest
import json
import os
import shutil
import tempfile

import numpy as np
import torch

from graphstorm.graphpartitioning.metadata_schema import MetadataSchema
from graph_dataset import *
from collections import namedtuple


def teardown_dataset(metadata_file, dir_names, file_names):
    os.remove(metadata_file)
    for file_name in file_names:
        os.remove(file_name)
    for dir_name in dir_names:
        shutil.rmtree(dir_name)


@pytest.fixture(params=["temp", "none", "empty"])
def root_dir(request):
    if request.param == "temp":
        with tempfile.TemporaryDirectory() as root_dir:
            yield root_dir
            # Tear down
            shutil.rmtree(root_dir)
    elif request.param == "none":
        yield None
    elif request.param == "empty":
        yield ""


@pytest.fixture(scope="function")
def load_json_data(root_dir):
    (
        file_name,
        input_dict,
        edge_dirs,
        edge_files,
    ) = create_graph_without_features(root_dir)
    yield root_dir, input_dict
    # Tear donw.
    teardown_dataset(file_name, edge_dirs, edge_files)


@pytest.fixture(params=[[], [NTYPE1], [NTYPE1, NTYPE2]])
def ntypes(request):
    yield request.param


@pytest.fixture(params=[[], [ETYPE1], [ETYPE1, ETYPE2]])
def etypes(request):
    yield request.param


@pytest.fixture(params=["csv", "numpy", "parquet"])
def file_fmt(request):
    yield request.param


@pytest.fixture(params=[" ", "/"])
def file_fmt_del(request):
    yield request.param


@pytest.fixture(params=[True, False])
def create_files(request):
    yield request.param


@pytest.fixture(params=["", "edges_"])
def edges_dir(request):
    yield request.param


@pytest.fixture(params=[True, False])
def absolute_path(request):
    yield request.param


@pytest.fixture(params=["", "node_data"])
def node_feat_dir(request):
    yield request.param


@pytest.fixture(params=["", "edge_data"])
def edge_feat_dir(request):
    yield request.param


def _assert_filenames(filenames, root_dir, absolute_path):
    for filename in filenames:
        if absolute_path:
            if root_dir is None or root_dir == "":
                assert os.path.isfile(
                    os.path.join(os.getcwd(), filename)
                ), f"File: {filename} is not an absolute path"
            else:
                assert os.path.isfile(
                    os.path.join(root_dir, filename)
                ), f"File: {filename} is not an absolute path"
        else:
            if root_dir is None or root_dir == "":
                assert os.path.isfile(
                    filename
                ), f"File: {filename} does not exist"
            else:
                assert os.path.isfile(
                    os.path.join(root_dir, filename)
                ), f"File: {filename} does not exist"


def test_load_json(load_json_data):
    # Run test.
    root_dir = load_json_data[0]
    actual_json_object = load_json_data[1]
    obj = MetadataSchema(METADATA_NAME, root_dir)
    assert (
        obj.data == actual_json_object
    ), f"Expected and actual metadata objects does not match."


@pytest.fixture(scope="function")
def ntypes_test_data(ntypes):
    with tempfile.TemporaryDirectory() as root_dir:
        input_dict = {}
        add_graph_name(input_dict)
        add_nodes(input_dict, {ntype: 10 for ntype in ntypes})
        file_metadata = namedtuple("file_metadata", "")
        file_metadata.root_dir = root_dir
        file_metadata.prefix_dir = ""
        file_metadata.create_files = True
        file_metadata.is_absolute = True
        file_metadata.file_prefix = "edges_"
        edge_dirs, edge_files = add_edges_files(
            input_dict,
            {ETYPE1: NUM_EDGES1, ETYPE2: NUM_EDGES2},
            {ETYPE1: 1, ETYPE2: 1},
            {NAME: "csv", DELIMITER: ":"},
            file_metadata,
        )
        file_name = os.path.join(root_dir, METADATA_NAME)
        with open(file_name, "w") as handle:
            json.dump(input_dict, handle, indent=4)

        # Run Test.
        yield root_dir, ntypes
        # Tear down.
        teardown_dataset(file_name, edge_dirs, edge_files)


def test_ntypes(ntypes_test_data):
    root_dir = ntypes_test_data[0]
    ntypes = ntypes_test_data[1]

    if ntypes == []:
        with pytest.raises(
            AssertionError, match=r".*Invalid number of nodes.*"
        ):
            obj = MetadataSchema(METADATA_NAME, root_dir)
    else:
        obj = MetadataSchema(METADATA_NAME, root_dir)
        actual_ntypes, actual_ntype_id_map = obj._init_ntype_maps(obj.data)

        # Test case here.
        # Check ntype <-> id map.
        expected_ntype_id_map = {ntype: idx for idx, ntype in enumerate(ntypes)}
        assert (
            expected_ntype_id_map == actual_ntype_id_map
        ), f"ntype_id_map initialization failure"

        # Check ntypes array.
        expected_ntypes = ntypes
        assert expected_ntypes == actual_ntypes, f"ntypes failure."

        # Check global node ids offsets for all the node types.
        actual_global_nid_offsets = obj._init_global_nid_offsets(obj.data)
        counts = np.cumsum([0] + [10 for ntype in ntypes])
        ranges = [
            (counts[idx], counts[idx + 1]) for idx in range(len(counts) - 1)
        ]
        expected_gnid_offsets = {
            ntype: ranges[idx] for idx, ntype in enumerate(ntypes)
        }
        assert expected_gnid_offsets == actual_global_nid_offsets


@pytest.fixture(scope="function")
def etypes_test_data(etypes):
    with tempfile.TemporaryDirectory() as root_dir:
        input_dict = {}
        add_graph_name(input_dict)
        add_nodes(input_dict, {ntype: 10 for ntype in graph_ntypes})
        file_metadata = namedtuple("file_metadata", "")
        file_metadata.root_dir = root_dir
        file_metadata.prefix_dir = ""
        file_metadata.create_files = True
        file_metadata.is_absolute = True
        file_metadata.file_prefix = "edges_"
        edge_dirs, edge_files = add_edges_files(
            input_dict,
            {etype: 10 for etype in etypes},
            {etype: 1 for etype in etypes},
            {NAME: "csv", DELIMITER: ":"},
            file_metadata,
        )
        file_name = os.path.join(root_dir, METADATA_NAME)
        with open(file_name, "w") as handle:
            json.dump(input_dict, handle, indent=4)

        # Run Test.
        yield root_dir, etypes
        # Tear down.
        teardown_dataset(file_name, edge_dirs, edge_files)


def test_etypes(etypes_test_data):
    root_dir = etypes_test_data[0]
    etypes = etypes_test_data[1]
    if etypes == []:
        with pytest.raises(
            AssertionError, match=r".*Invalid number of edges.*"
        ):
            obj = MetadataSchema(METADATA_NAME, root_dir)
    else:
        obj = MetadataSchema(METADATA_NAME, root_dir)
        actual_etypes, actual_etype_id_map = obj._init_etype_maps(obj.data)

        # Check etype <-> id map.
        expected_etype_id_map = {etype: idx for idx, etype in enumerate(etypes)}
        assert (
            expected_etype_id_map == actual_etype_id_map
        ), f"etype_id_map initialization failure"

        # Check etypes.
        expected_etypes = etypes
        assert expected_etypes == actual_etypes, f"etypes failure."

        # Check global edge ids offsets for etypes.
        actual_global_eid_offsets = obj._init_global_eid_offsets(obj.data)
        counts = np.cumsum([0] + [10 for etype in etypes])
        ranges = [
            (counts[idx], counts[idx + 1]) for idx in range(len(counts) - 1)
        ]
        expected_geid_offsets = {
            etype: ranges[idx] for idx, etype in enumerate(etypes)
        }
        assert expected_geid_offsets == actual_global_eid_offsets


@pytest.fixture(scope="function")
def etype_files_dataset(
    etypes, file_fmt, file_fmt_del, create_files, edges_dir, absolute_path
):
    with tempfile.TemporaryDirectory() as root_dir:
        # Add nodes to the metadata.
        input_dict = {}
        add_graph_name(input_dict)
        add_nodes(input_dict, {ntype: 10 for ntype in graph_ntypes})
        # Add edges and create edge files, if necessary
        file_metadata = namedtuple("file_metadata", "")
        file_metadata.root_dir = root_dir
        file_metadata.prefix_dir = edges_dir
        file_metadata.create_files = create_files
        file_metadata.is_absolute = absolute_path
        file_metadata.file_prefix = "edges_"
        edge_dirs, edge_files = add_edges_files(
            input_dict,
            {etype: 10 for etype in etypes},
            {etype: idx + 1 for idx, etype in enumerate(etypes)},
            {"name": file_fmt, "delimiter": file_fmt_del},
            file_metadata,
        )

        # Create metadata.json file.
        filename = os.path.join(root_dir, METADATA_NAME)
        with open(filename, "w") as input_handle:
            json.dump(input_dict, input_handle, indent=4)

        # Run Test.
        yield root_dir, etypes, create_files, file_fmt, file_fmt_del, absolute_path
        # Tear down.
        teardown_dataset(filename, edge_dirs, edge_files)


def test_etype_files(etype_files_dataset):
    root_dir = etype_files_dataset[0]
    etypes = etype_files_dataset[1]
    create_files = etype_files_dataset[2]
    edge_fmt = etype_files_dataset[3]
    edge_fmt_del = etype_files_dataset[4]
    absolute_path = etype_files_dataset[5]

    if not create_files and etypes != []:
        with pytest.raises(AssertionError, match="File.*does not exist for.*"):
            obj = MetadataSchema(METADATA_NAME, root_dir)
    elif etypes == []:
        with pytest.raises(AssertionError, match="Invalid number of edges"):
            obj = MetadataSchema(METADATA_NAME, root_dir)
    else:
        obj = MetadataSchema(METADATA_NAME, root_dir)
        for idx, etype in enumerate(etypes):
            file_type, delimiter, data_files = obj.etype_files[etype]
            assert file_type == edge_fmt, f"File type does not match."
            assert delimiter == edge_fmt_del, f"Delimiters does not match."
            assert (
                len(data_files) == idx + 1
            ), f"No. of edge files does not match."
            _assert_filenames(data_files, root_dir, absolute_path)


@pytest.fixture(scope="function")
def node_feats_dataset(
    root_dir,
    ntypes,
    file_fmt,
    file_fmt_del,
    create_files,
    node_feat_dir,
    absolute_path,
):
    input_dict = {}
    dir_names = []
    file_names = []
    # Add nodes to the metadata.
    add_graph_name(input_dict)
    file_metadata = namedtuple("file_metadata", "")
    file_metadata.root_dir = root_dir
    file_metadata.prefix_dir = node_feat_dir
    file_metadata.create_files = create_files
    file_metadata.is_absolute = absolute_path
    file_metadata.file_prefix = "node_feature_"
    node_feat_dirs, node_feat_files = add_node_features(
        input_dict,
        {ntype: 10 for ntype in ntypes},
        {ntype: [NFEAT1] for ntype in ntypes},
        {ntype: 5 for ntype in ntypes},
        {
            ntype + "/" + NFEAT1: {NAME: file_fmt, DELIMITER: file_fmt_del}
            for ntype in ntypes
        },
        file_metadata,
    )
    dir_names.extend(node_feat_dirs)
    file_names.extend(node_feat_files)
    # Add edges and create edge files.
    file_metadata.prefix_dir = "edges"
    file_metadata.create_files = True
    file_metadata.is_absolute = True
    file_metadata.file_prefix = "edges_"
    edge_dirs, edge_files = add_edges_files(
        input_dict,
        {etype: 10 for etype in graph_etypes},
        {etype: 1 for etype in graph_etypes},
        {NAME: "csv", DELIMITER: ","},
        file_metadata,
    )
    dir_names.extend(edge_dirs)
    file_names.extend(edge_files)

    # Create metadata.json file.
    filename = METADATA_NAME
    if root_dir is not None and root_dir != "":
        filename = os.path.join(root_dir, METADATA_NAME)
    with open(filename, "w") as input_handle:
        json.dump(input_dict, input_handle, indent=4)

    # Run Test.
    yield root_dir, ntypes, file_fmt, file_fmt_del, create_files, absolute_path
    # Tear down.
    teardown_dataset(filename, dir_names, file_names)


def test_node_features(node_feats_dataset):
    root_dir = node_feats_dataset[0]
    ntypes = node_feats_dataset[1]
    file_fmt = node_feats_dataset[2]
    file_fmt_del = node_feats_dataset[3]
    create_files = node_feats_dataset[4]
    absolute_path = node_feats_dataset[5]

    if not create_files and ntypes != []:
        with pytest.raises(AssertionError, match=".*File.*does not exist.*"):
            obj = MetadataSchema(METADATA_NAME, root_dir)
    elif ntypes == []:
        with pytest.raises(AssertionError, match="Invalid number of nodes"):
            obj = MetadataSchema(METADATA_NAME, root_dir)
    else:
        obj = MetadataSchema(METADATA_NAME, root_dir)
        for idx, ntype in enumerate(ntypes):
            (
                actual_file_fmt,
                actual_delimiter,
                data_files,
            ) = obj.ntype_feature_files[(ntype, NFEAT1)]
            assert file_fmt == actual_file_fmt, f"File type does not match."
            assert (
                file_fmt_del == actual_delimiter
            ), f"Delimiters does not match."
            assert (
                len(data_files) == 5
            ), f"No. of node feature files does not match."
            _assert_filenames(data_files, root_dir, absolute_path)


@pytest.fixture(scope="function")
def edge_feats_dataset(
    root_dir,
    etypes,
    file_fmt,
    file_fmt_del,
    create_files,
    edge_feat_dir,
    absolute_path,
):
    input_dict = {}
    dir_names = []
    file_names = []
    # Add nodes to the metadata.
    add_graph_name(input_dict)
    add_nodes(input_dict, {ntype: 10 for ntype in graph_ntypes})
    # Add edges and create edge files.
    file_metadata = namedtuple("file_metadata", "")
    file_metadata.root_dir = root_dir
    file_metadata.prefix_dir = "edges"
    file_metadata.create_files = True
    file_metadata.is_absolute = True
    file_metadata.file_prefix = "edges_"
    edge_dirs, edge_files = add_edges_files(
        input_dict,
        {etype: 10 for etype in etypes},
        {etype: 1 for etype in etypes},
        {NAME: file_fmt, DELIMITER: file_fmt_del},
        file_metadata,
    )
    dir_names.extend(edge_dirs)
    file_names.extend(edge_files)
    # Add edge features.
    file_metadata.prefix_dir = edge_feat_dir
    file_metadata.create_files = create_files
    file_metadata.is_absolute = absolute_path
    file_metadata.file_prefix = "edge_feature_"
    feat_dirs, feat_files = add_edge_features(
        input_dict,
        {etype: 10 for etype in etypes},
        {etype: [EFEAT1] for etype in etypes},
        {etype: 5 for etype in etypes},
        {
            etype + "/" + EFEAT1: {NAME: file_fmt, DELIMITER: file_fmt_del}
            for etype in etypes
        },
        file_metadata,
    )
    dir_names.extend(feat_dirs)
    file_names.extend(feat_files)

    # Create metadata.json file.
    filename = METADATA_NAME
    if root_dir is not None and root_dir != "":
        filename = os.path.join(root_dir, METADATA_NAME)
    with open(filename, "w") as input_handle:
        json.dump(input_dict, input_handle, indent=4)

    # Run Test.
    yield root_dir, etypes, file_fmt, file_fmt_del, create_files, absolute_path

    # Tear down.
    teardown_dataset(filename, dir_names, file_names)


def test_edge_features(edge_feats_dataset):
    root_dir = edge_feats_dataset[0]
    etypes = edge_feats_dataset[1]
    file_fmt = edge_feats_dataset[2]
    file_fmt_del = edge_feats_dataset[3]
    create_files = edge_feats_dataset[4]
    absolute_path = edge_feats_dataset[5]

    if not create_files and etypes != []:
        with pytest.raises(AssertionError, match=".*File.*does not exist.*"):
            obj = MetadataSchema(METADATA_NAME, root_dir)
    elif etypes == []:
        with pytest.raises(AssertionError, match="Invalid number of edges"):
            obj = MetadataSchema(METADATA_NAME, root_dir)
    else:
        obj = MetadataSchema(METADATA_NAME, root_dir)
        for idx, etype in enumerate(etypes):
            (
                actual_file_fmt,
                actual_delimiter,
                data_files,
            ) = obj.etype_feature_files[(etype, EFEAT1)]
            assert file_fmt == actual_file_fmt, f"File type does not match."
            assert (
                file_fmt_del == actual_delimiter
            ), f"Delimiters does not match."
            assert (
                len(data_files) == 5
            ), f"No. of edge feature files does not match."
            _assert_filenames(data_files, root_dir, absolute_path)


if __name__ == "__main__":
    test_load_json()
    test_ntypes([NTYPE1])
    test_etypes([ETYPE1])
    test_etype_files()
    test_node_features()
    test_edge_features()
