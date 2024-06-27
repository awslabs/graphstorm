"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
from pathlib import Path
import os
import shutil
import sys
from ast import literal_eval
from typing import Callable, List

from numpy.testing import assert_array_equal
import pytest
import pyarrow as pa
from pyarrow import parquet as pq

from graphstorm_processing.repartition_files import ParquetRepartitioner
from graphstorm_processing import repartition_files
from graphstorm_processing.constants import FilesystemType

_ROOT = os.path.abspath(os.path.dirname(__file__))
DUMMY_PREFIX = "s3://dummy_bucket/dummy_prefix"
TEMP_DATA_PREFIX = os.path.join(_ROOT, "resources/repartitioning/generated_parquet/")


def create_feature_table(col_name: str, num_rows: int, feature_start_val: int) -> pa.Table:
    """Creates a PyArrow table with a single column of integer values.

    For example passing `num_rows` 5 and `feature_start_val` 5 will
    return a table with 5 rows, containing [5, 6, 7, 8, 9] as values
    respectively.

    Parameters
    ----------
    col_name : str
        The name of the column of the returned Table
    num_rows : int
        The number of rows the returned table will have.
    feature_start_val : int
        The starting point for the values of the table.

    Returns
    -------
    pa.Table
        A PyArrow Table with one column with increasing integer values.
    """
    feature: pa.Array = pa.array(range(feature_start_val, feature_start_val + num_rows))

    return pa.table([feature], names=[col_name])


@pytest.fixture(scope="module", autouse=True)
def create_parquet_files_fixture():
    """This fixture creates all the Parquet files that are needed by the tests in this module.

    We create files for a graph with 1 edge type, 1 edge label, 2 edge features,
    with 50 edges. We also include 1 node type with 1 node feature.

    Each file source is represented using 5 parquet files, named part-[00000-00004].parquet.
    The files are created such that the most frequent row count between them is
    [10, 10, 10, 10, 10].

    The graph is described by the metadata file at
    `tests/resources/repartitioning/partitioned_metadata.json` that is modeled after
    the metadata file that GSProcessing produces, to be used as input to the distributed
    graph partitioning pipeline.

    Yields
    ------
    str
        The root path under which all the temp files were created.
    """
    # Create the dir for the temp data
    if os.path.exists(TEMP_DATA_PREFIX):
        shutil.rmtree(TEMP_DATA_PREFIX)
    os.mkdir(TEMP_DATA_PREFIX)
    # Copy over the metadata file to the temp directory
    metadata_path = os.path.join(_ROOT, "resources/repartitioning/partitioned_metadata.json")
    shutil.copyfile(metadata_path, os.path.join(TEMP_DATA_PREFIX, "partitioned_metadata.json"))

    # The edge structure files will have different row count from all data files.
    edges_rows = [8, 8, 8, 8, 18]
    # For edge and node data files, the label will have a different row count from the rest
    # and [10, 10, 10, 10, 10] will be the most frequent row count, so all other
    # files will be re-partitioned to match that.
    per_file_rows = [
        [10, 12, 10, 8, 10],
        [10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10],
    ]

    col_names = [
        "label",
        "feature_one",
        "feature_two",
    ]

    # Read the metadata to ensure our generated files will match the expected counts
    with open(
        os.path.join(TEMP_DATA_PREFIX, "partitioned_metadata.json"), "r", encoding="utf-8"
    ) as metafile:
        metadata_dict = json.load(metafile)
    # Check metadata file content against the row counts defined in this function
    for meta_etype in metadata_dict["edge_type"]:
        edge_structure_counts = metadata_dict["edges"][meta_etype]["row_counts"]
        assert edge_structure_counts == edges_rows
        for col_name, generated_rows in zip(col_names, per_file_rows):
            edge_data_counts = metadata_dict["edge_data"][meta_etype][col_name]["row_counts"]
            assert edge_data_counts == generated_rows
    # Do the same for the node files
    for col_name, generated_rows in zip(col_names, per_file_rows):
        edge_data_counts = metadata_dict["node_data"]["src"][col_name]["row_counts"]
        assert edge_data_counts == generated_rows

    # Generate data and write edge data files to disk
    for row_distribution, col_name in zip(per_file_rows, col_names):
        total_rows = 0
        edge_feat_path = os.path.join(
            TEMP_DATA_PREFIX,
            "edge_data",
            f"dummy_type-{col_name}",
            "parquet",
        )
        node_feat_path = os.path.join(
            TEMP_DATA_PREFIX,
            "node_data",
            f"src-{col_name}",
            "parquet",
        )
        os.makedirs(edge_feat_path)
        os.makedirs(node_feat_path)
        for i, row_count in enumerate(row_distribution):
            feat_part = create_feature_table(col_name, row_count, total_rows)
            filename = f"part-{str(i).zfill(5)}.parquet"
            pq.write_table(feat_part, os.path.join(edge_feat_path, filename))
            pq.write_table(feat_part, os.path.join(node_feat_path, filename))
            total_rows += row_count

    # Generate edge structure files
    # The edges will have src_int_ids from 0 to 49, and dst_ids from 50 to 99
    src_col: pa.Array = pa.array(range(0, 50))
    dst_col: pa.Array = pa.array(range(50, 100))
    edges_table = pa.table([src_col, dst_col], names=["src_int_id", "dst_int_id"])

    edges_path = os.path.join(TEMP_DATA_PREFIX, "edges", "dummy_type", "parquet")
    os.makedirs(edges_path)
    total_edges = 0
    for i, edge_rows in enumerate(edges_rows):
        filename = f"part-{str(i).zfill(5)}.parquet"
        pq.write_table(
            edges_table[total_edges : total_edges + edge_rows], os.path.join(edges_path, filename)
        )
        total_edges += edge_rows

    yield TEMP_DATA_PREFIX

    # Cleanup all temp files after the entire module has been tested
    shutil.rmtree(TEMP_DATA_PREFIX)


# Test multiple desired row counts and both partition functions, in-memory and per-file
@pytest.mark.parametrize(
    "desired_counts",
    [[10, 10, 10, 10, 10], [12, 12, 12, 9, 5], [10, 10, 15, 10, 5], [1, 1, 1, 1, 46]],
)
@pytest.mark.parametrize(
    "partition_function_name",
    [
        "_repartition_parquet_files_in_memory",
        "_repartition_parquet_files_streaming",
    ],
)
def test_repartition_functions(desired_counts: List[int], partition_function_name: str):
    """Test the repartition functions, streaming and in-memory"""
    assert sum(desired_counts) == 50

    my_partitioner = ParquetRepartitioner(TEMP_DATA_PREFIX, filesystem_type=FilesystemType.LOCAL)

    metadata_path = os.path.join(TEMP_DATA_PREFIX, "partitioned_metadata.json")

    with open(metadata_path, "r", encoding="utf-8") as metafile:
        metadata_dict = json.load(metafile)

    edge_type_meta = metadata_dict["edges"]["src:dummy_type:dst"]

    # We have parametrized the function name since they have the same signature and result
    partition_function: Callable[[dict, List[int]], dict] = getattr(
        my_partitioner, partition_function_name
    )
    updated_meta = partition_function(edge_type_meta, desired_counts)

    # Ensure we got the correct number of files, with the correct number of rows reported
    assert updated_meta["row_counts"] == desired_counts
    assert len(updated_meta["data"]) == len(desired_counts)

    # Ensure actual row counts match to expectation
    for expected_count, result_filepath in zip(desired_counts, updated_meta["data"]):
        assert (
            expected_count
            == pq.read_metadata(os.path.join(TEMP_DATA_PREFIX, result_filepath)).num_rows
        )

    # Ensure order/content of rows matches to expectation
    original_table = (
        pq.read_table(os.path.join(TEMP_DATA_PREFIX, Path(edge_type_meta["data"][0]).parent))
        .to_pandas()
        .to_numpy()
    )
    repartitioned_table = (
        pq.read_table(os.path.join(TEMP_DATA_PREFIX, Path(updated_meta["data"][0]).parent))
        .to_pandas()
        .to_numpy()
    )

    assert_array_equal(original_table, repartitioned_table)


# TODO: Add simple tests for the load functions


def test_verify_metadata_only_edge_data():
    """Ensure verify_metadata works as expected when provided with
    just edge structure and edge data
    """
    with open(
        os.path.join(TEMP_DATA_PREFIX, "partitioned_metadata.json"),
        "r",
        encoding="utf-8",
    ) as metafile:
        original_metadata_dict = json.load(metafile)

    # Ensure failure with wrong edges vs edge data counts
    with pytest.raises(RuntimeError):
        repartition_files.verify_metadata(
            original_metadata_dict["edges"], original_metadata_dict["edge_data"]
        )

    row_counts = [10, 10, 10, 10, 10]
    original_metadata_dict["edge_data"]["src:dummy_type:dst"]["label"]["row_counts"] = row_counts
    original_metadata_dict["edges"]["src:dummy_type:dst"]["row_counts"] = row_counts
    original_metadata_dict["edges"].pop("dst:dummy_type-rev:src")
    original_metadata_dict["edge_data"].pop("dst:dummy_type-rev:src")

    # Ensure success when counts match
    repartition_files.verify_metadata(
        original_metadata_dict["edges"], edge_data_meta=original_metadata_dict["edge_data"]
    )


def test_verify_metadata_with_node_data():
    """Ensure verify_metadata works as expected when provided with edge structure and node data"""
    with open(
        os.path.join(TEMP_DATA_PREFIX, "partitioned_metadata.json"),
        "r",
        encoding="utf-8",
    ) as metafile:
        original_metadata_dict = json.load(metafile)

    original_metadata_dict["node_data"] = {
        "node_type_1": {
            "feature_1": {
                "row_counts": [1, 2, 3],
            },
            "feature_2": {
                "row_counts": [3, 2, 1],
            },
        }
    }

    # Ensure failure with wrong counts
    with pytest.raises(RuntimeError):
        repartition_files.verify_metadata(
            original_metadata_dict["edges"], node_data_meta=original_metadata_dict["node_data"]
        )

    original_metadata_dict["node_data"] = {
        "node_type_1": {
            "feature_1": {
                "row_counts": [1, 2, 3],
            },
            "feature_2": {
                "row_counts": [1, 2, 3],
            },
        }
    }

    # Ensure success when counts match
    repartition_files.verify_metadata(
        original_metadata_dict["edges"], node_data_meta=original_metadata_dict["node_data"]
    )

    # Ensure failure with right node counts but wrong edge data counts
    with pytest.raises(RuntimeError):
        repartition_files.verify_metadata(
            original_metadata_dict["edges"],
            edge_data_meta=original_metadata_dict["edge_data"],
            node_data_meta=original_metadata_dict["node_data"],
        )


def test_collect_frequencies_for_data_counts():
    """Test functionality for collect_frequencies_for_data_counts"""
    input_dict = {
        "type_name_1": {
            "feature_1": {
                "row_counts": [1, 2, 3],
            },
            "feature_2": {
                "row_counts": [1, 2, 3],
            },
        },
        "type_name_2": {
            "feature_1": {
                "row_counts": [2, 2, 2],
            },
            "feature_2": {
                "row_counts": [1, 2, 3],
            },
        },
    }

    frequencies = repartition_files.collect_frequencies_for_data_counts(input_dict)

    assert frequencies.keys() == {"type_name_1", "type_name_2"}

    assert frequencies["type_name_1"] == {(1, 2, 3): 2}
    assert frequencies["type_name_2"] == {(2, 2, 2): 1, (1, 2, 3): 1}


# Integration test, run for different task types
@pytest.mark.parametrize(
    "task_type",
    [
        "edge_class",
        "link_predict",
        "link_prediction",
        "node_class",
    ],
)
def test_repartition_files_integration(monkeypatch, task_type):
    """Integration test for repartition script"""
    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            [
                "repartition_files.py",
                "--input-prefix",
                TEMP_DATA_PREFIX,
                "--input-metadata-file-name",
                "partitioned_metadata.json",
                "--updated-metadata-file-name",
                "updated_row_counts_metadata.json",
            ],
        )
        # We monkeypatch json.load to inject the task_type into the metadata
        orig_json_load = json.load

        def mock_json_load(file):
            orig_dict = orig_json_load(file)
            orig_dict["graph_info"]["task_type"] = task_type
            return orig_dict

        monkeypatch.setattr(json, "load", mock_json_load)

        # Execute main function to test side-effects
        repartition_files.main()

        with open(
            os.path.join(TEMP_DATA_PREFIX, "updated_row_counts_metadata.json"),
            "r",
            encoding="utf-8",
        ) as metafile:
            new_metadata_dict = json.load(metafile)

        # The most popular counts are all 10 rows
        expected_counts = [10, 10, 10, 10, 10]

        reported_edge_counts = new_metadata_dict["edges"]["src:dummy_type:dst"]["row_counts"]

        # Ensure all edge structure files have the correct reported counts
        assert reported_edge_counts == expected_counts

        edge_type = None
        for edge_type in new_metadata_dict["edge_type"]:
            edge_struct_files = new_metadata_dict["edges"][edge_type]["data"]
            # Ensure all edge structure files have the correct actual counts
            for expected_count, edge_struct_filepath in zip(expected_counts, edge_struct_files):
                absolute_edge_filepath = os.path.join(TEMP_DATA_PREFIX, edge_struct_filepath)
                assert expected_count == pq.read_metadata(absolute_edge_filepath).num_rows

            # Ensure all feature files have the correct counts, reported and actual
            for feature_name, edge_feature_dict in new_metadata_dict["edge_data"][
                edge_type
            ].items():
                assert edge_feature_dict["row_counts"] == expected_counts
                for expected_count, feature_filepath in zip(
                    expected_counts, edge_feature_dict["data"]
                ):
                    absolute_feature_filepath = os.path.join(TEMP_DATA_PREFIX, feature_filepath)
                    filemeta = pq.read_metadata(absolute_feature_filepath)
                    file_rows = filemeta.num_rows
                    assert (
                        expected_count == file_rows
                    ), f"Count mismatch for {feature_name}, {absolute_feature_filepath}"
                    # Ensure the flat array has the correct metadata embedded
                    if feature_name == "label" and task_type not in {
                        "link_predict",
                        "link_prediction",
                    }:
                        arrow_meta = filemeta.schema.to_arrow_schema().metadata
                        bshape = arrow_meta.get(b"shape", None)
                        shape = tuple(literal_eval(bshape.decode()))
                        assert shape == (file_rows,)

        # Do the same for node data
        for feature_name, node_feature_dict in new_metadata_dict["node_data"]["src"].items():
            assert node_feature_dict["row_counts"] == expected_counts
            for expected_count, feature_filepath in zip(expected_counts, node_feature_dict["data"]):
                absolute_feature_filepath = os.path.join(TEMP_DATA_PREFIX, feature_filepath)
                filemeta = pq.read_metadata(absolute_feature_filepath)
                file_rows = filemeta.num_rows
                assert (
                    expected_count == file_rows
                ), f"Count mismatch for {feature_name}, {absolute_feature_filepath}"
                if feature_name == "label":
                    arrow_meta = filemeta.schema.to_arrow_schema().metadata
                    bshape = arrow_meta.get(b"shape", None)
                    shape = tuple(literal_eval(bshape.decode()))
                    assert shape == (file_rows,)
