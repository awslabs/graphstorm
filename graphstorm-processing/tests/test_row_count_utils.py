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

import os
import json
import shutil
import pytest

import pyarrow as pa
import pyarrow.parquet as pq

from graphstorm_processing.graph_loaders.row_count_utils import (
    ParquetRowCounter,
    verify_metadata_match,
)
from graphstorm_processing.constants import FilesystemType

# pylint: disable=redefined-outer-name

_ROOT = os.path.abspath(os.path.dirname(__file__))
TEMP_DATA_PREFIX = os.path.join(_ROOT, "resources/row_counting/generated_parquet/")


def create_feature_table(col_name: str, num_rows: int, feature_start_val: int) -> pa.Table:
    """Creates a test PyArrow table."""
    feature: pa.Array = pa.array(range(feature_start_val, feature_start_val + num_rows))
    return pa.table([feature], names=[col_name])


@pytest.fixture(scope="module")
def test_metadata():
    """Create and return the test metadata structure."""
    return {
        "node_type": ["type1", "type2"],
        "edge_type": ["type1:edge:type2"],
        "node_data": {
            "type1": {
                "feature1": {
                    "format": {"name": "parquet"},
                    "data": [
                        "node_data/type1-feature1/part-00000.parquet",
                        "node_data/type1-feature1/part-00001.parquet",
                    ],
                },
                "feature2": {
                    "format": {"name": "parquet"},
                    "data": [
                        "node_data/type1-feature2/part-00000.parquet",
                        "node_data/type1-feature2/part-00001.parquet",
                    ],
                },
            },
            "type2": {
                "feature1": {
                    "format": {"name": "parquet"},
                    "data": [
                        "node_data/type2-feature1/part-00000.parquet",
                        "node_data/type2-feature1/part-00001.parquet",
                    ],
                }
            },
        },
        "edge_data": {
            "type1:edge:type2": {
                "weight": {
                    "format": {"name": "parquet"},
                    "data": [
                        "edge_data/type1_edge_type2-weight/part-00000.parquet",
                        "edge_data/type1_edge_type2-weight/part-00001.parquet",
                    ],
                }
            }
        },
        "raw_id_mappings": {
            "type1": {
                "format": {"name": "parquet"},
                "data": [
                    "raw_id_mappings/type1/part-00000.parquet",
                    "raw_id_mappings/type1/part-00001.parquet",
                ],
            },
            "type2": {
                "format": {"name": "parquet"},
                "data": [
                    "raw_id_mappings/type2/part-00000.parquet",
                    "raw_id_mappings/type2/part-00001.parquet",
                ],
            },
        },
        "edges": {
            "type1:edge:type2": {
                "format": {"name": "parquet"},
                "data": [
                    "edges/type1_edge_type2/part-00000.parquet",
                    "edges/type1_edge_type2/part-00001.parquet",
                ],
            }
        },
    }


@pytest.fixture(scope="module", autouse=True)
def create_test_files_fixture(test_metadata):
    """Creates test files with known row counts."""
    if os.path.exists(TEMP_DATA_PREFIX):
        shutil.rmtree(TEMP_DATA_PREFIX)
    os.makedirs(TEMP_DATA_PREFIX)

    # Write metadata
    with open(os.path.join(TEMP_DATA_PREFIX, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(test_metadata, f)

    # Create directory structure and files
    for path_type in [
        "node_data",
        "edge_data",
        "edges",
        "raw_id_mappings",
    ]:  # Added raw_id_mappings
        os.makedirs(os.path.join(TEMP_DATA_PREFIX, path_type))

    # Create node data files
    for type_name in ["type1", "type2"]:
        for feature in ["feature1", "feature2"]:
            feature_path = os.path.join(TEMP_DATA_PREFIX, "node_data", f"{type_name}-{feature}")
            os.makedirs(feature_path, exist_ok=True)

            # Create files with different row counts
            pq.write_table(
                create_feature_table(feature, 10, 0),
                os.path.join(feature_path, "part-00000.parquet"),
            )
            pq.write_table(
                create_feature_table(feature, 15, 10),
                os.path.join(feature_path, "part-00001.parquet"),
            )

    # Create raw ID mapping files
    for type_name in ["type1", "type2"]:
        mapping_path = os.path.join(TEMP_DATA_PREFIX, "raw_id_mappings", type_name)
        os.makedirs(mapping_path)
        # Create mapping files with the same row counts as other files
        pq.write_table(
            create_feature_table("id", 10, 0), os.path.join(mapping_path, "part-00000.parquet")
        )
        pq.write_table(
            create_feature_table("id", 15, 10), os.path.join(mapping_path, "part-00001.parquet")
        )

    # Create edge data files
    edge_feat_path = os.path.join(TEMP_DATA_PREFIX, "edge_data", "type1_edge_type2-weight")
    os.makedirs(edge_feat_path)
    pq.write_table(
        create_feature_table("weight", 10, 0), os.path.join(edge_feat_path, "part-00000.parquet")
    )
    pq.write_table(
        create_feature_table("weight", 15, 10), os.path.join(edge_feat_path, "part-00001.parquet")
    )

    # Create edge structure files
    edge_path = os.path.join(TEMP_DATA_PREFIX, "edges", "type1_edge_type2")
    os.makedirs(edge_path)
    pq.write_table(
        create_feature_table("edge", 10, 0), os.path.join(edge_path, "part-00000.parquet")
    )
    pq.write_table(
        create_feature_table("edge", 15, 10), os.path.join(edge_path, "part-00001.parquet")
    )

    yield TEMP_DATA_PREFIX

    # Cleanup
    shutil.rmtree(TEMP_DATA_PREFIX)


@pytest.fixture(scope="module")
def row_counter(test_metadata):
    """Create a ParquetRowCounter instance."""
    return ParquetRowCounter(test_metadata, TEMP_DATA_PREFIX, FilesystemType.LOCAL)


def test_row_counter_initialization(row_counter, test_metadata):
    """Test counter initialization."""
    assert row_counter.metadata_dict == test_metadata
    assert row_counter.output_prefix == TEMP_DATA_PREFIX
    assert row_counter.filesystem_type == FilesystemType.LOCAL


def test_get_row_count_for_single_file(row_counter):
    """Test counting rows in a single file."""
    count = row_counter.get_row_count_for_parquet_file(
        "node_data/type1-feature1/part-00000.parquet"
    )
    assert count == 10


def test_get_row_counts_for_multiple_files(row_counter):
    """Test counting rows across multiple files."""
    counts = row_counter.get_row_counts_for_parquet_files(
        [
            "node_data/type1-feature1/part-00000.parquet",
            "node_data/type1-feature1/part-00001.parquet",
        ]
    )
    assert counts == [10, 15]


def test_add_counts_to_metadata(row_counter, test_metadata):
    """Test adding row counts to metadata."""
    updated_metadata = row_counter.add_row_counts_to_metadata(test_metadata)

    # Check edge counts
    assert "row_counts" in updated_metadata["edges"]["type1:edge:type2"]
    assert updated_metadata["edges"]["type1:edge:type2"]["row_counts"] == [10, 15]

    # Check node feature counts for both types
    assert "row_counts" in updated_metadata["node_data"]["type1"]["feature1"]
    assert updated_metadata["node_data"]["type1"]["feature1"]["row_counts"] == [10, 15]
    assert "row_counts" in updated_metadata["node_data"]["type1"]["feature2"]
    assert "row_counts" in updated_metadata["node_data"]["type2"]["feature1"]


def test_edge_data_row_counts(row_counter, test_metadata):
    """Test the row counts for edge data features."""
    updated_metadata = row_counter.add_row_counts_to_metadata(test_metadata)

    # Check that edge data counts were added
    edge_type = "type1:edge:type2"
    edge_feature = "weight"

    # Verify row counts exist and are correct
    assert "row_counts" in updated_metadata["edge_data"][edge_type][edge_feature]
    assert updated_metadata["edge_data"][edge_type][edge_feature]["row_counts"] == [10, 15]

    # Verify edge data counts match edge structure counts
    edge_feature_counts = updated_metadata["edge_data"][edge_type][edge_feature]["row_counts"]
    edge_structure_counts = updated_metadata["edges"][edge_type]["row_counts"]
    assert edge_feature_counts == edge_structure_counts

    # Test that the total number of rows is correct
    assert sum(edge_feature_counts) == 25


def test_verify_features_and_structure_match():
    """Test verification of feature and structure row counts."""
    structure_meta = {"type1": {"row_counts": [10, 15], "data": ["file1.parquet", "file2.parquet"]}}

    # Test matching counts
    feature_meta = {
        "type1": {"feature1": {"row_counts": [10, 15], "data": ["feat1.parquet", "feat2.parquet"]}}
    }
    assert ParquetRowCounter.verify_features_and_graph_structure_match(feature_meta, structure_meta)

    # Test mismatched counts
    feature_meta["type1"]["feature1"]["row_counts"] = [10, 16]
    assert not ParquetRowCounter.verify_features_and_graph_structure_match(
        feature_meta, structure_meta
    )


def test_verify_all_features_match():
    """Test verification that all features for a type have matching counts."""
    feature_meta = {
        "type1": {
            "feature1": {"row_counts": [10, 15], "data": ["feat1.parquet", "feat2.parquet"]},
            "feature2": {"row_counts": [10, 15], "data": ["feat3.parquet", "feat4.parquet"]},
        }
    }

    # Test matching counts
    assert ParquetRowCounter.verify_all_features_match(feature_meta)

    # Test mismatched counts
    feature_meta["type1"]["feature2"]["row_counts"] = [10, 16]
    assert not ParquetRowCounter.verify_all_features_match(feature_meta)


def test_shared_feature_names(row_counter, test_metadata):
    """Test handling of shared feature names across different types."""
    updated_metadata = row_counter.add_row_counts_to_metadata(test_metadata)

    # Verify both types have row counts for feature1
    assert "row_counts" in updated_metadata["node_data"]["type1"]["feature1"]
    assert "row_counts" in updated_metadata["node_data"]["type2"]["feature1"]

    # Verify the counts are independent
    type1_counts = updated_metadata["node_data"]["type1"]["feature1"]["row_counts"]
    type2_counts = updated_metadata["node_data"]["type2"]["feature1"]["row_counts"]
    assert type1_counts == type2_counts  # Should be [10, 15] for both


def test_verify_metadata_match(row_counter, test_metadata):
    """Test the full metadata verification function."""
    updated_metadata = row_counter.add_row_counts_to_metadata(test_metadata)

    # Test with correct metadata
    assert verify_metadata_match(updated_metadata)

    # Test with corrupted metadata
    corrupted_metadata = updated_metadata.copy()
    corrupted_metadata["node_data"]["type1"]["feature1"]["row_counts"] = [10, 16]
    assert not verify_metadata_match(corrupted_metadata)
