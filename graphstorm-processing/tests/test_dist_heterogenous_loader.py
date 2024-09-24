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

from typing import Any, Dict, Optional
import json
import math
import os
import shutil
import tempfile

from numpy.testing import assert_allclose
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
import pyarrow.parquet as pq
import pytest

from graphstorm_processing.graph_loaders.dist_heterogeneous_loader import (
    DistHeterogeneousGraphLoader,
    HeterogeneousLoaderConfig,
    NODE_MAPPING_INT,
    NODE_MAPPING_STR,
)
from graphstorm_processing.data_transformations.dist_label_loader import SplitRates
from graphstorm_processing.config.label_config_base import (
    NodeLabelConfig,
    EdgeLabelConfig,
)
from graphstorm_processing.config.config_parser import (
    create_config_objects,
    EdgeConfig,
)
from graphstorm_processing.config.config_conversion import GConstructConfigConverter
from graphstorm_processing.constants import (
    COLUMN_NAME,
    MIN_VALUE,
    MAX_VALUE,
    VALUE_COUNTS,
    TRANSFORMATIONS_FILENAME,
)

pytestmark = pytest.mark.usefixtures("spark")
_ROOT = os.path.abspath(os.path.dirname(__file__))
STR_LABEL_COL = "str_label"
NUM_LABEL_COL = "num_label"
NUM_DATAPOINTS = 10000

NODE_CLASS_GRAPHINFO_UPDATES = {
    "nfeat_size": {
        "user": {
            "age": 1,
            "attention_mask": 16,
            "input_ids": 16,
            "token_type_ids": 16,
            "multi": 2,
            "no-op-truncated": 1,
            "state": 3,
        }
    },
    "efeat_size": {},
    "etype_label": [],
    "etype_label_property": [],
    "ntype_label": ["user"],
    "ntype_label_property": ["gender"],
    "task_type": "node_class",
    "label_map": {"male": 0, "female": 1},
    "label_properties": {
        "user": {
            "COLUMN_NAME": "gender",
            "VALUE_COUNTS": {"male": 3, "female": 1, "null": 1},
        }
    },
    "ntype_to_label_masks": {"user": ["train_mask", "val_mask", "test_mask"]},
    "etype_to_label_masks": {},
}


@pytest.fixture(autouse=True, name="tempdir")
def tempdir_fixture():
    """Create temp dir for output files"""
    tempdirectory = tempfile.mkdtemp(
        prefix=os.path.join(_ROOT, "resources/test_output/"),
    )
    yield tempdirectory
    shutil.rmtree(tempdirectory)


@pytest.fixture(scope="function", name="data_configs_with_label")
def data_configs_with_label_fixture():
    """Create data configuration object that contain features and labels"""
    config_path = os.path.join(
        _ROOT, "resources/small_heterogeneous_graph/gsprocessing-config.json"
    )

    with open(config_path, "r", encoding="utf-8") as conf_file:
        gsprocessing_config = json.load(conf_file)

    data_configs_dict = create_config_objects(gsprocessing_config["graph"])

    return data_configs_dict


@pytest.fixture(scope="function", name="no_label_data_configs")
def no_label_data_configs_fixture():
    """Create data configuration object without labels"""
    config_path = os.path.join(
        _ROOT, "resources/small_heterogeneous_graph/gconstruct-no-labels-config.json"
    )

    with open(config_path, "r", encoding="utf-8") as conf_file:
        gconstruct_config = json.load(conf_file)
        gsprocessing_config = GConstructConfigConverter().convert_to_gsprocessing(gconstruct_config)

    data_configs_dict = create_config_objects(gsprocessing_config["graph"])

    return data_configs_dict


@pytest.fixture(scope="function", name="dghl_loader")
def dghl_loader_fixture(spark, data_configs_with_label, tempdir) -> DistHeterogeneousGraphLoader:
    """Create a re-usable loader that includes labels"""
    input_path = os.path.join(_ROOT, "resources/small_heterogeneous_graph")
    loader_config = HeterogeneousLoaderConfig(
        add_reverse_edges=True,
        data_configs=data_configs_with_label,
        enable_assertions=True,
        graph_name="small_heterogeneous_graph",
        input_prefix=input_path,
        local_input_path=input_path,
        local_metadata_output_path=tempdir,
        num_output_files=1,
        output_prefix=tempdir,
        precomputed_transformations={},
    )
    dhgl = DistHeterogeneousGraphLoader(
        spark,
        loader_config=loader_config,
    )
    return dhgl


@pytest.fixture(scope="function", name="dghl_loader_no_label")
def dghl_loader_no_label_fixture(
    spark, no_label_data_configs, tempdir
) -> DistHeterogeneousGraphLoader:
    """Create a re-usable loader without labels"""
    input_path = os.path.join(_ROOT, "resources/small_heterogeneous_graph")
    loader_config = HeterogeneousLoaderConfig(
        add_reverse_edges=True,
        data_configs=no_label_data_configs,
        enable_assertions=True,
        graph_name="small_heterogeneous_graph",
        input_prefix=input_path,
        local_input_path=input_path,
        local_metadata_output_path=tempdir,
        num_output_files=1,
        output_prefix=tempdir,
        precomputed_transformations={},
    )
    dhgl = DistHeterogeneousGraphLoader(
        spark,
        loader_config,
    )
    return dhgl


@pytest.fixture(scope="function", name="dghl_loader_no_reverse_edges")
def dghl_loader_no_reverse_edges_fixture(
    spark, data_configs_with_label, tempdir
) -> DistHeterogeneousGraphLoader:
    """Create a re-usable loader that doesn't produce reverse edegs"""
    input_path = os.path.join(_ROOT, "resources/small_heterogeneous_graph")
    loader_config = HeterogeneousLoaderConfig(
        add_reverse_edges=False,
        data_configs=data_configs_with_label,
        enable_assertions=True,
        graph_name="small_heterogeneous_graph",
        input_prefix=input_path,
        local_input_path=input_path,
        local_metadata_output_path=tempdir,
        num_output_files=1,
        output_prefix=tempdir,
        precomputed_transformations={},
    )
    dhgl = DistHeterogeneousGraphLoader(
        spark,
        loader_config,
    )
    return dhgl


@pytest.fixture(scope="session", name="user_rated_movie_df")
def user_rated_movie_df_fixture(spark: SparkSession) -> DataFrame:
    """User rated movie edges DataFrame"""
    data_path = os.path.join(
        _ROOT, "resources/small_heterogeneous_graph/edges/user-rated-movie.csv"
    )
    edges_df = spark.read.csv(data_path, header=True, inferSchema=True)
    return edges_df


def verify_integ_test_output(
    metadata: Dict[str, Dict],
    loader: DistHeterogeneousGraphLoader,
    graphinfo_updates: Dict[str, Any],
) -> None:
    """Verifies the output of integration tests with expected data"""
    assert metadata["num_nodes_per_type"] == [3, 2, 4, 5]
    assert metadata["num_edges_per_type"] == [4, 4, 6, 6, 4, 4]
    assert metadata["node_type"] == ["director", "genre", "movie", "user"]
    assert metadata["edge_type"] == [
        "movie:included_in:genre",
        "genre:included_in-rev:movie",
        "user:rated:movie",
        "movie:rated-rev:user",
        "director:directed:movie",
        "movie:directed-rev:director",
    ]

    expected_node_counts = {"director": 3, "genre": 2, "movie": 4, "user": 5}
    # TODO: The following Parquet reads assume there's only one file in the output
    for node_type in metadata["node_type"]:
        nrows = pq.read_table(
            os.path.join(
                loader.output_path,
                os.path.dirname(metadata["raw_id_mappings"][node_type]["data"][0]),
            )
        ).num_rows
        assert nrows == expected_node_counts[node_type]

    expected_edge_counts = {
        "movie:included_in:genre": 4,
        "genre:included_in-rev:movie": 4,
        "user:rated:movie": 6,
        "movie:rated-rev:user": 6,
        "director:directed:movie": 4,
        "movie:directed-rev:director": 4,
    }

    for edge_type in metadata["edge_type"]:
        nrows = pq.read_table(
            os.path.join(
                loader.output_path,
                os.path.dirname(metadata["edges"][edge_type]["data"][0]),
            )
        ).num_rows
        assert nrows == expected_edge_counts[edge_type]

    shared_expected_graphinfo = {
        "graph_type": "heterogeneous",
        "num_nodes": 14,
        "num_edges": 28,
        "num_rels": 6,
        "num_ntype": 4,
        "num_labels": 1,
        "num_nodes_ntype": expected_node_counts,
        "num_edges_etype": expected_edge_counts,
        "is_multilabel": False,
    }

    shared_expected_graphinfo.update(graphinfo_updates)

    actual_graphinfo = metadata["graph_info"]
    for actual_key, actual_val in actual_graphinfo.items():
        assert (
            actual_key in shared_expected_graphinfo
        ), f"Key '{actual_key}' not found in expected graphinfo"
        assert (
            actual_val == shared_expected_graphinfo[actual_key]
        ), f"Value for '{actual_key}' does not match"


# TODO: Test with forced multiple partitions, with autogenerated data.
def test_load_dist_heterogen_node_class(dghl_loader: DistHeterogeneousGraphLoader):
    """End 2 end test for node classification"""
    dghl_loader.load()

    with open(
        os.path.join(dghl_loader.output_path, "metadata.json"), "r", encoding="utf-8"
    ) as mfile:
        metadata = json.load(mfile)

    verify_integ_test_output(metadata, dghl_loader, NODE_CLASS_GRAPHINFO_UPDATES)

    expected_node_data = {
        "user": {
            "gender",
            "train_mask",
            "val_mask",
            "test_mask",
            "age",
            "multi",
            "no-op-truncated",
            "state",
            "input_ids",
            "attention_mask",
            "token_type_ids",
        },
    }

    for node_type in metadata["node_data"]:
        assert metadata["node_data"][node_type].keys() == expected_node_data[node_type]

    with open(
        os.path.join(dghl_loader.output_path, TRANSFORMATIONS_FILENAME),
        "r",
        encoding="utf-8",
    ) as transformation_file:
        transformations_dict = json.load(transformation_file)

        assert "user" in transformations_dict["node_features"]
        assert "state" in transformations_dict["node_features"]["user"]


def test_load_dist_hgl_without_labels(
    dghl_loader_no_label: DistHeterogeneousGraphLoader,
):
    """End 2 end test when no labels are provided"""
    dghl_loader_no_label.load()

    with open(
        os.path.join(dghl_loader_no_label.output_path, "metadata.json"),
        "r",
        encoding="utf-8",
    ) as mfile:
        metadata = json.load(mfile)

    graphinfo_updates = {
        "nfeat_size": {},
        "efeat_size": {},
        "task_type": "link_prediction",
        "etype_label": [],
        "etype_label_property": [],
        "ntype_label": [],
        "ntype_label_property": [],
        "label_map": {},
        "label_properties": {},
        "ntype_to_label_masks": {},
        "etype_to_label_masks": {},
    }

    verify_integ_test_output(metadata, dghl_loader_no_label, graphinfo_updates)

    expected_edge_data = {}

    assert metadata["edge_data"] == expected_edge_data


def test_write_edge_structure_no_reverse_edges(
    data_configs_with_label,
    dghl_loader_no_reverse_edges: DistHeterogeneousGraphLoader,
    user_rated_movie_df: DataFrame,
):
    """Writing edge structure when no reverse edges are requested"""
    edge_configs = data_configs_with_label["edges"]

    # We need these two for the side-effects of creating the mappings
    missing_node_types = dghl_loader_no_reverse_edges._get_missing_node_types(edge_configs, [])
    dghl_loader_no_reverse_edges.create_node_id_maps_from_edges(edge_configs, missing_node_types)

    edge_dict: Dict[str, Dict] = {
        "data": {
            "format": "csv",
            "files": ["edges/user-rated-movie.csv"],
            "separator": ",",
        },
        "source": {"column": "~from", "type": "user"},
        "relation": {"type": "rated"},
        "dest": {"column": "~to", "type": "movie"},
    }

    edge_config = EdgeConfig(edge_dict, edge_dict["data"])
    _, _, rev_edge_path_list = dghl_loader_no_reverse_edges.write_edge_structure(
        user_rated_movie_df, edge_config
    )

    assert len(rev_edge_path_list) == 0


def test_create_all_mapppings_from_edges(
    spark: SparkSession,
    data_configs_with_label,
    dghl_loader: DistHeterogeneousGraphLoader,
):
    """Test creating all node mappings only from edge files"""
    edge_configs = data_configs_with_label["edges"]

    missing_node_types = dghl_loader._get_missing_node_types(edge_configs, [])
    assert missing_node_types == {"user", "director", "genre", "movie"}
    dghl_loader.create_node_id_maps_from_edges(edge_configs, missing_node_types)

    expected_node_counts = {
        "user": 5,
        "director": 3,
        "genre": 2,
        "movie": 4,
    }

    assert len(dghl_loader.node_mapping_paths) == 4
    for node_type, mapping_files in dghl_loader.node_mapping_paths.items():
        files_with_prefix = [os.path.join(dghl_loader.output_path, x) for x in mapping_files]
        mapping_count = spark.read.parquet(*files_with_prefix).count()
        assert mapping_count == expected_node_counts[node_type]


def test_create_some_mapppings_from_edges(
    data_configs_with_label, dghl_loader: DistHeterogeneousGraphLoader
):
    """Test creating only some node mappings from edge files"""
    edge_configs = data_configs_with_label["edges"]

    dghl_loader.create_node_id_maps_from_edges(edge_configs, missing_node_types={"director"})

    assert len(dghl_loader.node_mapping_paths) == 1


def test_extend_mapping_from_edges(spark: SparkSession, dghl_loader: DistHeterogeneousGraphLoader):
    """Test extending an existing node mapping from a new edge source"""
    existing_mapping_data = [
        ("node_id_1", 0),
        ("node_id_2", 1),
        ("node_id_3", 2),
    ]
    edges_data = [
        ("node_id_1", "dst_id_1", "edge_feature_val1"),
        ("node_id_4", "dst_id_2", "edge_feature_val2"),
        ("node_id_3", "dst_id_3", "edge_feature_val3"),
    ]

    edge_columns = ["src", "dst", "feature"]
    edges_df = spark.createDataFrame(edges_data, schema=edge_columns)
    existing_mapping_df = spark.createDataFrame(
        existing_mapping_data, schema=[NODE_MAPPING_STR, NODE_MAPPING_INT]
    )

    node_df_with_ids = dghl_loader._extend_mapping_from_edges(existing_mapping_df, edges_df, "src")

    assert node_df_with_ids.select(NODE_MAPPING_INT).distinct().count() == 4

    id_list = [x[NODE_MAPPING_INT] for x in node_df_with_ids.select(NODE_MAPPING_INT).collect()]

    assert id_list == [0, 1, 2, 3]


def test_extend_mapping_from_edges_with_incoming_missing(
    spark: SparkSession, dghl_loader: DistHeterogeneousGraphLoader
):
    """Test extending an existing node mapping from a new edge source
    that doesn't have all nodes already in the mapping"""
    existing_mapping_data = [
        ("node_id_1", 0),
        ("node_id_2", 1),
        ("node_id_3", 2),
        ("node_id_4", 3),
    ]
    edges_data = [
        ("node_id_1", "dst_id_1", "edge_feature_val1"),
        ("node_id_2", "dst_id_2", "edge_feature_val2"),
        ("node_id_3", "dst_id_3", "edge_feature_val3"),
    ]

    edge_columns = ["src", "dst", "feature"]
    edges_df = spark.createDataFrame(edges_data, schema=edge_columns)
    existing_mapping_df = spark.createDataFrame(
        existing_mapping_data, schema=[NODE_MAPPING_STR, NODE_MAPPING_INT]
    )

    node_df_with_ids = dghl_loader._extend_mapping_from_edges(existing_mapping_df, edges_df, "src")

    assert node_df_with_ids.select(NODE_MAPPING_INT).distinct().count() == 4

    id_list = [x[NODE_MAPPING_INT] for x in node_df_with_ids.select(NODE_MAPPING_INT).collect()]

    assert id_list == [0, 1, 2, 3]


def create_edges_df(spark: SparkSession, missing_data_points: int) -> DataFrame:
    """Create an edges DF for testing"""
    edges_data = [("src_id_val", "dst_id_val", "edge_feature_val", "label_val")] * NUM_DATAPOINTS

    # Set certain number of datapoints to be missing
    edges_data[:missing_data_points] = [
        ("src_id_val", "dst_id_val", "edge_feature_val", "")
    ] * missing_data_points

    edge_columns = ["src", "dst", "feature", STR_LABEL_COL]
    return spark.createDataFrame(edges_data, schema=edge_columns)


def create_edges_df_num_label(spark: SparkSession, missing_data_points: int) -> DataFrame:
    """Create an edges DF with a numeric and a string label for testing"""
    edges_data = [("src_id_val", "dst_id_val", -1, "label_val_1")] * NUM_DATAPOINTS
    edges_data[: (NUM_DATAPOINTS // 2)] = [("src_id_val", "dst_id_val", 1, "label_val_2")] * (
        NUM_DATAPOINTS // 2
    )

    # Set certain number of datapoints to have missing label values
    edges_data[:missing_data_points] = [("src_id_val", "dst_id_val", 0, "")] * missing_data_points

    edge_columns = ["src", "dst", NUM_LABEL_COL, STR_LABEL_COL]
    df = spark.createDataFrame(edges_data, schema=edge_columns)
    df = df.withColumn(NUM_LABEL_COL, F.col(NUM_LABEL_COL).cast("float").alias(NUM_LABEL_COL))
    return df


def ensure_masks_are_correct(
    train_mask_df: DataFrame,
    val_mask_df: DataFrame,
    test_mask_df: DataFrame,
    expected_data_points: int,
    requested_rates: dict[str, float],
    mask_names: Optional[list[str]] = None,
):
    """Check the correctness of train/val/test maps"""

    def sum_col(in_df: DataFrame, col_name="mask"):
        """Sum the values in a column"""
        return in_df.agg(F.sum(col_name)).collect()[0][0]

    if mask_names is None:
        mask_names = [
            "train_mask",
            "val_mask",
            "test_mask",
        ]

    train_mask_sum = sum_col(train_mask_df, mask_names[0])
    val_mask_sum = sum_col(val_mask_df, mask_names[1])
    test_mask_sum = sum_col(test_mask_df, mask_names[2])

    sum_counts = train_mask_sum + test_mask_sum + val_mask_sum

    # Ensure all masks have the correct number of rows
    assert train_mask_df.count() == test_mask_df.count() == val_mask_df.count() == NUM_DATAPOINTS

    # Ensure the total number of 1's sums to the number of expected datapoints
    if math.fsum(requested_rates.values()) == 1.0:
        assert sum_counts == expected_data_points

    # TODO: Can we tighten tolerance without causing inadvertent test failures?
    # Approximately check split ratio requested vs. actual +/- 0.1
    assert_allclose(train_mask_sum / sum_counts, requested_rates["train"], atol=0.1)
    assert_allclose(val_mask_sum / sum_counts, requested_rates["val"], atol=0.1)
    assert_allclose(test_mask_sum / sum_counts, requested_rates["test"], atol=0.1)


def read_masks_from_disk(
    spark: SparkSession,
    loader: DistHeterogeneousGraphLoader,
    output_dicts: dict,
    mask_names: Optional[list[str]] = None,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Helper function to read mask DFs from disk.

    Parameters
    ----------
    spark : SparkSession
        Active SparkSession
    loader : DistHeterogeneousGraphLoader
        Graph loader from which we get path root path
    output_dicts : dict
        The output dicts from which we get relative filepaths
    mask_names : list[str], optional
        List of train, val, test mask names, by default None

    Returns
    -------
    Tuple[DataFrame, DataFrame, DataFrame]
        Train mask, val mask, test mask DFs.
    """
    if not mask_names:
        mask_names = [
            "train_mask",
            "val_mask",
            "test_mask",
        ]
    train_mask_df = spark.read.parquet(
        *[
            os.path.join(loader.output_prefix, rel_path)
            for rel_path in output_dicts[mask_names[0]]["data"]
        ]
    )
    val_mask_df = spark.read.parquet(
        *[
            os.path.join(loader.output_prefix, rel_path)
            for rel_path in output_dicts[mask_names[1]]["data"]
        ]
    )
    test_mask_df = spark.read.parquet(
        *[
            os.path.join(loader.output_prefix, rel_path)
            for rel_path in output_dicts[mask_names[2]]["data"]
        ]
    )

    return train_mask_df, val_mask_df, test_mask_df


@pytest.mark.parametrize("missing_label_percentage", [0.0, 0.2])
def test_create_split_files_from_rates(
    spark: SparkSession,
    dghl_loader: DistHeterogeneousGraphLoader,
    tempdir,
    missing_label_percentage: float,
):
    """Test creating split files from provided rates, with missing labels"""
    split_rates = SplitRates(0.8, 0.1, 0.1)
    missing_data_points = int(NUM_DATAPOINTS * missing_label_percentage)
    non_missing_data_points = NUM_DATAPOINTS - missing_data_points
    edges_df = create_edges_df(spark, missing_data_points)

    output_dicts = dghl_loader._create_split_files(
        edges_df,
        STR_LABEL_COL,
        split_rates,
        os.path.join(tempdir, "sample_masks"),
        None,
        seed=42,
    )

    train_mask_df, val_mask_df, test_mask_df = read_masks_from_disk(
        spark, dghl_loader, output_dicts
    )

    ensure_masks_are_correct(
        train_mask_df,
        val_mask_df,
        test_mask_df,
        non_missing_data_points,
        split_rates.todict(),
    )


def test_at_least_one_label_exists(no_label_data_configs, data_configs_with_label):
    """Test the functionality of _at_least_one_label_exists"""
    assert not DistHeterogeneousGraphLoader._at_least_one_label_exists(no_label_data_configs)

    assert DistHeterogeneousGraphLoader._at_least_one_label_exists(data_configs_with_label)


def test_create_split_files_from_rates_empty_col(
    spark: SparkSession, dghl_loader: DistHeterogeneousGraphLoader, tempdir
):
    """Test creating split files, even when label column is empty"""
    edges_df = create_edges_df(spark, 0).drop(STR_LABEL_COL)
    split_rates = SplitRates(0.8, 0.1, 0.1)

    output_dicts = dghl_loader._create_split_files(
        edges_df, "", split_rates, os.path.join(tempdir, "sample_masks"), None, seed=42
    )

    train_mask_df, val_mask_df, test_mask_df = read_masks_from_disk(
        spark, dghl_loader, output_dicts
    )

    ensure_masks_are_correct(
        train_mask_df, val_mask_df, test_mask_df, NUM_DATAPOINTS, split_rates.todict()
    )


def test_process_edge_labels_link_prediction(
    spark: SparkSession, dghl_loader: DistHeterogeneousGraphLoader
):
    """Test processing link prediction edge labels"""
    edges_df = create_edges_df(spark, 0)
    split_rates = {"train": 0.8, "val": 0.1, "test": 0.1}

    config_dict = {
        "column": "",
        "type": "link_prediction",
        "split_rate": split_rates,
    }
    label_configs = [EdgeLabelConfig(config_dict)]

    label_metadata_dicts = dghl_loader._process_edge_labels(
        label_configs, edges_df, "src:to:dst", ""
    )

    # For link prediction only the masks should be produced
    assert label_metadata_dicts.keys() == {"train_mask", "test_mask", "val_mask"}

    train_mask_df, val_mask_df, test_mask_df = read_masks_from_disk(
        spark, dghl_loader, label_metadata_dicts
    )

    ensure_masks_are_correct(train_mask_df, val_mask_df, test_mask_df, NUM_DATAPOINTS, split_rates)


def test_process_edge_labels_multitask(
    spark: SparkSession, dghl_loader: DistHeterogeneousGraphLoader
):
    """Test processing multi-task link prediction and regression edge labels"""
    edges_df = create_edges_df_num_label(spark, 0)
    lp_split_rates = {"train": 0.6, "val": 0.2, "test": 0.2}
    reg_split_rates = {"train": 0.8, "val": 0.1, "test": 0.1}

    lp_mask_names = [
        "lp_train_mask",
        "lp_val_mask",
        "lp_test_mask",
    ]
    lp_config_dict = {
        "column": "",
        "type": "link_prediction",
        "split_rate": lp_split_rates,
        "mask_field_names": lp_mask_names,
    }
    reg_mask_names = [
        f"train_mask_{NUM_LABEL_COL}",
        f"val_mask_{NUM_LABEL_COL}",
        f"test_mask_{NUM_LABEL_COL}",
    ]
    regression_config_dict = {
        "column": NUM_LABEL_COL,
        "type": "regression",
        "mask_field_names": reg_mask_names,
    }
    label_configs = [
        EdgeLabelConfig(lp_config_dict),
        EdgeLabelConfig(regression_config_dict),
    ]

    label_metadata_dicts = dghl_loader._process_edge_labels(
        label_configs, edges_df, "src:to:dst", "to"
    )

    # For multi-task link prediction and regression, each task should
    # have its own custom mask names, and we should have one entry
    # for the regression label itself
    assert label_metadata_dicts.keys() == {
        *lp_mask_names,
        *reg_mask_names,
        NUM_LABEL_COL,
    }

    # Check LP mask correctness
    train_mask_df, val_mask_df, test_mask_df = read_masks_from_disk(
        spark,
        dghl_loader,
        label_metadata_dicts,
        mask_names=lp_mask_names,
    )
    ensure_masks_are_correct(
        train_mask_df,
        val_mask_df,
        test_mask_df,
        NUM_DATAPOINTS,
        lp_split_rates,
        lp_mask_names,
    )
    # Check regression mask correctness
    train_mask_df, val_mask_df, test_mask_df = read_masks_from_disk(
        spark,
        dghl_loader,
        label_metadata_dicts,
        mask_names=reg_mask_names,
    )
    ensure_masks_are_correct(
        train_mask_df,
        val_mask_df,
        test_mask_df,
        NUM_DATAPOINTS,
        reg_split_rates,
        reg_mask_names,
    )


def test_process_node_labels_multitask(
    spark: SparkSession, dghl_loader: DistHeterogeneousGraphLoader
):
    """Test processing multi-task link prediction and regression edge labels"""
    nodes_df = create_edges_df_num_label(spark, 0)
    class_split_rates = {"train": 0.6, "val": 0.3, "test": 0.1}
    reg_split_rates = {"train": 0.7, "val": 0.2, "test": 0.1}

    class_mask_names = [
        f"{STR_LABEL_COL}_train_mask",
        f"{STR_LABEL_COL}_val_mask",
        f"{STR_LABEL_COL}_test_mask",
    ]
    class_config_dict = {
        "column": STR_LABEL_COL,
        "type": "classification",
        "split_rate": class_split_rates,
        "mask_field_names": class_mask_names,
    }
    reg_mask_names = [
        f"train_mask_{NUM_LABEL_COL}",
        f"val_mask_{NUM_LABEL_COL}",
        f"test_mask_{NUM_LABEL_COL}",
    ]
    regression_config_dict = {
        "column": NUM_LABEL_COL,
        "type": "regression",
        "split_rate": reg_split_rates,
        "mask_field_names": reg_mask_names,
    }
    label_configs = [
        NodeLabelConfig(class_config_dict),
        NodeLabelConfig(regression_config_dict),
    ]

    label_metadata_dicts = dghl_loader._process_node_labels(label_configs, nodes_df, "test_ntype")

    # For multi-task node classification and regression, each task should
    # have its own masks, and entries for each label column
    assert label_metadata_dicts.keys() == {
        *class_mask_names,
        *reg_mask_names,
        NUM_LABEL_COL,
        STR_LABEL_COL,
    }

    # Check classification mask correctness
    train_mask_df, val_mask_df, test_mask_df = read_masks_from_disk(
        spark,
        dghl_loader,
        label_metadata_dicts,
        mask_names=class_mask_names,
    )
    ensure_masks_are_correct(
        train_mask_df,
        val_mask_df,
        test_mask_df,
        NUM_DATAPOINTS,
        class_split_rates,
        class_mask_names,
    )
    # Check regression mask correctness
    train_mask_df, val_mask_df, test_mask_df = read_masks_from_disk(
        spark,
        dghl_loader,
        label_metadata_dicts,
        mask_names=reg_mask_names,
    )
    ensure_masks_are_correct(
        train_mask_df,
        val_mask_df,
        test_mask_df,
        NUM_DATAPOINTS,
        reg_split_rates,
        reg_mask_names,
    )


def test_update_label_properties_classification(
    user_df: DataFrame, dghl_loader: DistHeterogeneousGraphLoader
):
    """Test updating label properties for classification"""
    node_type = "user"

    classification_config = {
        "column": "gender",
        "type": "classification",
        "split_rate": {"train": 0.8, "val": 0.2, "test": 0.0},
    }

    classification_label_config = NodeLabelConfig(classification_config)
    dghl_loader._update_label_properties(node_type, user_df, classification_label_config)

    result_properties = dghl_loader.label_properties

    assert list(result_properties.keys()) == [node_type]

    user_properties = result_properties[node_type]

    assert user_properties[COLUMN_NAME] == "gender"

    assert user_properties[VALUE_COUNTS] == {"male": 3, "female": 1, None: 1}


def test_update_label_properties_regression(
    user_df: DataFrame, dghl_loader: DistHeterogeneousGraphLoader
):
    """Test updating label properties for regression"""
    node_type = "user"

    regression_config = {
        "column": "age",
        "type": "regression",
        "split_rate": {"train": 0.8, "val": 0.2, "test": 0.0},
    }

    regression_label_config = NodeLabelConfig(regression_config)

    dghl_loader._update_label_properties(node_type, user_df, regression_label_config)

    result_properties = dghl_loader.label_properties

    assert list(result_properties.keys()) == [node_type]

    user_properties = result_properties[node_type]

    assert user_properties[COLUMN_NAME] == "age"
    assert user_properties[MIN_VALUE] == 22
    assert user_properties[MAX_VALUE] == 33


def test_update_label_properties_multilabel(
    user_df: DataFrame, dghl_loader: DistHeterogeneousGraphLoader
):
    """Test updating label properties for multi-label classification"""
    node_type = "user"

    multilabel_config = {
        "column": "multi",
        "type": "classification",
        "split_rate": {"train": 0.8, "val": 0.2, "test": 0.0},
        "separator": "|",
    }

    multilabel_label_config = NodeLabelConfig(multilabel_config)

    dghl_loader._update_label_properties(node_type, user_df, multilabel_label_config)

    result_properties = dghl_loader.label_properties

    assert list(result_properties.keys()) == [node_type]

    user_properties = result_properties[node_type]

    assert user_properties[COLUMN_NAME] == "multi"
    assert user_properties[VALUE_COUNTS] == {str(i): 1 for i in range(1, 11)}


def test_node_custom_label(spark, dghl_loader: DistHeterogeneousGraphLoader, tmp_path):
    """Test using custom label splits for nodes"""
    data = [(i,) for i in range(1, 11)]

    # Create DataFrame
    nodes_df = spark.createDataFrame(data, ["orig"])

    train_df = spark.createDataFrame([(i,) for i in range(1, 6)], ["mask_id"])
    val_df = spark.createDataFrame([(i,) for i in range(6, 9)], ["mask_id"])
    test_df = spark.createDataFrame([(i,) for i in range(9, 11)], ["mask_id"])

    train_df.repartition(1).write.parquet(f"{tmp_path}/train.parquet")
    val_df.repartition(1).write.parquet(f"{tmp_path}/val.parquet")
    test_df.repartition(1).write.parquet(f"{tmp_path}/test.parquet")
    config_dict = {
        "column": "orig",
        "type": "classification",
        "split_rate": {"train": 0.8, "val": 0.1, "test": 0.1},
        "custom_split_filenames": {
            "train": f"{tmp_path}/train.parquet",
            "valid": f"{tmp_path}/val.parquet",
            "test": f"{tmp_path}/test.parquet",
            "column": ["mask_id"],
        },
    }
    dghl_loader.input_prefix = ""
    label_configs = [NodeLabelConfig(config_dict)]
    label_metadata_dicts = dghl_loader._process_node_labels(label_configs, nodes_df, "orig")

    assert label_metadata_dicts.keys() == {
        "train_mask",
        "test_mask",
        "val_mask",
        "orig",
    }

    train_mask_df, val_mask_df, test_mask_df = read_masks_from_disk(
        spark, dghl_loader, label_metadata_dicts
    )

    train_total_ones = train_mask_df.agg(F.sum("train_mask")).collect()[0][0]
    val_total_ones = val_mask_df.agg(F.sum("val_mask")).collect()[0][0]
    test_total_ones = test_mask_df.agg(F.sum("test_mask")).collect()[0][0]
    assert train_total_ones == 5
    assert val_total_ones == 3
    assert test_total_ones == 2


def test_edge_custom_label(spark, dghl_loader: DistHeterogeneousGraphLoader, tmp_path):
    """Test using custom label splits for edges"""
    data = [(i, j) for i in range(1, 4) for j in range(11, 14)]
    # Create DataFrame
    edges_df = spark.createDataFrame(data, ["src_str_id", "dst_str_id"])

    train_df = spark.createDataFrame(
        [(i, j) for i in range(1, 2) for j in range(11, 14)],
        ["mask_src_id", "mask_dst_id"],
    )
    val_df = spark.createDataFrame(
        [(i, j) for i in range(2, 3) for j in range(11, 14)],
        ["mask_src_id", "mask_dst_id"],
    )
    test_df = spark.createDataFrame(
        [(i, j) for i in range(3, 4) for j in range(11, 14)],
        ["mask_src_id", "mask_dst_id"],
    )

    train_df.repartition(1).write.parquet(f"{tmp_path}/train.parquet")
    val_df.repartition(1).write.parquet(f"{tmp_path}/val.parquet")
    test_df.repartition(1).write.parquet(f"{tmp_path}/test.parquet")
    config_dict = {
        "column": "",
        "type": "link_prediction",
        "custom_split_filenames": {
            "train": f"{tmp_path}/train.parquet",
            "valid": f"{tmp_path}/val.parquet",
            "test": f"{tmp_path}/test.parquet",
            "column": ["mask_src_id", "mask_dst_id"],
        },
    }
    dghl_loader.input_prefix = ""
    label_configs = [EdgeLabelConfig(config_dict)]
    label_metadata_dicts = dghl_loader._process_edge_labels(
        label_configs, edges_df, "src_str_id:to:dst_str_id", ""
    )

    assert label_metadata_dicts.keys() == {"train_mask", "test_mask", "val_mask"}

    train_mask_df, val_mask_df, test_mask_df = read_masks_from_disk(
        spark, dghl_loader, label_metadata_dicts
    )

    train_total_ones = train_mask_df.agg(F.sum("train_mask")).collect()[0][0]
    val_total_ones = val_mask_df.agg(F.sum("val_mask")).collect()[0][0]
    test_total_ones = test_mask_df.agg(F.sum("test_mask")).collect()[0][0]
    assert train_total_ones == 3
    assert val_total_ones == 3
    assert test_total_ones == 3
