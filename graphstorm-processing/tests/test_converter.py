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

import pytest

from graphstorm_processing.config.config_conversion import (
    GConstructConfigConverter,
)


@pytest.fixture(name="converter")
def fixture_create_converter() -> GConstructConfigConverter:
    """Creates a new converter object for each test."""
    yield GConstructConfigConverter()


@pytest.fixture(name="node_dict")
def create_node_dict() -> dict:
    """Creates a node dictionary for each test."""
    text_input: dict[str, list[dict]] = {"nodes": [{}]}
    # nodes only with required elements
    text_input["nodes"][0] = {
        "node_type": "author",
        "format": {"name": "parquet", "separator": ","},
        "files": "/tmp/acm_raw/nodes/author.parquet",
        "node_id_col": "node_id",
    }
    return text_input


@pytest.mark.parametrize("wildcard", ["*", "?"])
def test_try_read_file_with_wildcard(
    converter: GConstructConfigConverter, node_dict: dict, wildcard
):
    """We don't currently support wildcards in filenames, so should error out."""
    node_dict["nodes"][0]["files"] = f"/tmp/acm_raw/nodes/author{wildcard}.parquet"

    with pytest.raises(ValueError):
        _ = converter.convert_nodes(node_dict["nodes"])


def test_try_read_unsupported_feature(converter: GConstructConfigConverter, node_dict: dict):
    """We should test about giving unknown feature transformation type."""
    node_dict["nodes"][0]["features"] = [
        {
            "feature_col": ["paper_title"],
            "transform": {"name": "unknown"},
        }
    ]

    with pytest.raises(ValueError):
        _ = converter.convert_nodes(node_dict["nodes"])


def test_try_read_invalid_gconstruct_config(converter: GConstructConfigConverter, node_dict: dict):
    """Custom Split Columns"""
    node_dict["nodes"][0]["labels"] = [
        {
            "label_col": "label",
            "task_type": "classification",
            "custom_split_filenames": {
                "column": ["src", "dst", "inter"],
            },
            "label_stats_type": "frequency_cnt",
        }
    ]

    with pytest.raises(AssertionError):
        _ = converter.convert_nodes(node_dict["nodes"])

    """Feature Name must exist for multiple feature columns"""
    node_dict["nodes"][0]["features"] = [{"feature_col": ["feature_1", "feature_2"]}]

    with pytest.raises(AssertionError):
        _ = converter.convert_nodes(node_dict["nodes"])

    """Unsupported output dtype"""
    node_dict["nodes"][0]["features"] = [{"feature_col": ["feature_1"], "out_dtype": "float16"}]

    with pytest.raises(AssertionError):
        _ = converter.convert_nodes(node_dict["nodes"])

    """Unsupported format type"""
    node_dict["nodes"][0]["format"] = {"name": "txt", "separator": ","}

    with pytest.raises(AssertionError):
        _ = converter.convert_nodes(node_dict["nodes"])


def test_try_read_multi_task_gconstruct_config(
    converter: GConstructConfigConverter, node_dict: dict
):
    """Check unsupported mask column"""
    node_dict["nodes"][0]["labels"] = [
        {"label_col": "label", "task_type": "classification", "mask_field_names": "train_mask"}
    ]

    with pytest.raises(AssertionError):
        _ = converter.convert_nodes(node_dict["nodes"])


@pytest.mark.parametrize("transform", ["max_min_norm", "rank_gauss"])
@pytest.mark.parametrize("out_dtype", ["float16", "float32", "float64"])
def test_try_convert_out_dtype(
    converter: GConstructConfigConverter, node_dict: dict, transform: str, out_dtype: str
):
    node_dict["nodes"][0]["features"] = [
        {
            "feature_col": ["paper_title"],
            "transform": {"name": transform, "out_dtype": out_dtype},
        }
    ]

    normalizer_dict = {"max_min_norm": "min-max", "rank_gauss": "rank-gauss"}
    res = converter.convert_nodes(node_dict["nodes"])[0]
    if out_dtype == "float32":
        assert res.features == [
            {
                "column": "paper_title",
                "transformation": {
                    "kwargs": {
                        "imputer": "none",
                        "normalizer": normalizer_dict[transform],
                        "out_dtype": "float32",
                    },
                    "name": "numerical",
                },
            }
        ]
    elif out_dtype == "float64":
        assert res.features == [
            {
                "column": "paper_title",
                "transformation": {
                    "kwargs": {
                        "imputer": "none",
                        "normalizer": normalizer_dict[transform],
                        "out_dtype": "float64",
                    },
                    "name": "numerical",
                },
            }
        ]
    elif out_dtype == "float16":
        assert res.features == [
            {
                "column": "paper_title",
                "transformation": {
                    "kwargs": {"imputer": "none", "normalizer": normalizer_dict[transform]},
                    "name": "numerical",
                },
            }
        ]


@pytest.mark.parametrize("col_name", ["citation_time", ["citation_time"]])
def test_read_node_gconstruct(converter: GConstructConfigConverter, node_dict: dict, col_name: str):
    """Multiple test cases for GConstruct node conversion"""
    # test case with only necessary components
    node_config = converter.convert_nodes(node_dict["nodes"])[0]
    assert len(converter.convert_nodes(node_dict["nodes"])) == 1
    assert node_config.node_type == "author"
    assert node_config.file_format == "parquet"
    assert node_config.files == ["/tmp/acm_raw/nodes/author.parquet"]
    assert node_config.separator == ","
    assert node_config.column == "node_id"
    assert node_config.features is None
    assert node_config.labels is None

    node_dict["nodes"].append(
        {
            "node_type": "paper",
            "format": {"name": "parquet"},
            "files": ["/tmp/acm_raw/nodes/paper.parquet"],
            "node_id_col": "node_id",
            "features": [{"feature_col": col_name, "feature_name": "feat"}],
            "labels": [
                {"label_col": "label", "task_type": "classification", "split_pct": [0.8, 0.1, 0.1]}
            ],
        }
    )

    # nodes with all elements
    # [self.type, self.format, self.files, self.separator, self.column, self.features, self.labels]
    node_config = converter.convert_nodes(node_dict["nodes"])[1]
    assert len(converter.convert_nodes(node_dict["nodes"])) == 2
    assert node_config.node_type == "paper"
    assert node_config.file_format == "parquet"
    assert node_config.files == ["/tmp/acm_raw/nodes/paper.parquet"]
    assert node_config.separator is None
    assert node_config.column == "node_id"
    assert node_config.features == [
        {"column": "citation_time", "transformation": {"name": "no-op"}, "name": "feat"}
    ]
    assert node_config.labels == [
        {
            "column": "label",
            "type": "classification",
            "split_rate": {"train": 0.8, "val": 0.1, "test": 0.1},
        }
    ]

    node_dict["nodes"].append(
        {
            "node_type": "paper_custom",
            "format": {"name": "parquet"},
            "files": ["/tmp/acm_raw/nodes/paper_custom.parquet"],
            "node_id_col": "node_id",
            "labels": [
                {
                    "label_col": "label",
                    "task_type": "classification",
                    "custom_split_filenames": {
                        "train": "customized_label/node_train_idx.parquet",
                        "valid": "customized_label/node_val_idx.parquet",
                        "test": "customized_label/node_test_idx.parquet",
                        "column": ["ID"],
                    },
                    "label_stats_type": "frequency_cnt",
                }
            ],
        }
    )

    # nodes with all elements
    # [self.type, self.format, self.files, self.separator, self.column, self.features, self.labels]
    node_config = converter.convert_nodes(node_dict["nodes"])[2]
    assert len(converter.convert_nodes(node_dict["nodes"])) == 3
    assert node_config.node_type == "paper_custom"
    assert node_config.file_format == "parquet"
    assert node_config.files == ["/tmp/acm_raw/nodes/paper_custom.parquet"]
    assert node_config.separator is None
    assert node_config.column == "node_id"
    assert node_config.labels == [
        {
            "column": "label",
            "type": "classification",
            "custom_split_filenames": {
                "train": "customized_label/node_train_idx.parquet",
                "valid": "customized_label/node_val_idx.parquet",
                "test": "customized_label/node_test_idx.parquet",
                "column": ["ID"],
            },
        }
    ]


@pytest.mark.parametrize("col_name", ["author", ["author"]])
def test_read_edge_gconstruct(converter: GConstructConfigConverter, col_name):
    """Multiple test cases for GConstruct edges conversion"""
    text_input: dict[str, list[dict]] = {"edges": [{}]}
    # nodes only with required elements
    text_input["edges"][0] = {
        "relation": ["author", "writing", "paper"],
        "format": {"name": "parquet"},
        "files": "/tmp/acm_raw/edges/author_writing_paper.parquet",
        "source_id_col": "~from",
        "dest_id_col": "~to",
    }
    # Test with only required attributes
    # [self.source_col, self.source_type, self.dest_col, self.dest_type,
    #  self.format, self.files, self.separator, self.relation, self.features, self.labels]
    edge_config = converter.convert_edges(text_input["edges"])[0]
    assert len(converter.convert_edges(text_input["edges"])) == 1
    assert edge_config.source_col == "~from"
    assert edge_config.source_type == "author"
    assert edge_config.dest_col == "~to"
    assert edge_config.dest_type == "paper"
    assert edge_config.file_format == "parquet"
    assert edge_config.files == ["/tmp/acm_raw/edges/author_writing_paper.parquet"]
    assert edge_config.separator is None
    assert edge_config.relation == "writing"
    assert edge_config.features is None
    assert edge_config.labels is None

    # Test with all attributes available
    text_input["edges"].append(
        {
            "relation": ["author", "writing", "paper"],
            "format": {"name": "parquet"},
            "files": ["/tmp/acm_raw/edges/author_writing_paper.parquet"],
            "source_id_col": "~from",
            "dest_id_col": "~to",
            "features": [{"feature_col": col_name, "feature_name": "feat"}],
            "labels": [
                {
                    "label_col": "edge_col",
                    "task_type": "classification",
                    "split_pct": [0.8, 0.2, 0.0],
                },
                {
                    "label_col": "edge_col2",
                    "task_type": "classification",
                    "split_pct": [0.9, 0.1, 0.0],
                },
            ],
        }
    )

    edge_config = converter.convert_edges(text_input["edges"])[1]
    assert len(converter.convert_edges(text_input["edges"])) == 2
    assert edge_config.source_col == "~from"
    assert edge_config.source_type == "author"
    assert edge_config.dest_col == "~to"
    assert edge_config.dest_type == "paper"
    assert edge_config.file_format == "parquet"
    assert edge_config.files == ["/tmp/acm_raw/edges/author_writing_paper.parquet"]
    assert edge_config.separator is None
    assert edge_config.relation == "writing"
    assert edge_config.features == [
        {"column": "author", "transformation": {"name": "no-op"}, "name": "feat"}
    ]
    assert edge_config.labels == [
        {
            "column": "edge_col",
            "type": "classification",
            "split_rate": {"train": 0.8, "val": 0.2, "test": 0.0},
        },
        {
            "column": "edge_col2",
            "type": "classification",
            "split_rate": {"train": 0.9, "val": 0.1, "test": 0.0},
        },
    ]


def test_convert_gsprocessing(converter: GConstructConfigConverter):
    """Multiple test cases for end2end GConstruct-to-GSProcessing conversion"""
    # test empty
    assert converter.convert_to_gsprocessing({}) == {
        "version": "gsprocessing-v1.0",
        "graph": {"nodes": [], "edges": []},
    }

    gcons_conf = {}
    gcons_conf["nodes"] = [
        {
            "node_type": "paper",
            "format": {"name": "parquet"},
            "files": ["/tmp/acm_raw/nodes/paper.parquet"],
            "separator": ",",
            "node_id_col": "node_id",
            "features": [
                {"feature_col": ["citation_time"], "feature_name": "feat"},
                {"feature_col": ["num_citations"], "transform": {"name": "max_min_norm"}},
                {
                    "feature_col": ["num_citations"],
                    "transform": {
                        "name": "bucket_numerical",
                        "bucket_cnt": 9,
                        "range": [10, 100],
                        "slide_window_size": 5,
                    },
                },
                {
                    "feature_col": ["num_citations"],
                    "feature_name": "rank_gauss1",
                    "transform": {"name": "rank_gauss"},
                },
                {
                    "feature_col": ["num_citations"],
                    "feature_name": "rank_gauss2",
                    "transform": {"name": "rank_gauss", "epsilon": 0.1},
                },
                {
                    "feature_col": ["num_citations"],
                    "transform": {"name": "to_categorical", "mapping": {"1", "2", "3"}},
                },
                {
                    "feature_col": ["num_citations"],
                    "transform": {"name": "to_categorical", "separator": ","},
                },
                {
                    "feature_col": ["citation_name"],
                    "transform": {
                        "name": "tokenize_hf",
                        "bert_model": "bert",
                        "max_seq_length": 64,
                    },
                },
                {
                    "feature_col": ["citation_name"],
                    "transform": {
                        "name": "bert_hf",
                        "bert_model": "bert",
                        "max_seq_length": 64,
                    },
                },
            ],
            "labels": [
                {"label_col": "label", "task_type": "classification", "split_pct": [0.8, 0.1, 0.1]}
            ],
        }
    ]
    gcons_conf["edges"] = [
        {
            "relation": ["author", "writing", "paper"],
            "format": {"name": "parquet"},
            "files": ["/tmp/acm_raw/edges/author_writing_paper.parquet"],
            "source_id_col": "~from",
            "dest_id_col": "~to",
            "features": [{"feature_col": ["author"], "feature_name": "feat"}],
            "labels": [
                {
                    "label_col": "edge_col",
                    "task_type": "classification",
                    "split_pct": [0.8, 0.2, 0.0],
                },
                {
                    "label_col": "edge_col2",
                    "task_type": "classification",
                    "split_pct": [0.9, 0.1, 0.0],
                },
            ],
        }
    ]

    assert len(converter.convert_to_gsprocessing(gcons_conf)["graph"]["nodes"]) == 1
    nodes_output = converter.convert_to_gsprocessing(gcons_conf)["graph"]["nodes"][0]
    assert nodes_output["data"]["format"] == "parquet"
    assert nodes_output["data"]["files"] == ["/tmp/acm_raw/nodes/paper.parquet"]
    assert nodes_output["type"] == "paper"
    assert nodes_output["column"] == "node_id"
    assert nodes_output["features"] == [
        {"column": "citation_time", "transformation": {"name": "no-op"}, "name": "feat"},
        {
            "column": "num_citations",
            "transformation": {
                "name": "numerical",
                "kwargs": {"normalizer": "min-max", "imputer": "none"},
            },
        },
        {
            "column": "num_citations",
            "transformation": {
                "name": "bucket-numerical",
                "kwargs": {
                    "bucket_cnt": 9,
                    "range": [10, 100],
                    "slide_window_size": 5,
                    "imputer": "none",
                },
            },
        },
        {
            "column": "num_citations",
            "name": "rank_gauss1",
            "transformation": {
                "name": "numerical",
                "kwargs": {"normalizer": "rank-gauss", "imputer": "none"},
            },
        },
        {
            "column": "num_citations",
            "name": "rank_gauss2",
            "transformation": {
                "name": "numerical",
                "kwargs": {"epsilon": 0.1, "normalizer": "rank-gauss", "imputer": "none"},
            },
        },
        {
            "column": "num_citations",
            "transformation": {
                "name": "categorical",
                "kwargs": {},
            },
        },
        {
            "column": "num_citations",
            "transformation": {
                "name": "multi-categorical",
                "kwargs": {"separator": ","},
            },
        },
        {
            "column": "citation_name",
            "transformation": {
                "name": "huggingface",
                "kwargs": {"action": "tokenize_hf", "hf_model": "bert", "max_seq_length": 64},
            },
        },
        {
            "column": "citation_name",
            "transformation": {
                "name": "huggingface",
                "kwargs": {"action": "embedding_hf", "hf_model": "bert", "max_seq_length": 64},
            },
        },
    ]
    assert nodes_output["labels"] == [
        {
            "column": "label",
            "type": "classification",
            "split_rate": {"train": 0.8, "val": 0.1, "test": 0.1},
        }
    ]

    assert len(converter.convert_to_gsprocessing(gcons_conf)["graph"]["edges"]) == 1
    edges_output = converter.convert_to_gsprocessing(gcons_conf)["graph"]["edges"][0]
    assert edges_output["data"]["format"] == "parquet"
    assert edges_output["data"]["files"] == ["/tmp/acm_raw/edges/author_writing_paper.parquet"]
    assert edges_output["source"] == {"column": "~from", "type": "author"}
    assert edges_output["dest"] == {"column": "~to", "type": "paper"}
    assert edges_output["relation"] == {"type": "writing"}
    assert edges_output["features"] == [
        {"column": "author", "transformation": {"name": "no-op"}, "name": "feat"}
    ]
    assert edges_output["labels"] == [
        {
            "column": "edge_col",
            "type": "classification",
            "split_rate": {"train": 0.8, "val": 0.2, "test": 0.0},
        },
        {
            "column": "edge_col2",
            "type": "classification",
            "split_rate": {"train": 0.9, "val": 0.1, "test": 0.0},
        },
    ]
