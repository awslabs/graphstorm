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

import pytest

from graphstorm.gconstruct.config_conversion import (
    GSProcessingConfigConverter,
)


@pytest.fixture(name="converter")
def fixture_create_converter() -> GSProcessingConfigConverter:
    """Creates a new converter object for each test."""
    return GSProcessingConfigConverter()


@pytest.fixture(name="node_dict")
def create_node_dict() -> dict:
    """Creates a node dictionary for each test."""
    text_input: dict[str, list[dict]] = {"nodes": [{}]}
    # nodes only with required elements
    text_input["nodes"][0] = {
        "data": {
            "format": "parquet",
            "files": ["/tmp/acm_raw/nodes/author.parquet"],
            "separator": ","
        },
        "type": "author",
        "column": "node_id",
    }
    return text_input


def test_try_read_unsupported_feature(converter: GSProcessingConfigConverter, node_dict: dict):
    """We should test about giving unknown feature transformation type."""
    node_dict["nodes"][0]["features"] = [
        {
            "column": ["paper_title"],
            "transformation": {"name": "unknown"},
        }
    ]

    with pytest.raises(ValueError):
        _ = converter.convert_nodes(node_dict["nodes"])


def test_custom_split_config_conversion(converter: GSProcessingConfigConverter):
    """Test custom split file config conversion"""
    gsprocessing_label_dicts = [
        {
            "column": "label",
            "type": "classification",
            "custom_split_filenames": {
                "column": ["src", "dst"],
            },
            "label_stats_type": "frequency_cnt",
        }
    ]

    # Should raise when none of train/val/test are in the input
    with pytest.raises(AssertionError):
        converter._convert_label(gsprocessing_label_dicts)

    # Ensure single strings are converted to list of strings
    gsprocessing_label_dicts[0]["custom_split_filenames"]["train"] = "fake_file"
    gconstruct_label_dict = converter._convert_label(gsprocessing_label_dicts)[0]

    assert gconstruct_label_dict["custom_split_filenames"]["train"] == ["fake_file"]


@pytest.mark.parametrize("col_name", ["citation_time"])
def test_read_node_gsprocessing(converter: GSProcessingConfigConverter, node_dict: dict, col_name: str):
    """Multiple test cases for GSProcessing node conversion"""
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
            "data": {
                "format": "parquet",
                "files": ["/tmp/acm_raw/nodes/paper.parquet"]
            },
            "type": "paper",
            "column": "node_id",
            "features": [{"column": col_name, "name": "feat"}],
            "labels": [
                {"column": "label", "type": "classification",
                 "split_rate": {
                     "train": 0.8,
                     "val": 0.1,
                     "test": 0.1
                 }}
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
        {"feature_col": "citation_time", "transform": {"name": "no-op"}, "feature_name": "feat"}
    ]
    assert node_config.labels == [
        {
            "label_col": "label",
            "task_type": "classification",
            "split_pct": [0.8, 0.1, 0.1],
        }
    ]

    node_dict["nodes"].append(
        {
            "data": {
                "format": "parquet",
                "files": ["/tmp/acm_raw/nodes/paper_custom.parquet"]
            },
            "type": "paper_custom",
            "column": "node_id",
            "labels": [
                {
                    "column": "label",
                    "type": "classification",
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
            "label_col": "label",
            "task_type": "classification",
            "custom_split_filenames": {
                "train": ["customized_label/node_train_idx.parquet"],
                "valid": ["customized_label/node_val_idx.parquet"],
                "test": ["customized_label/node_test_idx.parquet"],
                "column": ["ID"],
            },
        }
    ]


@pytest.mark.parametrize("col_name", ["author"])
def test_read_edge_gconstruct(converter: GSProcessingConfigConverter, col_name):
    """Multiple test cases for GConstruct edges conversion"""
    text_input: dict[str, list[dict]] = {"edges": [{}]}
    # nodes only with required elements
    text_input["edges"][0] = {
        "data": {
            "format": "parquet",
            "files": ["/tmp/acm_raw/edges/author_writing_paper.parquet"]
        },
        "source": {
            "column": "~from",
            "type": "author"
        },
        "dest": {
            "column": "~to",
            "type": "paper"
        },
        "relation": {
            "type": "writing"
        },
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
            "data": {
                "format": "parquet",
                "files": ["/tmp/acm_raw/edges/author_writing_paper.parquet"]
            },
            "source": {
                "column": "~from",
                "type": "author"
            },
            "dest": {
                "column": "~to",
                "type": "paper"
            },
            "relation": {
                "type": "writing"
            },
            "features": [{"column": col_name, "name": "feat"}],
            "labels": [
                {
                    "column": "edge_col",
                    "type": "classification",
                    "split_rate": {
                        "train": 0.8,
                        "val": 0.2,
                        "test": 0.0
                    },
                },
                {
                    "column": "edge_col2",
                    "type": "classification",
                    "split_rate": {
                        "train": 0.9,
                        "val": 0.1,
                        "test": 0.0
                    },
                },
            ]
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
        {"feature_col": "author", "transform": {"name": "no-op"}, "feature_name": "feat"}
    ]
    assert edge_config.labels == [
        {
            "label_col": "edge_col",
            "task_type": "classification",
            "split_pct": [0.8, 0.2, 0.0],
        },
        {
            "label_col": "edge_col2",
            "task_type": "classification",
            "split_pct": [0.9, 0.1, 0.0]
        },
    ]


def test_convert_gsprocessing_config(converter: GSProcessingConfigConverter):
    """Multiple test cases for end2end GSProcessing-to-GConstruct conversion"""
    # test empty
    assert converter.convert_to_gconstruct({}) == {
        "version": "gconstruct-v0.1",
        "nodes": [],
        "edges": [],
    }

    gsp_conf = {}
    gsp_conf["version"] = "gsprocessing-v0.4.1"
    gsp_conf["graph"] = {}
    gsp_conf["graph"]["nodes"] = [
        {
            "data": {
                "format": "parquet",
                "files": ["/tmp/acm_raw/nodes/paper.parquet"],
                "separator": ","
            },
            "type": "paper",
            "column": "node_id",
            "features": [
                {"column": "citation_time", "name": "feat"},
                {"column": "num_citations",
                 "transformation": {"name": "numerical", "kwargs": {"normalizer": "min-max", "imputer": "mean"}}},
                {
                    "column": "num_citations",
                    "transformation": {
                        "name": "bucket_numerical",
                        "kwargs":{
                            "bucket_cnt": 9,
                            "range": [10, 100],
                            "slide_window_size": 5,
                            "imputer": "mean"
                        }
                    },
                },
                {
                    "column": "num_citations",
                    "name": "rank_gauss1",
                    "transformation": {"name": "numerical", "kwargs":
                        {"normalizer": "rank-gauss", "imputer": "mean"}
                    },
                },
                {
                    "column": "num_citations",
                    "name": "rank_gauss2",
                    "transformation": {"name": "numerical", "kwargs":
                        {"normalizer": "rank-gauss", "epsilon": 0.1,
                         "imputer": "mean"}
                    },
                },
                {
                    "column": "num_citations",
                    "transformation": {"name": "categorical", "kwargs": {}},
                },
                {
                    "column": "num_citations",
                    "transformation": {"name": "multi-categorical", "kwargs": {"separator": ","}},
                },
                {
                    "column": "citation_name",
                    "transformation": {
                        "name": "huggingface",
                        "kwargs": {
                            "action": "tokenize_hf",
                            "hf_model": "bert",
                            "max_seq_length": 64
                        }
                    },
                },
                {
                    "column": "citation_name",
                    "transformation": {
                        "name": "huggingface",
                        "kwargs": {
                            "action": "bert_hf",
                            "hf_model": "bert",
                            "max_seq_length": 64
                        }
                    },
                },
            ],
            "labels": [
                {"column": "label", "type": "classification", "split_rate": {"train": 0.8, "val": 0.1, "test": 0.1}}
            ],
        }
    ]
    gsp_conf["graph"]["edges"] = [
        {
            "data": {
                "format": "parquet",
                "files": ["/tmp/acm_raw/edges/author_writing_paper.parquet"]
            },
            "source": {
                "column": "~from",
                "type": "author"
            },
            "dest": {
                "column": "~to",
                "type": "paper"
            },
            "relation": {
                "type": "writing"
            },
            "features": [
                {"column": "author", "name": "feat"},
                {
                    "column": "author",
                    "name": "hard_negative",
                    "transformation": {"name": "edge_dst_hard_negative",
                                       "kwargs": {"separator": ";"}},
                },
                {
                    "column": "num_feature",
                    "transformation": {"name": "numerical",
                                       "kwargs": {"normalizer": "standard", "imputer": "mean"}},
                },
            ],
            "labels": [
                {
                    "column": "edge_col",
                    "type": "classification",
                    "split_rate": {
                        "train": 0.8,
                        "val": 0.2,
                        "test": 0.0
                    },
                },
                {
                    "column": "edge_col2",
                    "type": "classification",
                    "split_rate": {
                        "train": 0.9,
                        "val": 0.1,
                        "test": 0.0
                    },
                },
            ],
        }
    ]

    assert len(converter.convert_to_gconstruct(gsp_conf["graph"])["nodes"]) == 1
    nodes_output = converter.convert_to_gconstruct(gsp_conf["graph"])["nodes"][0]
    assert nodes_output["format"] == {"name": "parquet"}
    assert nodes_output["files"] == ["/tmp/acm_raw/nodes/paper.parquet"]
    assert nodes_output["node_type"] == "paper"
    assert nodes_output["node_id_col"] == "node_id"
    assert nodes_output["features"] == [
        {"feature_col": "citation_time", "transform": {"name": "no-op"}, "feature_name": "feat"},
        {"feature_col": "num_citations", "transform": {"name": "max_min_norm"}},
        {
            "feature_col": "num_citations",
            "transform": {
                "name": "bucket_numerical",
                "bucket_cnt": 9,
                "range": [10, 100],
                "slide_window_size": 5,
            },
        },
        {
            "feature_col": "num_citations",
            "feature_name": "rank_gauss1",
            "transform": {"name": "rank_gauss"},
        },
        {
            "feature_col": "num_citations",
            "feature_name": "rank_gauss2",
            "transform": {"name": "rank_gauss", "epsilon": 0.1},
        },
        {
            "feature_col": "num_citations",
            "transform": {"name": "to_categorical"},
        },
        {
            "feature_col": "num_citations",
            "transform": {"name": "to_categorical", "separator": ","},
        },
        {
            "feature_col": "citation_name",
            "transform": {
                "name": "tokenize_hf",
                "bert_model": "bert",
                "max_seq_length": 64,
            },
        },
        {
            "feature_col": "citation_name",
            "transform": {
                "name": "bert_hf",
                "bert_model": "bert",
                "max_seq_length": 64,
            },
        }
    ]
    assert nodes_output["labels"] == [
        {"label_col": "label", "task_type": "classification", "split_pct": [0.8, 0.1, 0.1]}
    ]

    assert len(converter.convert_to_gconstruct(gsp_conf["graph"])["edges"]) == 1
    edges_output = converter.convert_to_gconstruct(gsp_conf["graph"])["edges"][0]
    assert edges_output["format"] == {"name": "parquet"}
    assert edges_output["files"] == ["/tmp/acm_raw/edges/author_writing_paper.parquet"]
    assert edges_output["relation"] == ["author", "writing", "paper"]
    assert edges_output["source_id_col"] == "~from"
    assert edges_output["dest_id_col"] == "~to"
    assert edges_output["features"] == [
        {"feature_col": "author", "feature_name": "feat", "transform": {"name": "no-op"}},
        {
            "feature_col": "author",
            "feature_name": "hard_negative",
            "transform": {"name": "edge_dst_hard_negative", "separator": ";"},
        },
        {
            "feature_col": "num_feature",
            "transform": {"name": "standard"},
        },
    ]
    assert edges_output["labels"] == [
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
    ]