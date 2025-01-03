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

from graphstorm_processing.config.config_parser import (
    NodeConfig,
    EdgeConfig,
    create_config_objects,
    parse_feat_config,
)
from graphstorm_processing.config.numerical_configs import (
    NumericalFeatureConfig,
)
from graphstorm_processing.config.label_config_base import (
    EdgeLabelConfig,
    NodeLabelConfig,
)


def test_parse_edge_lp_config():
    """Test parsing an edge configuration with link prediction task"""
    label_config_dict = [
        {
            "column": "",
            "type": "link_prediction",
            "split_rate": {"train": 0.8, "val": 0.1, "test": 0.1},
        }
    ]
    edge_dict = {
        "source": {"column": "~from", "type": "movie"},
        "relation": {"type": "included_in"},
        "dest": {"column": "~to", "type": "genre"},
        "labels": label_config_dict,
    }
    data_dict = {"format": "csv", "files": ["edges/movie-included_in-genre.csv"], "separator": ","}
    edge_config = EdgeConfig(edge_dict, data_dict)

    assert edge_config.src_ntype == "movie"
    assert edge_config.dst_ntype == "genre"
    assert edge_config.rel_type == "included_in"
    assert edge_config.src_col == "~from"
    assert edge_config.dst_col == "~to"
    assert edge_config.rel_col is None
    assert edge_config.format == "csv"
    assert edge_config.files == ["edges/movie-included_in-genre.csv"]
    assert edge_config.separator == ","
    assert edge_config.label_configs
    assert len(edge_config.label_configs) == 1
    lp_config = edge_config.label_configs[0]
    assert isinstance(lp_config, EdgeLabelConfig)
    assert lp_config.task_type == "link_prediction"
    assert lp_config.split_rate == {"train": 0.8, "val": 0.1, "test": 0.1}


def test_parse_basic_node_config():
    """Test parsing a basic node configuration"""
    node_config = {"column": "~id", "type": "user"}
    data_config = {"format": "csv", "files": ["nodes/user.csv"], "separator": ","}
    node_config_obj = NodeConfig(node_config, data_config)

    assert node_config_obj.ntype == "user"
    assert node_config_obj.node_col == "~id"
    assert node_config_obj.format == "csv"
    assert node_config_obj.files == ["nodes/user.csv"]
    assert node_config_obj.separator == ","


def test_parse_num_configs():
    """Test parsing a numerical features configuration"""
    feature_dict = {
        "column": "age",
        "transformation": {
            "name": "numerical",
            "kwargs": {"imputer": "mean", "normalizer": "min-max"},
        },
    }
    feature_config = parse_feat_config(feature_dict)

    assert isinstance(feature_config, NumericalFeatureConfig)
    assert feature_config._cols == ["age"]
    assert feature_config.feat_type == "numerical"
    assert feature_config._transformation_kwargs["imputer"] == "mean"
    assert feature_config._transformation_kwargs["normalizer"] == "min-max"


def test_parse_node_label_configs():
    """Test parsing a node configuration with a classification label"""
    label_config_dict = {
        "column": "gender",
        "type": "classification",
        "split_rate": {"train": 0.8, "val": 0.1, "test": 0.1},
    }
    node_config_dict = {"column": "~id", "type": "user", "labels": [label_config_dict]}
    data_config = {"format": "csv", "files": ["nodes/user.csv"], "separator": ","}
    node_config_obj = NodeConfig(node_config_dict, data_config)

    assert node_config_obj.label_configs
    assert len(node_config_obj.label_configs) == 1
    label_config = node_config_obj.label_configs[0]
    assert isinstance(label_config, NodeLabelConfig)
    assert label_config.label_column == "gender"
    assert label_config.task_type == "classification"
    assert label_config.split_rate == {"train": 0.8, "val": 0.1, "test": 0.1}


def test_create_config_objects():
    """Test conversion of input dicts to config objects"""
    graph_config = {
        "edges": [
            {
                "data": {
                    "format": "csv",
                    "files": ["edges/movie-included_in-genre.csv"],
                    "separator": ",",
                },
                "source": {"column": "~from", "type": "movie"},
                "relation": {"type": "included_in"},
                "dest": {"column": "~to", "type": "genre"},
            }
        ],
        "nodes": [
            {
                "data": {"format": "csv", "files": ["nodes/genre.csv"], "separator": ","},
                "column": "~id",
                "type": "genre",
            }
        ],
    }

    config_objects = create_config_objects(graph_config)

    assert len(config_objects["edges"]) == 1
    assert len(config_objects["nodes"]) == 1
    assert isinstance(config_objects["edges"][0], EdgeConfig)
    assert isinstance(config_objects["nodes"][0], NodeConfig)


def test_unsupported_transformation():
    """Test that an unsupported transformation raises an error"""

    feature_dict = {"column": "feature", "transformation": {"name": "unsupported_transform"}}

    with pytest.raises(
        RuntimeError,
        match="Unknown transformation name: 'unsupported_transform'",
    ):
        parse_feat_config(feature_dict)
