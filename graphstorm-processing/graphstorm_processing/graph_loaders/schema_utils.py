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

This module is used to parse the schema of CSV input files, inferring
the type of the columns from the type mentioned in the configuration.
"""

import logging
from typing import Sequence, List, Type

from pyspark.sql.types import StructType, StructField, StringType, DataType, DoubleType

from ..config.config_parser import EdgeConfig, NodeConfig
from ..config.label_config_base import LabelConfig
from ..config.feature_config_base import FeatureConfig


def parse_edge_file_schema(edge_config: EdgeConfig) -> StructType:
    """Parses schema of CSV edge file according to edge configuration.

    Parameters
    ----------
    edge_config : EdgeConfig
        The configuration object for the edge type.

    Returns
    -------
    StructType
        A Spark StructType describing the typed schema for the edge files.
    """
    # TODO: Will need to add support for relation types when we support relation col
    edge_fields_list = [
        StructField(edge_config.src_col, StringType(), False),
        StructField(edge_config.dst_col, StringType(), False),
    ]

    if edge_config.feature_configs:
        features_schema_list = _parse_features_schema(edge_config.feature_configs)
        edge_fields_list.extend(features_schema_list)
    if edge_config.label_configs:
        labels_schema_list = _parse_edge_labels_schema(edge_config.label_configs)
        edge_fields_list.extend(labels_schema_list)

    return StructType(edge_fields_list)


def _parse_features_schema(features_objects: Sequence[FeatureConfig]) -> Sequence[StructField]:
    field_list = []
    for feature_config in features_objects:
        feature_type = feature_config.feat_type
        for feature_col, _ in zip(feature_config.cols, feature_config.feat_name):
            spark_feature_type = determine_spark_feature_type(feature_type)
            if StructField(feature_col, spark_feature_type(), True) in field_list:
                continue
            field_list.append(StructField(feature_col, spark_feature_type(), True))

    return field_list


def determine_spark_feature_type(feature_type: str) -> Type[DataType]:
    """Returns a DataType class, depending on the type of feature provided.

    Parameters
    ----------
    feature_type : str
        Name of the feature type, e.g. 'numerical' or 'no-op'

    Returns
    -------
    Type[DataType]
        Corresponding Spark type class.

    Raises
    ------
    NotImplementedError
        In case an unsupported feature_type is provided.
    """
    # TODO: Replace with pattern matching after moving to Python 3.10?
    if feature_type in [
        "no-op",
        "multi-numerical",
        "categorical",
        "multi-categorical",
        "huggingface",
    ] or feature_type.startswith("text"):
        return StringType
    if feature_type in ["numerical", "bucket-numerical"]:
        return DoubleType
    else:
        raise NotImplementedError(f"Unknown feature type: {feature_type}")


def _parse_edge_labels_schema(edge_labels_objects: Sequence[LabelConfig]) -> Sequence[StructField]:
    field_list = []

    for label_config in edge_labels_objects:
        label_col = label_config.label_column
        target_task_type = label_config.task_type

        if target_task_type == "classification":
            field_list.append(StructField(label_col, StringType(), True))
        elif target_task_type == "regression":
            field_list.append(StructField(label_col, DoubleType(), True))
        elif target_task_type == "link_prediction" and label_col:
            logging.info(
                "Bypassing edge label %s, as it is only used for link prediction", label_col
            )

    return field_list


def parse_node_file_schema(node_config: NodeConfig) -> StructType:
    """Parses schema of CSV node file according to node configuration.

    Parameters
    ----------
    node_config : NodeConfig
        The configuration object for the node type.

    Returns
    -------
    StructType
        A Spark StructType describing the typed schema for the node files.
    """
    node_id_col = node_config.node_col

    node_field_list = [StructField(node_id_col, StringType(), False)]

    if node_config.feature_configs:
        features_schema_list = _parse_features_schema(node_config.feature_configs)
        node_field_list.extend(features_schema_list)
    if node_config.label_configs:
        labels_schema_list = _parse_node_labels_schema(node_config.label_configs)
        node_field_list.extend(labels_schema_list)

    return StructType(node_field_list)


def _parse_node_labels_schema(node_labels_objects: List[LabelConfig]) -> Sequence[StructField]:
    field_list = []

    for label_config in node_labels_objects:
        label_col = label_config.label_column
        target_task_type = label_config.task_type

        if target_task_type == "classification":
            # TODO: Are we certain all classification labels will be strings?
            # Could be ints, would that be an issue?
            field_list.append(StructField(label_col, StringType(), True))
        elif target_task_type == "regression":
            field_list.append(StructField(label_col, DoubleType(), True))

    return field_list
